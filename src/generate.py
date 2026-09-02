"""
Takes a question, retrieves relevant chunks, and asks an LLM to synthesize
an answer grounded in those chunks.
"""

import os
from dotenv import load_dotenv
from groq import Groq
from retrieve import retrieve

load_dotenv()  # loads GROQ_API_KEY from .env

client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL = "openai/gpt-oss-20b"  # fast, free-tier friendly


def build_prompt(question, chunks):
    context = "\n\n".join(
        f"[Source {i+1}: {c['title']}]\n{c['text']}"
        for i, c in enumerate(chunks)
    )

    prompt = f"""You are a medical evidence assistant. Answer the question using ONLY the information in the sources below. If the sources don't fully answer the question, say so clearly. Cite sources by number (e.g. [Source 1]) when making claims.

Sources:
{context}

Question: {question}

Answer:"""
    return prompt


def answer_question(question, top_k=5):
    chunks = retrieve(question, top_k=top_k)
    prompt = build_prompt(question, chunks)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    answer = response.choices[0].message.content
    return answer, chunks


if __name__ == "__main__":
    question = "How does metformin compare to GLP-1 agonists for weight loss?"
    answer, chunks = answer_question(question)

    print("QUESTION:", question)
    print("\nANSWER:\n", answer)
    print("\n\nSOURCES USED:")
    for i, c in enumerate(chunks):
        print(f"[{i+1}] {c['title']}")
