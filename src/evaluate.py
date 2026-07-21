# """
# Evaluates the RAG pipeline using ragas: faithfulness and answer relevancy,
# across a fixed set of test questions.
# """

# import json
# from datasets import Dataset
# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy
# from ragas.llms import LangchainLLMWrapper
# from ragas.embeddings import LangchainEmbeddingsWrapper
# from langchain_groq import ChatGroq
# from langchain_huggingface import HuggingFaceEmbeddings

# from generate import answer_question

# TEST_QUESTIONS_PATH = "data/eval/test_questions.json"


# def build_eval_dataset():
#     with open(TEST_QUESTIONS_PATH) as f:
#         questions = json.load(f)

#     records = {"question": [], "answer": [], "contexts": []}

#     for q in questions:
#         print(f"Answering: {q}")
#         answer, chunks = answer_question(q)
#         records["question"].append(q)
#         records["answer"].append(answer)
#         records["contexts"].append([c["text"] for c in chunks])

#     return Dataset.from_dict(records)


# def main():
#     dataset = build_eval_dataset()

#     # ragas needs an LLM to act as the "judge" for these metrics,
#     # and an embedding model for semantic similarity checks
#     judge_llm = LangchainLLMWrapper(ChatGroq(model="llama-3.1-8b-instant", temperature=0))
#     judge_embeddings = LangchainEmbeddingsWrapper(
#         HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     )

#     print("\nRunning ragas evaluation...")
#     results = evaluate(
#         dataset,
#         metrics=[faithfulness, answer_relevancy],
#         llm=judge_llm,
#         embeddings=judge_embeddings,
#     )

#     print("\n=== RESULTS ===")
#     print(results)

#     df = results.to_pandas()
#     df.to_csv("data/eval/results.csv", index=False)
#     print("\nSaved detailed results to data/eval/results.csv")


# if __name__ == "__main__":
#     main()


"""
Lightweight evaluation: uses an LLM to judge faithfulness and relevance
of RAG answers, with retry/backoff to handle Groq's free-tier rate limits.
"""

import json
import os
import time
from dotenv import load_dotenv
from groq import Groq, RateLimitError
from generate import answer_question

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

JUDGE_MODEL = "llama-3.1-8b-instant"
TEST_QUESTIONS_PATH = "data/eval/test_questions.json"

SECONDS_BETWEEN_CALLS = 3  # pace requests to stay under the per-minute token cap


def call_with_retry(prompt, max_retries=5):
    """Call Groq with automatic retry on rate limit errors."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content
        except RateLimitError as e:
            wait_time = 5 * (attempt + 1)  # backs off longer each retry
            print(f"    Rate limited, waiting {wait_time}s (attempt {attempt+1}/{max_retries})...")
            time.sleep(wait_time)
    raise Exception("Max retries exceeded due to rate limiting")


def judge_faithfulness(answer, contexts):
    context_text = "\n\n".join(contexts)
    prompt = f"""You are evaluating whether an answer is faithful to its source material.

Sources:
{context_text}

Answer to evaluate:
{answer}

Does the answer contain ONLY claims that are supported by the sources above? 
Respond with a score from 1-5 (5 = fully supported, no unsupported claims; 1 = mostly unsupported/hallucinated) 
and one sentence of reasoning. Format exactly as:
SCORE: <number>
REASON: <one sentence>"""
    return call_with_retry(prompt)


def judge_relevance(question, answer):
    prompt = f"""Question: {question}

Answer: {answer}

Does this answer directly address the question asked? 
Respond with a score from 1-5 (5 = fully addresses it; 1 = off-topic/evasive) 
and one sentence of reasoning. Format exactly as:
SCORE: <number>
REASON: <one sentence>"""
    return call_with_retry(prompt)


def parse_score(judged_text):
    for line in judged_text.split("\n"):
        if line.startswith("SCORE:"):
            try:
                return int(line.replace("SCORE:", "").strip())
            except ValueError:
                return None
    return None


def main():
    with open(TEST_QUESTIONS_PATH) as f:
        questions = json.load(f)

    results = []

    for i, q in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] Evaluating: {q}")

        answer, chunks = answer_question(q)
        time.sleep(SECONDS_BETWEEN_CALLS)

        contexts = [c["text"] for c in chunks]

        faith_judgment = judge_faithfulness(answer, contexts)
        time.sleep(SECONDS_BETWEEN_CALLS)

        rel_judgment = judge_relevance(q, answer)
        time.sleep(SECONDS_BETWEEN_CALLS)

        faith_score = parse_score(faith_judgment)
        rel_score = parse_score(rel_judgment)

        print(f"  Faithfulness: {faith_score}/5")
        print(f"  Relevance: {rel_score}/5")

        results.append({
            "question": q,
            "answer": answer,
            "faithfulness_score": faith_score,
            "faithfulness_reason": faith_judgment,
            "relevance_score": rel_score,
            "relevance_reason": rel_judgment,
        })

    os.makedirs("data/eval", exist_ok=True)
    with open("data/eval/results.json", "w") as f:
        json.dump(results, f, indent=2)

    valid_faith = [r["faithfulness_score"] for r in results if r["faithfulness_score"]]
    valid_rel = [r["relevance_score"] for r in results if r["relevance_score"]]

    avg_faith = sum(valid_faith) / len(valid_faith) if valid_faith else 0
    avg_rel = sum(valid_rel) / len(valid_rel) if valid_rel else 0

    print(f"\n=== SUMMARY ===")
    print(f"Average faithfulness: {avg_faith:.2f}/5")
    print(f"Average relevance: {avg_rel:.2f}/5")
    print(f"\nDetailed results saved to data/eval/results.json")


if __name__ == "__main__":
    main()