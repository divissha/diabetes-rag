"""
Builds a silver retrieval eval set: one LLM-written question per sampled
paper, with that paper's PMID as the gold answer.
Resumable: re-running continues where it stopped.
"""

import json
import os
import random
import time

from llm import call_llm

PAPERS_PATH = "data/raw/diabetes_papers.json"
OUT_PATH = "data/eval/retrieval_set.json"
PER_CLASS = 15
SEED = 42
SLEEP_SECONDS = 8  # keeps us under Groq's tokens-per-minute cap

PROMPT = """You are writing test questions for a medical literature search system.
Read the abstract below and write ONE question a clinician or researcher might ask
whose answer is found in this paper.

Rules:
- One sentence, phrased generally.
- Do NOT mention "this study", "the paper", authors, or the abstract.
- Do NOT copy long phrases from the abstract; paraphrase the concepts.
- Output only the question.

Abstract:
{abstract}"""


def sample_papers(papers):
    rng = random.Random(SEED)
    chosen, seen = [], set()
    for drug_class in ["metformin", "glp1", "sglt2", "insulin"]:
        pool = [p for p in papers
                if p.get(f"in_{drug_class}") and len(p["abstract"].split()) >= 120]
        rng.shuffle(pool)
        added = 0
        for p in pool:
            if p["pmid"] not in seen and added < PER_CLASS:
                chosen.append(p)
                seen.add(p["pmid"])
                added += 1
    return chosen


def main():
    os.makedirs("data/eval", exist_ok=True)
    papers = json.load(open(PAPERS_PATH))
    targets = sample_papers(papers)

    items = json.load(open(OUT_PATH)) if os.path.exists(OUT_PATH) else []
    done = {it["pmid"] for it in items}

    for i, p in enumerate(targets):
        if p["pmid"] in done:
            continue
        question = call_llm(PROMPT.format(abstract=p["abstract"]), temperature=0.3)
        question = question.split("\n")[0].strip().strip('"')
        items.append({"pmid": p["pmid"], "title": p["title"], "question": question})
        json.dump(items, open(OUT_PATH, "w"), indent=2)
        print(f"[{len(items)}/{len(targets)}] {question}")
        time.sleep(SLEEP_SECONDS)

    print(f"\nSaved {len(items)} questions to {OUT_PATH}")


if __name__ == "__main__":
    main()
