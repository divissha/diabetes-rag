"""
Generation evaluation over a fixed question set (resumable).
Metrics: citation accuracy, faithfulness, relevance, invalid citations.
Also exports a blind hand-labelling file to measure the verifier itself.
Usage: python3 src/eval_generation.py
"""

import json
import os
import random
import re
import time

from llm import call_llm
from research import run_research

QUESTIONS_PATH = "data/eval/generation_questions.json"
RESULTS_PATH = "data/eval/generation_results.json"
LABEL_PATH = "data/eval/label_me.txt"
LABEL_KEY_PATH = "data/eval/label_key.json"
SLEEP_SECONDS = 45      # keeps us under Groq's tokens-per-minute cap
N_LABEL_EACH = 12
SEED = 7

QUESTIONS = [
    ("compare", "How do SGLT2 inhibitors and GLP-1 receptor agonists compare for cardiovascular and kidney outcomes in type 2 diabetes?"),
    ("compare", "How does metformin compare with GLP-1 receptor agonists for weight loss in type 2 diabetes?"),
    ("compare", "How does insulin therapy compare with oral medications as second-line treatment in type 2 diabetes?"),
    ("compare", "How do SGLT2 inhibitors differ from metformin in effects on heart failure and mortality?"),
    ("compare", "What evidence supports combining insulin with GLP-1 receptor agonists in type 2 diabetes?"),
    ("subgroup", "What does the evidence say about metformin safety in patients with reduced kidney function?"),
    ("subgroup", "What patient characteristics should guide the choice between SGLT2 inhibitors and GLP-1 receptor agonists?"),
    ("safety", "What are the main adverse effects of SGLT2 inhibitors compared with GLP-1 receptor agonists?"),
    ("safety", "What is the risk of lactic acidosis with metformin compared with other glucose-lowering drugs?"),
    ("mechanism", "What mechanisms explain the kidney-protective effects of SGLT2 inhibitors?"),
    ("out_of_scope", "What dietary patterns are recommended for people with type 1 diabetes?"),
    ("out_of_scope", "How effective are SGLT2 inhibitors for treating type 1 diabetes?"),
]


def judge_relevance(question, answer):
    prompt = f"""Question: {question}

Answer: {answer}

Does the answer directly address the question asked? If the evidence was insufficient and the answer says so clearly, that counts as a relevant answer.
Score 1-5 (5 = fully addresses it, or clearly explains why the evidence cannot; 1 = off-topic or evasive).
Output exactly:
SCORE: <number>
REASON: <one sentence>"""
    raw = call_llm(prompt)
    m = re.search(r"SCORE:\s*([1-5])", raw)
    return (int(m.group(1)) if m else None), raw


def metrics(rs):
    keys = ("claims", "supported", "partial", "unsupported", "miscited", "not_checkable")
    t = {k: sum(r["summary"][k] for r in rs) for k in keys}
    checkable = t["claims"] - t["not_checkable"]
    rel = [r["relevance"] for r in rs if r["relevance"]]
    return {
        "questions": len(rs),
        "claims": t["claims"],
        "checkable": checkable,
        "citation_accuracy": round(t["supported"] / checkable, 3) if checkable else None,
        "faithfulness": round((t["supported"] + t["miscited"]) / checkable, 3) if checkable else None,
        "partial": t["partial"],
        "relevance": round(sum(rel) / len(rel), 2) if rel else None,
        "citation_coverage": (
            round(t["claims"] / (t["claims"] + sum(r["summary"].get("uncited_statements", 0) for r in rs)), 3)
            if t["claims"] else None
        ),
        "invalid_citations": sum(len(r["invalid_citations"]) for r in rs),
    }


def report(results):
    print("\n=== OVERALL ===")
    for k, v in metrics(results).items():
        print(f"  {k:18s} {v}")
    print("\n=== BY CATEGORY ===")
    for cat in dict.fromkeys(r["category"] for r in results):
        m = metrics([r for r in results if r["category"] == cat])
        print(f"  {cat:13s} claims={m['claims']:3d}  cit_acc={m['citation_accuracy']}  "
              f"faith={m['faithfulness']}  relevance={m['relevance']}")
        if cat == "out_of_scope":
            rs = [r for r in results if r["category"] == cat]
            clean = sum(1 for r in rs
                        if r["summary"]["supported"] + r["summary"]["partial"]
                        + r["summary"]["unsupported"] + r["summary"]["miscited"] == 0)
            print(f"      clean refusals (no factual claims asserted): {clean}/{len(rs)}")


def export_label_file(results):
    if os.path.exists(LABEL_PATH):
        print(f"\n{LABEL_PATH} already exists, not overwriting (delete it to regenerate)")
        return
    rng = random.Random(SEED)
    verified, flagged = [], []
    for qi, r in enumerate(results):
        for c in r["claims"]:
            entry = {"qi": qi, "claim": c["claim"], "cited": c["cited"],
                     "status": c["status"], "method": c["method"]}
            if c["status"] == "supported":
                verified.append(entry)
            elif c["status"] in ("partial", "unsupported", "miscited"):
                flagged.append(entry)
    rng.shuffle(verified)
    rng.shuffle(flagged)
    chosen = verified[:N_LABEL_EACH] + flagged[:N_LABEL_EACH]
    rng.shuffle(chosen)

    lines = [
        "HAND-LABELLING INSTRUCTIONS",
        "For each item, read the CITED EVIDENCE and decide whether it states the CLAIM.",
        "Write ONE word after LABEL:",
        "  supported    = the text states the claim (paraphrase ok; every number and direction must match)",
        "  partial      = part of the claim is stated, part is not",
        "  unsupported  = not stated in the text, or contradicted",
        "  not_checkable = the claim is about something being absent or missing",
        "Judge ONLY against the text shown, not your own medical knowledge.",
        "=" * 70, "",
    ]
    key = {}
    for n, e in enumerate(chosen, 1):
        r = results[e["qi"]]
        lines += [f"ITEM {n}", f"QUESTION: {r['question']}", f"CLAIM: {e['claim']}", "CITED EVIDENCE"]
        for ev in r["evidence"]:
            if ev["id"] in e["cited"]:
                lines += [f"[{ev['id']}] ({ev['study_type']}, {ev['year']}) {ev['title']}", ev["text"], ""]
        lines += ["LABEL: ", "-" * 70, ""]
        key[n] = {"status": e["status"], "method": e["method"], "question": r["question"]}
    open(LABEL_PATH, "w").write("\n".join(lines))
    json.dump(key, open(LABEL_KEY_PATH, "w"), indent=2)
    print(f"\nWrote {len(chosen)} items to {LABEL_PATH} (verifier verdicts hidden in {LABEL_KEY_PATH})")


def main():
    os.makedirs("data/eval", exist_ok=True)
    json.dump([{"category": c, "question": q} for c, q in QUESTIONS],
              open(QUESTIONS_PATH, "w"), indent=2)
    results = json.load(open(RESULTS_PATH)) if os.path.exists(RESULTS_PATH) else []
    done = {r["question"] for r in results}

    for i, (category, question) in enumerate(QUESTIONS):
        if question in done:
            continue
        print(f"\n[{i + 1}/{len(QUESTIONS)}] ({category}) {question}")
        try:
            state = run_research(question)
            relevance, rel_raw = judge_relevance(question, state["answer"])
        except Exception as e:
            print(f"  FAILED: {e}  (run again to retry)")
            time.sleep(30)
            continue

        v = state["verification"]
        results.append({
            "category": category,
            "question": question,
            "sub_questions": state["sub_questions"],
            "answer": state["answer"],
            "evidence": state["evidence"],
            "claims": v["claims"],
            "summary": v["summary"],
            "relevance": relevance,
            "relevance_note": rel_raw,
            "invalid_citations": state["invalid_citations"],
            "uncited_evidence": state["uncited_evidence"],
            "timings": state["timings"],
        })
        json.dump(results, open(RESULTS_PATH, "w"), indent=2)
        print(f"  claims={v['summary']['claims']} supported={v['summary']['supported']} "
              f"relevance={relevance} timings={state['timings']}")
        time.sleep(SLEEP_SECONDS)

    report(results)
    if len(results) == len(QUESTIONS):
        export_label_file(results)
    else:
        print(f"\n{len(QUESTIONS) - len(results)} question(s) still missing: "
              "run again before labelling")


if __name__ == "__main__":
    main()
