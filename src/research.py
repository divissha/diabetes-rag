"""
Research pipeline: decompose -> retrieve per sub-question -> merge evidence -> synthesize.
Each stage is a plain function over a state dict (easy to port to LangGraph later).

Usage: python3 src/research.py "your question"
"""

import json
import re
import sys
import time

from llm import call_llm
from retrieve_v2 import retrieve_v2
from verify import citation_stats, verify_answer

MAX_SUBQ = 4
CHUNKS_PER_SUBQ = 15     # chunks pulled per sub-question before de-duplicating to papers
PAPERS_PER_SUBQ = 3
MAX_EVIDENCE = 5         # keeps the synthesis prompt inside Groq's token-per-minute cap
MAX_CHARS = 2600         # safety cap only; almost every abstract is shorter

DECOMPOSE_PROMPT = """You break a clinical research question into focused sub-questions for searching a PubMed abstract database on type 2 diabetes drug therapy (metformin, GLP-1 receptor agonists, SGLT2 inhibitors, insulin).

Rules:
- If the question is already focused on a single topic, return it unchanged as the only sub-question.
- Otherwise return 2 to 4 self-contained sub-questions, each understandable without the original (for example one per drug, population, or outcome being compared).
- Output ONLY JSON in this form: {"sub_questions": ["...", "..."]}

Question: <<QUESTION>>"""

SYNTH_PROMPT = """You are a medical evidence assistant. Answer the question using ONLY the numbered evidence below.

Rules:
- Every sentence that states a finding must end with citations like [1] or [2][4]. Cite only numbers that appear in the evidence list.
- When sources differ, give more weight to stronger designs (meta-analysis, RCT) and name the design and year behind each claim.
- If the evidence does not directly answer part of the question, say so explicitly. Never fill gaps from outside knowledge.
- Attribute a finding to a source ONLY if that source's text reports it. Do not add numbers, effect directions, subgroups, or outcomes that the text does not state. If a source does not report a result, do not state one.
- Each evidence item shows its publication year. When sources from different years disagree, or an older source gives a recommendation that newer evidence may have changed, report the more recent evidence first and say the older source predates it. Never present an older recommendation as current.
- No dosing instructions and no personal medical advice.
- Format: a short summary paragraph, then bullet points per comparison point, then one final line starting with "Evidence gaps:".

Question: <<QUESTION>>

Sub-questions investigated:
<<SUBQS>>

Evidence:
<<EVIDENCE>>

Answer:"""


MAX_REVISIONS = 0

REVISE_PROMPT = """Revise the answer below. The listed claims were NOT supported by the evidence they cite. For each one, either delete it, or soften it to say only what the cited text states, using wording from that text. You may fix a citation number if a different evidence item states the same thing. Do NOT add any new facts, numbers or claims. Keep everything else and the same format.

Question: <<QUESTION>>

Evidence:
<<EVIDENCE>>

Current answer:
<<ANSWER>>

Claims to fix:
<<BAD>>

Revised answer:"""


import os

PAPERS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data/raw/diabetes_papers.json",
)
_ABSTRACTS = {}


def abstract_for(pmid, fallback):
    """Full abstract for a paper (a retrieved chunk may be only part of it)."""
    if not _ABSTRACTS:
        for p in json.load(open(PAPERS_PATH)):
            _ABSTRACTS[p["pmid"]] = p["abstract"]
    return _ABSTRACTS.get(pmid, fallback)


def study_type(c):
    if c.get("is_guideline"):
        return "Guideline"
    if c.get("is_meta_analysis"):
        return "Meta-analysis"
    if c.get("is_rct"):
        return "RCT"
    if c.get("is_review"):
        return "Review"
    return "Study"


def decompose(state):
    raw = call_llm(DECOMPOSE_PROMPT.replace("<<QUESTION>>", state["question"]))
    subs = []
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            subs = [s.strip() for s in data.get("sub_questions", [])
                    if isinstance(s, str) and s.strip()]
        except json.JSONDecodeError:
            pass
    state["sub_questions"] = subs[:MAX_SUBQ] or [state["question"]]
    return state


def retrieve_evidence(state):
    per_sq = []
    for sq in state["sub_questions"]:
        papers, seen = [], set()
        for c in retrieve_v2(sq, top_k=CHUNKS_PER_SUBQ, filters=state.get("filters")):
            if c["pmid"] not in seen:
                seen.add(c["pmid"])
                papers.append(c)
            if len(papers) == PAPERS_PER_SUBQ:
                break
        per_sq.append(papers)

    # round-robin merge: best paper from each sub-question first, then second-best, ...
    evidence, index = [], {}
    for rank in range(PAPERS_PER_SUBQ):
        for sq_idx, papers in enumerate(per_sq):
            if rank >= len(papers):
                continue
            c = papers[rank]
            if c["pmid"] in index:
                index[c["pmid"]]["sub_questions"].append(sq_idx)
                continue
            if len(evidence) >= MAX_EVIDENCE:
                continue
            item = {
                "id": len(evidence) + 1,
                "pmid": c["pmid"],
                "chunk_id": c["chunk_id"],
                "title": c["title"],
                "year": c["year"],
                "journal": c["journal"],
                "study_type": study_type(c),
                "text": abstract_for(c["pmid"], c["text"])[:MAX_CHARS],
                "score": c["score"],
                "sub_questions": [sq_idx],
            }
            evidence.append(item)
            index[c["pmid"]] = item

    state["evidence"] = evidence
    return state


def synthesize(state):
    evidence_block = "\n\n".join(
        f"[{e['id']}] ({e['study_type']}, {e['year']}) {e['title']}\n{e['text']}"
        for e in state["evidence"]
    )
    subqs = "\n".join(f"- {s}" for s in state["sub_questions"])
    prompt = (SYNTH_PROMPT
              .replace("<<QUESTION>>", state["question"])
              .replace("<<SUBQS>>", subqs)
              .replace("<<EVIDENCE>>", evidence_block))
    answer = call_llm(prompt, temperature=0.1)
    state["answer"] = answer

    cited = set()
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer):
        cited.update(int(n) for n in re.split(r"\s*,\s*", group))
    valid = {e["id"] for e in state["evidence"]}
    state["invalid_citations"] = sorted(cited - valid)
    state["uncited_evidence"] = sorted(valid - cited)
    return state


def verify_and_revise(state):
    report = verify_answer(state["answer"], state["evidence"])
    state["verification_initial"] = report["summary"]
    state["revised"] = False

    bad = [c for c in report["claims"]
           if c["status"] != "supported" or c["unverified_numbers"]]
    if bad and MAX_REVISIONS > 0:
        evidence_block = "\n\n".join(
            f"[{e['id']}] ({e['study_type']}, {e['year']}) {e['title']}\n{e['text']}"
            for e in state["evidence"]
        )
        bad_block = "\n".join(
            f"- \"{c['claim']}\" (cited {c['cited']}; problem: {c['status']}"
            + (f", numbers not in evidence: {c['unverified_numbers']}" if c["unverified_numbers"] else "")
            + ")"
            for c in bad
        )
        prompt = (REVISE_PROMPT
                  .replace("<<QUESTION>>", state["question"])
                  .replace("<<EVIDENCE>>", evidence_block)
                  .replace("<<ANSWER>>", state["answer"])
                  .replace("<<BAD>>", bad_block))
        state["answer"] = call_llm(prompt, temperature=0.1)
        state["revised"] = True
        report = verify_answer(state["answer"], state["evidence"])

    state["verification"] = report
    state["invalid_citations"], state["uncited_evidence"] = citation_stats(
        state["answer"], state["evidence"]
    )
    return state


def run_research(question, filters=None):
    state = {"question": question, "filters": filters, "timings": {}}
    for name, fn in [("decompose", decompose),
                     ("retrieve", retrieve_evidence),
                     ("synthesize", synthesize),
                     ("verify", verify_and_revise)]:
        t0 = time.time()
        state = fn(state)
        state["timings"][name] = round(time.time() - t0, 1)
    return state


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or \
        "How do SGLT2 inhibitors and GLP-1 receptor agonists compare for cardiovascular and kidney outcomes in type 2 diabetes?"
    result = run_research(question)

    print("\nSUB-QUESTIONS")
    for i, s in enumerate(result["sub_questions"]):
        print(f"  {i+1}. {s}")

    print("\nEVIDENCE")
    for e in result["evidence"]:
        print(f"  [{e['id']}] {e['study_type']}, {e['year']}: {e['title'][:85]}")

    print("\nANSWER\n")
    print(result["answer"])

    v = result["verification"]
    print("\nVERIFICATION")
    if result["revised"]:
        print("  answer was revised once after the first check")
        print("  before:", result["verification_initial"])
    print("  final: ", v["summary"])
    for c in v["claims"]:
        if c["status"] != "supported" or c["unverified_numbers"]:
            print(f"  - [{c['status']}] cited {c['cited']} (entailment {c['best_cited']:.2f}) "
                  f"numbers not in evidence: {c['unverified_numbers'] or 'none'}")
            print(f"      {c['claim'][:170]}")
            if c["note"]:
                print(f"      note: {c['note']}")

    print("\nCHECKS")
    print("  invalid citations:", result["invalid_citations"] or "none")
    print("  evidence never cited:", result["uncited_evidence"] or "none")
    print("  timings (s):", result["timings"])
