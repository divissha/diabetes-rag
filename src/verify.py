"""
Claim-level verification, two tiers:
  1. Local NLI screen (free, fast).
  2. LLM judge (one batched call) for claims the NLI screen rejects. The judge must
     quote the supporting sentence, and we check the quote really appears in the
     cited evidence, so the judge cannot invent support.
Statuses: supported | partial | unsupported | miscited | not_checkable
"""

import json
import re

import numpy as np
from sentence_transformers import CrossEncoder

from llm import call_llm
from retrieve_v2 import normalize

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
SUPPORT_THRESHOLD = 0.5
WINDOW = 3
MIN_QUOTE_CHARS = 25

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(])")
_CITE_GROUP = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_CITE_STRIP = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")
_state = {}

JUDGE_PROMPT = """You are a strict fact-checker. For each numbered claim, decide whether the evidence it cites supports it.

Verdicts:
- SUPPORTED: the evidence states the claim (paraphrase is fine, but every number and every direction of effect must match).
- PARTIAL: part of the claim is stated, but part is not (extra detail, a different magnitude, stronger wording, or a subgroup the text does not mention).
- UNSUPPORTED: the evidence does not state it, or contradicts it.
- NOT_CHECKABLE: the claim says something is absent, lacking, or unreported.

For SUPPORTED or PARTIAL, give "quote": the single sentence from the evidence that best supports the claim, copied EXACTLY word for word, with no ellipses.

Output ONLY this JSON:
{"verdicts": [{"id": 1, "verdict": "SUPPORTED", "quote": "...", "note": "at most 15 words on what is missing or wrong"}]}

Evidence:
<<EVIDENCE>>

Claims:
<<CLAIMS>>"""


def _model():
    if "m" not in _state:
        m = CrossEncoder(NLI_MODEL)
        id2label = m.model.config.id2label
        idx = next((int(i) for i, l in id2label.items() if str(l).lower() == "entailment"), 1)
        _state["m"], _state["idx"] = m, idx
    return _state["m"], _state["idx"]


def extract_claims(answer):
    claims = []
    for line in answer.split("\n"):
        line = re.sub(r"\*\*", "", line.strip()).lstrip("-*• ").strip()
        if not line:
            continue
        for sent in _SENT_SPLIT.split(line):
            ids = [int(n) for g in _CITE_GROUP.findall(sent)
                   for n in re.split(r"\s*,\s*", g)]
            if ids:
                claims.append({"claim": _CITE_STRIP.sub("", sent).strip(),
                               "cited": sorted(set(ids))})
    return claims


def citation_stats(answer, evidence):
    cited = set()
    for g in _CITE_GROUP.findall(answer):
        cited.update(int(n) for n in re.split(r"\s*,\s*", g))
    valid = {e["id"] for e in evidence}
    return sorted(cited - valid), sorted(valid - cited)


def _windows(text):
    sents = _SENT_SPLIT.split(normalize(text))
    if len(sents) <= WINDOW:
        return [normalize(text)]
    return [" ".join(sents[i:i + WINDOW]) for i in range(len(sents) - WINDOW + 1)]


def _score(claim, items):
    """Max entailment probability of the claim against each evidence item."""
    model, idx = _model()
    pairs, owners = [], []
    for e in items:
        for w in _windows(e["text"]):
            pairs.append((w, normalize(claim)))
            owners.append(e["id"])
    if not pairs:
        return {}
    logits = np.asarray(model.predict(pairs, show_progress_bar=False, batch_size=32))
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    ent = (exp / exp.sum(axis=1, keepdims=True))[:, idx]
    best = {}
    for owner, p in zip(owners, ent):
        best[owner] = max(best.get(owner, 0.0), float(p))
    return best


def _numbers(text):
    nums = set()
    for n in re.findall(r"\d[\d,]*\.?\d*", normalize(text)):
        n = n.replace(",", "").rstrip(".")
        if len(n.replace(".", "")) >= 2:
            nums.add(n)
    return nums


def _squash(s):
    return re.sub(r"[^a-z0-9]+", " ", normalize(s).lower()).strip()


def _quote_in_evidence(quote, items):
    q = _squash(quote or "")
    if len(q) < MIN_QUOTE_CHARS:
        return False
    return any(q in _squash(e["text"]) for e in items)


def _llm_judge(pending, by_id):
    ids = sorted({i for r in pending for i in r["cited"] if i in by_id})
    evidence_block = "\n\n".join(f"[{i}] {by_id[i]['text']}" for i in ids)
    claims_block = "\n".join(
        f"{n + 1}. (cites {r['cited']}) {r['claim']}" for n, r in enumerate(pending)
    )
    prompt = (JUDGE_PROMPT.replace("<<EVIDENCE>>", evidence_block)
                          .replace("<<CLAIMS>>", claims_block))
    raw = call_llm(prompt, temperature=0.0)

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return
    try:
        verdicts = json.loads(match.group(0)).get("verdicts", [])
    except json.JSONDecodeError:
        return

    for v in verdicts:
        try:
            r = pending[int(v["id"]) - 1]
        except (KeyError, ValueError, IndexError, TypeError):
            continue
        verdict = str(v.get("verdict", "")).upper()
        cited_items = [by_id[i] for i in r["cited"] if i in by_id]
        r["note"] = v.get("note", "")
        r["quote"] = v.get("quote", "")
        r["method"] = "llm_judge"

        if verdict in ("SUPPORTED", "PARTIAL"):
            if _quote_in_evidence(v.get("quote"), cited_items):
                r["status"] = "supported" if verdict == "SUPPORTED" else "partial"
            else:
                r["status"] = "unsupported"
                r["note"] = "judge's quote not found in cited evidence"
        elif verdict == "NOT_CHECKABLE":
            r["status"] = "not_checkable"
        else:
            r["status"] = "unsupported"


def verify_answer(answer, evidence, use_llm=True):
    by_id = {e["id"]: e for e in evidence}
    years = {str(e["year"]) for e in evidence}
    results = []

    for c in extract_claims(answer):
        cited = [i for i in c["cited"] if i in by_id]
        scores = _score(c["claim"], [by_id[i] for i in cited])
        best = max(scores.values(), default=0.0)

        cited_nums = set()
        for i in cited:
            cited_nums |= _numbers(by_id[i]["text"])

        results.append({
            "claim": c["claim"],
            "cited": c["cited"],
            "status": "supported" if best >= SUPPORT_THRESHOLD else "pending",
            "method": "nli",
            "best_cited": best,
            "quote": "",
            "note": "",
            "supported_by_other": [],
            "unverified_numbers": sorted(_numbers(c["claim"]) - cited_nums - years),
        })

    pending = [r for r in results if r["status"] == "pending"]
    if pending and use_llm:
        try:
            _llm_judge(pending, by_id)
        except Exception as e:  # keep the pipeline alive if the judge call fails
            print(f"    judge call failed: {e}")
    for r in pending:
        if r["status"] == "pending":
            r["status"] = "unsupported"

    for r in results:
        if r["status"] == "supported" and r["unverified_numbers"]:
            r["status"] = "partial"
            r["note"] = f"numbers not in cited evidence: {r['unverified_numbers']}"
        if r["status"] == "unsupported":
            others = [e for e in evidence if e["id"] not in r["cited"]]
            alt = sorted(i for i, v in _score(r["claim"], others).items()
                         if v >= SUPPORT_THRESHOLD)
            if alt:
                r["status"], r["supported_by_other"] = "miscited", alt

    n = len(results)
    count = lambda s: sum(r["status"] == s for r in results)
    checkable = n - count("not_checkable")
    summary = {
        "claims": n,
        "supported": count("supported"),
        "partial": count("partial"),
        "unsupported": count("unsupported"),
        "miscited": count("miscited"),
        "not_checkable": count("not_checkable"),
        "citation_precision": round(count("supported") / checkable, 3) if checkable else 0.0,
    }
    return {"claims": results, "summary": summary}
