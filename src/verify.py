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

JUDGE_PROMPT = """You are a strict fact-checker. For each numbered claim:
1. Split the claim into its separate assertions ("parts"). Each number, each comparison, each population or subgroup, each direction of effect, and each statement joined by "and", ";" or "while" is its own part. Use 1 to 5 parts.
2. For each part give a verdict from the cited evidence:
   - STATED: the evidence states this part. Paraphrase is fine, but every number and every direction of effect must match. A comparison such as "greater", "more", "greatest" or "superior" is STATED only if the evidence reports that compared result, not merely each side separately.
   - NOT_STATED: the evidence does not state it, states something different, or contradicts it.
   - ABSENCE: the part says something is missing, lacking, or unreported.
3. For STATED, give "quote": ONE sentence copied EXACTLY, word for word, from the evidence, with no ellipses, that states this part.

Judge ONLY against the evidence shown, never against your own knowledge.

Output ONLY this JSON:
{"verdicts": [{"id": 1, "parts": [{"part": "short restatement", "verdict": "STATED", "quote": "exact sentence"}], "note": "at most 15 words on what is missing, if anything"}]}

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


def _units(answer):
    """Sentence-level units: {text, cited, gaps}. A citation that lands after the
    period (or on its own) attaches to the sentence before it. Headings are skipped."""
    units, in_gaps = [], False
    for raw in answer.split("\n"):
        if not raw.strip():
            continue
        if re.fullmatch(r"[-*\u2022\s]*\*\*[^*]+\*\*:?\s*", raw):
            if re.match(r"(?i)[-*\u2022\s]*\*\*evidence gaps?", raw):
                in_gaps = True
            continue
        line = re.sub(r"\*\*", "", raw.strip()).lstrip("-*\u2022 ").strip()
        if not line:
            continue
        if re.match(r"(?i)^evidence gaps?\b", line):
            in_gaps = True
        for sent in _SENT_SPLIT.split(line):
            ids = [int(n) for g in _CITE_GROUP.findall(sent)
                   for n in re.split(r"\s*,\s*", g)]
            text = _CITE_STRIP.sub("", sent).strip()
            if not text:  # citation-only fragment
                if units and ids:
                    units[-1]["cited"] = sorted(set(units[-1]["cited"]) | set(ids))
                continue
            units.append({"text": text, "cited": sorted(set(ids)), "gaps": in_gaps})
    return units


def extract_claims(answer):
    return [{"claim": u["text"], "cited": u["cited"]} for u in _units(answer) if u["cited"]]


def uncited_statements(answer):
    """Substantive statements with no citation (headings, evidence gaps, absence claims excluded)."""
    return [u["text"] for u in _units(answer)
            if not u["cited"] and not u["gaps"]
            and len(u["text"].split()) >= 6 and not _is_absence(u["text"])]


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
    t = normalize(text)
    # "7 053" / "47,000" / "7053" all compare equal
    t = re.sub(r"(?<=\d)[ \u00a0\u2009\u202f,](?=\d{3}(?!\d))", "", t)
    nums = set()
    for n in re.findall(r"\d+(?:\.\d+)?", t):
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


def _is_compound(claim):
    """Long claims, or ones joining several assertions, need part-by-part checking."""
    return (len(claim.split()) > 18 or ";" in claim
            or bool(re.search(r",\s*(and|while|whereas|but)\b|\b(whereas|while|in addition|as well as)\b",
                              claim, re.I)))


def _llm_judge(pending, by_id, batch_size=6):
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        ids = sorted({i for r in batch for i in r["cited"] if i in by_id})
        evidence_block = "\n\n".join(f"[{i}] {by_id[i]['text']}" for i in ids)
        claims_block = "\n".join(
            f"{n + 1}. (cites {r['cited']}) {r['claim']}" for n, r in enumerate(batch)
        )
        prompt = (JUDGE_PROMPT.replace("<<EVIDENCE>>", evidence_block)
                              .replace("<<CLAIMS>>", claims_block))
        raw = call_llm(prompt, temperature=0.0)

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            continue
        try:
            verdicts = json.loads(match.group(0)).get("verdicts", [])
        except json.JSONDecodeError:
            continue

        for v in verdicts:
            try:
                r = batch[int(v["id"]) - 1]
            except (KeyError, ValueError, IndexError, TypeError):
                continue
            cited_items = [by_id[i] for i in r["cited"] if i in by_id]
            stated = total = absence = 0
            missing, quotes = [], []
            for p in v.get("parts") or []:
                pv = str(p.get("verdict", "")).upper()
                if pv == "ABSENCE":
                    absence += 1
                    continue
                total += 1
                if pv == "STATED" and _quote_in_evidence(p.get("quote"), cited_items):
                    stated += 1
                    quotes.append(p.get("quote"))
                else:
                    missing.append(str(p.get("part", ""))[:80])

            if total == 0:
                r["status"] = "not_checkable" if absence else "unsupported"
            elif stated == total:
                r["status"] = "supported"
            elif stated:
                r["status"] = "partial"
            else:
                r["status"] = "unsupported"
            r["method"] = "llm_judge"
            r["quote"] = " | ".join(quotes[:2])
            note = str(v.get("note", "")).strip()
            if missing:
                note = (note + " " if note else "") + "not stated: " + "; ".join(missing)
            r["note"] = note

_ABSENCE = re.compile(
    r"\b(no (direct |head-to-head |randomi[sz]ed )?(evidence|trial|trials|rct|rcts|study|studies|data|comparison)"
    r"|(is|are|remains?) (still )?(lacking|limited|unclear|unknown|unresolved|pending|absent)"
    r"|lack(s|ing)?\b"
    r"|not (been )?(stratified|studied|compared|reported|addressed|evaluated)"
    r"|did not (stratify|report|address|evaluate|compare))",
    re.I,
)


def _is_absence(claim):
    """Claims about missing evidence can't be checked against a passage.
    Claims that contain numbers are still verified normally."""
    return bool(_ABSENCE.search(normalize(claim))) and not re.search(r"\d", claim)


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

    for r in results:
        if r["status"] == "supported" and _is_compound(r["claim"]):
            r["status"], r["nli_pass"] = "pending", True

    for r in results:
        if r["status"] == "pending" and _is_absence(r["claim"]) and not _is_compound(r["claim"]):
            r["status"], r["method"] = "not_checkable", "rule"

    pending = [r for r in results if r["status"] == "pending"]
    if pending and use_llm:
        try:
            _llm_judge(pending, by_id)
        except Exception as e:  # keep the pipeline alive if the judge call fails
            print(f"    judge call failed: {e}")
    for r in pending:
        if r["status"] == "pending":
            r["status"] = "supported" if r.get("nli_pass") else "unsupported"

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
    unc = uncited_statements(answer)
    total = n + len(unc)
    summary["uncited_statements"] = len(unc)
    summary["citation_coverage"] = round(n / total, 3) if total else None
    return {"claims": results, "uncited": unc, "summary": summary}
