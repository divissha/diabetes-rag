"""Re-scores the CURRENT verifier output against your existing hand labels."""

import json
import re

from score_verifier import parse_labels

text = open("data/eval/label_me.txt").read()
labels = parse_labels(text)
results = json.load(open("data/eval/generation_results.json"))
lookup = {(r["question"], c["claim"]): c for r in results for c in r["claims"]}

rows, missing = [], []
for block in re.split(r"(?m)^ITEM\s+", text)[1:]:
    n = re.match(r"(\d+)", block).group(1)
    question = re.search(r"(?m)^QUESTION:\s*(.*)$", block).group(1).strip()
    claim = re.search(r"(?m)^CLAIM:\s*(.*)$", block).group(1).strip()
    c = lookup.get((question, claim))
    if c is None:
        missing.append(n)
        continue
    rows.append((n, c["status"], labels[n], claim, c))

if missing:
    print("could not match items:", missing)

approved = [h == "supported" for _, st, h, _, _ in rows if st == "supported"]
flagged = [h == "supported" for _, st, h, _, _ in rows if st != "supported"]
agree = sum((st == "supported") == (h == "supported") for _, st, h, _, _ in rows)

print(f"Items matched: {len(rows)}")
print(f"Agreement on supported vs not: {agree}/{len(rows)}")
if approved:
    print(f"APPROVED ({len(approved)}): {sum(approved)} you judged fully supported = {sum(approved)/len(approved):.0%}")
if flagged:
    print(f"FLAGGED  ({len(flagged)}): {sum(flagged)} you judged fully supported = {sum(flagged)/len(flagged):.0%}  (false alarms)")

cols = ["supported", "partial", "unsupported", "not_checkable"]
print(f"\n{'verifier / you':16s}" + "".join(f"{c:>15s}" for c in cols))
for status in ["supported", "partial", "unsupported", "miscited", "not_checkable"]:
    row = [sum(1 for _, st, h, _, _ in rows if st == status and h == c) for c in cols]
    if any(row):
        print(f"{status:16s}" + "".join(f"{v:>15d}" for v in row))

print("\nDisagreements:")
for n, st, h, claim, c in rows:
    if (st == "supported") != (h == "supported"):
        print(f"  ITEM {n}: verifier={st}  you={h}")
        print(f"     {claim[:150]}")
        if c.get("note"):
            print(f"     note: {c['note'][:160]}")
