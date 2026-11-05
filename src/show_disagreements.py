import json
import re

from score_verifier import parse_labels

text = open("data/eval/label_me.txt").read()
labels = parse_labels(text)
key = json.load(open("data/eval/label_key.json"))

claims = {}
for block in re.split(r"(?m)^ITEM\s+", text)[1:]:
    n = re.match(r"(\d+)", block).group(1)
    claims[n] = re.search(r"(?m)^CLAIM:\s*(.*)$", block).group(1).strip()

results = json.load(open("data/eval/generation_results.json"))
lookup = {(r["question"], c["claim"]): c for r in results for c in r["claims"]}

for n in sorted(key, key=int):
    k = key[n]
    if (k["status"] == "supported") == (labels[n] == "supported"):
        continue
    c = lookup.get((k["question"], claims[n]), {})
    print(f"ITEM {n}: verifier={k['status']} ({k['method']})  you={labels[n]}")
    print(f"  claim: {claims[n][:220]}")
    if c.get("unverified_numbers"):
        print(f"  numbers not found in cited text: {c['unverified_numbers']}")
    if c.get("note"):
        print(f"  verifier note: {c['note']}")
    if c.get("quote"):
        print(f"  judge quote: {c['quote'][:200]}")
    print()
