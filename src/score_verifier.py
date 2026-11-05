"""Scores the claim verifier against hand labels in data/eval/label_me.txt."""

import json
import re

LABEL_PATH = "data/eval/label_me.txt"
KEY_PATH = "data/eval/label_key.json"
VALID = {"supported", "partial", "unsupported", "not_checkable"}


def parse_labels(text):
    labels = {}
    for block in re.split(r"(?m)^ITEM\s+", text)[1:]:
        m_id = re.match(r"(\d+)", block)
        m_label = re.search(r"(?m)^LABEL:\s*([A-Za-z_]+)", block)
        if m_id and m_label:
            labels[m_id.group(1)] = m_label.group(1).lower()
    return labels


def main():
    labels = parse_labels(open(LABEL_PATH).read())
    key = json.load(open(KEY_PATH))

    missing = [n for n in key if n not in labels]
    bad = {n: l for n, l in labels.items() if l not in VALID}
    if missing or bad:
        print("Fix before scoring:")
        if missing:
            print("  unlabelled items:", missing)
        if bad:
            print("  invalid labels:", bad)
        return

    approved, flagged = [], []
    for n, k in key.items():
        human_ok = labels[n] == "supported"
        (approved if k["status"] == "supported" else flagged).append(human_ok)

    print(f"Labelled {len(labels)} items\n")
    if approved:
        p = sum(approved) / len(approved)
        print(f"Of claims the verifier APPROVED ({len(approved)}): "
              f"{sum(approved)} you judged supported = {p:.0%}")
        print("   -> this is the real citation accuracy of approved claims")
    if flagged:
        f = sum(flagged) / len(flagged)
        print(f"Of claims the verifier FLAGGED ({len(flagged)}): "
              f"{sum(flagged)} you judged supported = {f:.0%}")
        print("   -> this is the verifier's false-alarm rate")

    print("\nBy verifier status (rows) vs your label (columns):")
    cols = ["supported", "partial", "unsupported", "not_checkable"]
    print(f"  {'':12s}" + "".join(f"{c:>15s}" for c in cols))
    for status in ["supported", "partial", "unsupported", "miscited"]:
        row = [sum(1 for n, k in key.items() if k["status"] == status and labels[n] == c) for c in cols]
        if any(row):
            print(f"  {status:12s}" + "".join(f"{v:>15d}" for v in row))

    disagree = [n for n, k in key.items() if (k["status"] == "supported") != (labels[n] == "supported")]
    print(f"\nItems where you and the verifier disagree: {sorted(disagree, key=int)}")


if __name__ == "__main__":
    main()
