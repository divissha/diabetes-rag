"""Re-runs claim verification on saved answers + evidence (no regeneration)."""

import json
import os
import shutil
import time

from eval_generation import RESULTS_PATH, export_label_file, report
from verify import citation_stats, verify_answer

BACKUP = "data/eval/generation_results_v1.json"
SLEEP_SECONDS = 20  # keeps the judge calls under Groq's tokens-per-minute cap


def main():
    if not os.path.exists(BACKUP):
        shutil.copy(RESULTS_PATH, BACKUP)
    results = json.load(open(RESULTS_PATH))

    for i, r in enumerate(results):
        print(f"[{i + 1}/{len(results)}] {r['question'][:80]}")
        v = verify_answer(r["answer"], r["evidence"])
        r["claims"], r["summary"], r["uncited"] = v["claims"], v["summary"], v["uncited"]
        r["invalid_citations"], r["uncited_evidence"] = citation_stats(r["answer"], r["evidence"])
        print("   ", v["summary"])
        json.dump(results, open(RESULTS_PATH, "w"), indent=2)
        time.sleep(SLEEP_SECONDS)

    report(results)
    export_label_file(results)


if __name__ == "__main__":
    main()
