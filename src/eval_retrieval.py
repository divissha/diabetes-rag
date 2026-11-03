"""
Scores retrieval systems on the silver set: Recall@K, MRR, nDCG@10.
Add new systems to SYSTEMS in later phases to compare them.
"""

import json
import math

from retrieve import retrieve

SET_PATH = "data/eval/retrieval_set.json"
RESULTS_PATH = "data/eval/retrieval_results.json"
KS = [1, 3, 5, 10]
POOL = 40  # chunks fetched per query before de-duplicating to papers


def dense_system(query):
    """Current system: dense embeddings only. Returns ranked unique PMIDs."""
    ranked = []
    for c in retrieve(query, top_k=POOL):
        if c["pmid"] not in ranked:
            ranked.append(c["pmid"])
    return ranked


from retrieve_v2 import retrieve_v2


def make_system(mode):
    def system(query):
        ranked = []
        for c in retrieve_v2(query, top_k=POOL, mode=mode):
            if c["pmid"] not in ranked:
                ranked.append(c["pmid"])
        return ranked
    return system


SYSTEMS = {
    "dense (v1 baseline)": dense_system,
    "dense": make_system("dense"),
    "bm25": make_system("bm25"),
    "hybrid": make_system("hybrid"),
    "hybrid+rerank (minilm)": make_system("hybrid_rerank"),
    "hybrid+rerank (medcpt)": make_system("hybrid_rerank:medcpt"),
}


def score(system_fn, items):
    hits = {k: 0 for k in KS}
    rr, ndcg, misses = 0.0, 0.0, []
    for it in items:
        ranked = system_fn(it["question"])
        gold = it["pmid"]
        if gold not in ranked:
            misses.append(it["question"])
            continue
        rank = ranked.index(gold) + 1
        for k in KS:
            if rank <= k:
                hits[k] += 1
        rr += 1 / rank
        if rank <= 10:
            ndcg += 1 / math.log2(rank + 1)
    n = len(items)
    metrics = {f"Recall@{k}": hits[k] / n for k in KS}
    metrics["MRR"] = rr / n
    metrics["nDCG@10"] = ndcg / n
    return metrics, misses


def main():
    items = json.load(open(SET_PATH))
    print(f"Evaluating on {len(items)} questions\n")

    all_results = {}
    for name, fn in SYSTEMS.items():
        metrics, misses = score(fn, items)
        all_results[name] = {"metrics": metrics, "num_misses": len(misses)}
        print(f"=== {name} ===")
        for k, v in metrics.items():
            print(f"  {k:10s} {v:.3f}")
        print(f"  not found in top {POOL} chunks: {len(misses)}\n")

    json.dump(all_results, open(RESULTS_PATH, "w"), indent=2)
    print(f"Saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
