"""
Advanced retrieval: dense + BM25 hybrid (Reciprocal Rank Fusion),
cross-encoder reranking, and metadata filtering.
retrieve.py (v1) is left untouched.

filters example:
  {"year_min": 2020, "study_types": ["rct", "meta_analysis"], "drug_classes": ["sglt2"]}
"""

import json
import os
import re

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_PATH = os.path.join(PROJECT_ROOT, "data/processed/chunks.json")
CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, "data/processed/chroma_db")
COLLECTION_NAME = "diabetes_chunks"

EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CANDIDATES = 50    # candidates per retriever before fusion
RERANK_POOL = 40   # fused candidates passed to the cross-encoder
RRF_K = 60

STUDY_TYPE_FIELDS = {
    "rct": "is_rct",
    "meta_analysis": "is_meta_analysis",
    "review": "is_review",
    "guideline": "is_guideline",
}

_S = {}  # lazily loaded state


def normalize(text):
    """Map unicode hyphens/dashes to '-' (LLM output often contains U+2011)."""
    return re.sub(r"[\u2010-\u2015\u2212]", "-", text)


def tokenize(text):
    return re.findall(r"[a-z0-9]+", normalize(text).lower())


def _load(need_rerank=False):
    if "chunks" not in _S:
        chunks = json.load(open(CHUNKS_PATH))
        _S["chunks"] = chunks
        _S["by_id"] = {c["chunk_id"]: c for c in chunks}
        _S["bm25"] = BM25Okapi([tokenize(c["text"]) for c in chunks])
        _S["embedder"] = SentenceTransformer(EMBED_MODEL)
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _S["collection"] = client.get_collection(COLLECTION_NAME)
    if need_rerank and "reranker" not in _S:
        _S["reranker"] = CrossEncoder(RERANK_MODEL)


# ---------- metadata filtering ----------

def matches(meta, f):
    if not f:
        return True
    if f.get("year_min") and meta.get("year", 0) < f["year_min"]:
        return False
    if f.get("year_max") and meta.get("year", 0) > f["year_max"]:
        return False
    types = f.get("study_types")
    if types and not any(meta.get(STUDY_TYPE_FIELDS[t]) for t in types):
        return False
    classes = f.get("drug_classes")
    if classes and not any(meta.get(f"in_{d}") for d in classes):
        return False
    return True


def _any_of(conds):
    return conds[0] if len(conds) == 1 else {"$or": conds}


def build_where(f):
    """Translate filters into a Chroma where-clause (None if no filters)."""
    if not f:
        return None
    conds = []
    if f.get("year_min"):
        conds.append({"year": {"$gte": f["year_min"]}})
    if f.get("year_max"):
        conds.append({"year": {"$lte": f["year_max"]}})
    if f.get("study_types"):
        conds.append(_any_of([{STUDY_TYPE_FIELDS[t]: True} for t in f["study_types"]]))
    if f.get("drug_classes"):
        conds.append(_any_of([{f"in_{d}": True} for d in f["drug_classes"]]))
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


# ---------- retrievers ----------

def _dense(query, filters):
    emb = _S["embedder"].encode(normalize(query)).tolist()
    res = _S["collection"].query(
        query_embeddings=[emb], n_results=CANDIDATES, where=build_where(filters)
    )
    return res["ids"][0]


def _bm25(query, filters):
    scores = _S["bm25"].get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    out = []
    for i in ranked:
        if scores[i] <= 0:
            break
        chunk = _S["chunks"][i]
        if matches(chunk, filters):
            out.append(chunk["chunk_id"])
        if len(out) >= CANDIDATES:
            break
    return out


def _rrf(rank_lists):
    fused = {}
    for ranks in rank_lists:
        for pos, cid in enumerate(ranks):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + pos + 1)
    return sorted(fused, key=fused.get, reverse=True), fused


def _rerank(query, chunk_ids):
    pairs = [(normalize(query), _S["by_id"][cid]["text"]) for cid in chunk_ids]
    scores = _S["reranker"].predict(pairs)
    order = sorted(range(len(chunk_ids)), key=lambda i: scores[i], reverse=True)
    return [chunk_ids[i] for i in order], {chunk_ids[i]: float(scores[i]) for i in order}


# ---------- public API ----------

def retrieve_v2(query, top_k=10, mode="hybrid_rerank", filters=None):
    """mode: dense | bm25 | hybrid | hybrid_rerank. Returns chunk dicts with a 'score'."""
    _load(need_rerank=(mode == "hybrid_rerank"))

    if mode == "dense":
        ids = _dense(query, filters)
        scores = {cid: 1.0 / (i + 1) for i, cid in enumerate(ids)}
    elif mode == "bm25":
        ids = _bm25(query, filters)
        scores = {cid: 1.0 / (i + 1) for i, cid in enumerate(ids)}
    else:
        ids, scores = _rrf([_dense(query, filters), _bm25(query, filters)])
        if mode == "hybrid_rerank":
            ids, scores = _rerank(query, ids[:RERANK_POOL])

    return [dict(_S["by_id"][cid], score=scores[cid]) for cid in ids[:top_k]]
