"""
Reusable retrieval function: given a question, returns the top-k most relevant chunks.
"""

import chromadb
from sentence_transformers import SentenceTransformer

import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, "data/processed/chroma_db")
COLLECTION_NAME = "diabetes_chunks"

_model = None
_collection = None


def _load():
    """Lazy-load the model and collection so this only happens once."""
    global _model, _collection
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _collection = client.get_collection(COLLECTION_NAME)


def retrieve(query, top_k=5):
    """Return the top_k most relevant chunks for a given query."""
    _load()
    query_embedding = _model.encode(query).tolist()
    results = _collection.query(query_embeddings=[query_embedding], n_results=top_k)

    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        chunks.append({
            "text": doc,
            "title": results["metadatas"][0][i]["title"],
            "pmid": results["metadatas"][0][i]["pmid"],
            "distance": results["distances"][0][i],
        })
    return chunks


if __name__ == "__main__":
    # quick manual test when run directly
    test_query = "How does metformin compare to GLP-1 agonists for weight loss?"
    results = retrieve(test_query)
    for i, r in enumerate(results):
        print(f"--- Result {i+1} (distance: {r['distance']:.3f}) ---")
        print(f"Title: {r['title']}")
        print(r['text'][:200])
        print()