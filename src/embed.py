"""
Embeds all chunks and stores them in a local Chroma vector database.
Reads from data/processed/chunks.json
"""

import json
import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "data/processed/chunks.json"
CHROMA_DB_PATH = "data/processed/chroma_db"
COLLECTION_NAME = "diabetes_chunks"


def main():
    print("Loading chunks...")
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")

    print("Loading embedding model (this may take a minute the first time)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Setting up Chroma...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    # delete the collection if it already exists, so re-runs start fresh
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    print("Embedding and storing chunks...")
    batch_size = 64
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        ids = [c["chunk_id"] for c in batch]
        metadatas = [{"pmid": c["pmid"], "title": c["title"]} for c in batch]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        print(f"  Processed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

    print(f"\nDone. Vector store saved to {CHROMA_DB_PATH}")


if __name__ == "__main__":
    main()