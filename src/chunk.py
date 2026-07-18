"""
Splits paper abstracts into smaller passages ("chunks") suitable for embedding.
Reads from data/raw/diabetes_papers.json, writes to data/processed/chunks.json
"""

import json
import os

INPUT_PATH = "data/raw/diabetes_papers.json"
OUTPUT_PATH = "data/processed/chunks.json"

CHUNK_SIZE = 300      # target words per chunk
CHUNK_OVERLAP = 50    # words shared between consecutive chunks


def chunk_text(text, chunk_size, overlap):
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # move forward, but overlap with previous

    return chunks


def main():
    os.makedirs("data/processed", exist_ok=True)

    with open(INPUT_PATH) as f:
        papers = json.load(f)

    all_chunks = []
    for paper in papers:
        # combine title + abstract so title context isn't lost
        full_text = f"{paper['title']}. {paper['abstract']}"
        text_chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk in enumerate(text_chunks):
            all_chunks.append({
                "chunk_id": f"{paper['pmid']}_{i}",
                "pmid": paper["pmid"],
                "title": paper["title"],
                "text": chunk,
            })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"Created {len(all_chunks)} chunks from {len(papers)} papers")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()