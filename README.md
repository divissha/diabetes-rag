# Diabetes Treatment Evidence RAG

A RAG system that synthesizes evidence from PubMed research on Type 2 
diabetes drug therapy, answering comparison questions across drug
classes (metformin, GLP-1 agonists, SGLT2 inhibitors, insulin) with
citations back to source papers. Scoped for synthesis across
multiple papers, not single-document lookup — correctly declines to
answer when retrieved evidence doesn't support a claim, rather than
hallucinating.

## Tech stack

- **Data**: NCBI E-utilities API (PubMed), 384 papers, 452 chunks
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector store**: ChromaDB
- **LLM**: Groq (openai/gpt-oss-20b)
- **Backend**: FastAPI
- **Frontend**: React (Vite)
- **Evaluation**: custom LLM-as-judge (ragas had unresolved Python 3.13 async issues)

## Results

| Metric | Score |
|---|---|
| Faithfulness | 4.38 / 5 |
| Answer relevance | 4.12 / 5 |
