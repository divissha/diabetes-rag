"""
FastAPI backend exposing the RAG pipeline as a REST endpoint.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from generate import answer_question

app = FastAPI(title="Diabetes Treatment RAG API")

# Allow the React dev server to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default dev port
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str


class SourceInfo(BaseModel):
    title: str
    pmid: str
    distance: float


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    answer, chunks = answer_question(request.question)
    sources = [
        SourceInfo(title=c["title"], pmid=c["pmid"], distance=c["distance"])
        for c in chunks
    ]
    return AnswerResponse(answer=answer, sources=sources)


@app.get("/health")
def health():
    return {"status": "ok"}