"""FastAPI wrapper.  Run:  uvicorn app.api:app --reload"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import config
from app.rag import answer, retrieve
from app.store import get_client

app = FastAPI(title="Docs RAG", version="0.1.0",
              description="Baseline RAG over the FastAPI docs with cited answers.")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000,
                          examples=["How do I declare an optional query parameter?"])
    top_k: int = Field(config.TOP_K, ge=1, le=20)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000)
    top_k: int = Field(config.TOP_K, ge=1, le=20)


@app.get("/health")
def health():
    client = get_client()
    if not client.collection_exists(config.COLLECTION):
        return {"status": "no_index", "hint": "run: python -m app.ingest"}
    count = client.count(config.COLLECTION).count
    return {"status": "ok", "chunks": count, "corpus_version": config.CORPUS_VERSION, "model": config.LLM_MODEL}


@app.post("/ask")
def ask(req: AskRequest):
    try:
        return answer(req.question, k=req.top_k)
    except Exception as e:  # surface provider/auth errors clearly
        raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")


@app.post("/search")
def search(req: SearchRequest):
    """Retrieval only (no LLM call) - handy for debugging."""
    return {"query": req.query, "results": retrieve(req.query, req.top_k)}
