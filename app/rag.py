"""Plain RAG: retrieve top-k chunks -> answer with an LLM that must cite sources as [n]."""
import re
import time

import litellm

from app import config
from app.store import embed, get_client

litellm.drop_params = True  # ignore params a provider doesn't support instead of erroring

SYSTEM_PROMPT = """You answer questions about the FastAPI documentation using ONLY the numbered context passages.
Rules:
- Cite every claim with the passage number in square brackets, e.g. [1] or [2][4].
- If the context does not contain the answer, reply exactly: "I don't know based on the provided documentation."
- Be concise: 2-6 sentences, plus a short code snippet only if it is essential."""


def retrieve(query: str, k: int = config.TOP_K) -> list[dict]:
    qvec = embed([query])[0]
    hits = get_client().query_points(config.COLLECTION, query=qvec, limit=k, with_payload=True).points
    return [
        {
            "doc_id": h.payload["doc_id"],
            "title": h.payload["title"],
            "chunk_index": h.payload["chunk_index"],
            "text": h.payload["text"],
            "score": round(float(h.score), 4),
        }
        for h in hits
    ]


def build_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {c['doc_id']})\n{c['text']}" for i, c in enumerate(chunks, start=1)
    )


def _cost(response, model: str, tokens_in: int, tokens_out: int) -> float:
    try:
        return float(litellm.completion_cost(completion_response=response))
    except Exception:
        price = config.PRICES.get(model)
        if not price:
            return 0.0
        return (tokens_in * price["in"] + tokens_out * price["out"]) / 1_000_000


def generate(query: str, chunks: list[dict], model: str = config.LLM_MODEL) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{build_context(chunks)}\n\nQuestion: {query}"},
    ]
    response = litellm.completion(
        model=model, messages=messages, temperature=0, max_tokens=config.MAX_ANSWER_TOKENS,
        num_retries=3, timeout=60,
    )
    answer = (response.choices[0].message.content or "").strip()
    usage = response.usage
    tokens_in, tokens_out = usage.prompt_tokens, usage.completion_tokens
    return {
        "answer": answer,
        "model": model,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": round(_cost(response, model, tokens_in, tokens_out), 8),
    }


def cited_sources(answer: str, chunks: list[dict]) -> list[str]:
    """Map [n] markers in the answer back to doc_ids (deduplicated, in order)."""
    seen: list[str] = []
    for n in re.findall(r"\[(\d+)\]", answer):
        idx = int(n) - 1
        if 0 <= idx < len(chunks) and chunks[idx]["doc_id"] not in seen:
            seen.append(chunks[idx]["doc_id"])
    return seen


def answer(query: str, model: str = config.LLM_MODEL, k: int = config.TOP_K) -> dict:
    t0 = time.perf_counter()
    chunks = retrieve(query, k)
    t_retrieval = time.perf_counter() - t0
    gen = generate(query, chunks, model)
    latency = time.perf_counter() - t0
    return {
        "query": query,
        **gen,
        "citations": cited_sources(gen["answer"], chunks),
        "retrieved": [
            {"doc_id": c["doc_id"], "chunk_index": c["chunk_index"], "score": c["score"]} for c in chunks
        ],
        "retrieval_ms": round(t_retrieval * 1000, 1),
        "latency_ms": round(latency * 1000, 1),
    }
