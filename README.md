# docs-rag: a measured RAG baseline over the FastAPI docs

A retrieval-augmented Q&A API that answers questions about the FastAPI documentation with **cited sources**, plus a **repeatable evaluation harness** so every later change (caching, routing, re-ranking) can be measured against a fixed baseline.

## Results (baseline)

<!-- Fill from results/summary.csv after running the eval -->
| Metric | Value |
|---|---|
| Retrieval hit rate @5 | _TBD_ |
| MRR | _TBD_ |
| Answer accuracy (LLM judge) | _TBD_ |
| Citation rate (cites the right doc) | _TBD_ |
| Cost per request | _TBD_ |
| p50 / p95 latency | _TBD_ |

Eval set: 65 questions (62 answerable across lookup / how-to / conceptual, 3 deliberately unanswerable), each grounded by a verbatim evidence quote from its source doc.

## Architecture

```
data/docs/*.md ──clean──> 500-token chunks (75 overlap)
                              │  all-MiniLM-L6-v2
                              ▼
                        Qdrant (cosine)
                              │  top-5
question ──embed──> retrieve ─┘──> LLM (LiteLLM) ──> answer + [n] citations
                                                    + tokens, cost, latency
```

| Component | Choice |
|---|---|
| Corpus | 120 pages of the FastAPI docs (tutorial, advanced, deployment, how-to), ~105k words |
| Chunking | tiktoken `cl100k_base`, 500 tokens, 75 overlap, doc title prefixed for embedding |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| Vector store | Qdrant (embedded local mode; set `QDRANT_URL` for a server) |
| LLM | Any LiteLLM model, default `gemini/gemini-2.5-flash` |
| API | FastAPI: `GET /health`, `POST /ask`, `POST /search` |
| Eval | Hit rate, MRR, citation rate, LLM-judge correctness, cost, p50/p95 latency to CSV |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.10+
pip install -r requirements.txt
git config core.hooksPath .githooks                   # strip AI attribution from commits
cp .env.example .env                                   # add GEMINI_API_KEY (or another provider)
python -m app.ingest                                   # build the index
uvicorn app.api:app --reload                           # http://127.0.0.1:8000/docs
```

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "How do I hide a query parameter from the OpenAPI docs?"}'
```

## Evaluation

```bash
pytest -q                                                   # unit tests, no keys needed
python eval/run_eval.py --label baseline --retrieval-only   # free: hit rate + MRR
python eval/run_eval.py --label baseline --rpm 10           # full run (throttled for free-tier limits)
```

Each run writes `results/<timestamp>_<label>.csv` (per question) and appends one row to `results/summary.csv`, so improvements are always compared run-to-run.
Stop the API server before running the eval (local Qdrant takes a file lock).

## Roadmap
- [x] Baseline RAG + eval harness
- [ ] Semantic cache with corpus-versioned invalidation and a threshold tuned on paraphrase / near-miss pairs
- [ ] Cost-aware routing (small vs large model) with escalation on weak answers
- [ ] Dashboard: cost/request, cache hit rate, p50/p95, quality per route

## Data
Corpus: English docs from [fastapi/fastapi](https://github.com/fastapi/fastapi) (MIT License, © Sebastián Ramírez), copied unmodified into `data/docs/`.
