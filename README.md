# docs-rag: a measured RAG baseline over the FastAPI docs

A retrieval-augmented Q&A API that answers questions about the FastAPI documentation with **cited sources**, plus a **repeatable evaluation harness** so every later change (caching, routing, re-ranking) can be measured against a fixed baseline.

## Results (baseline)

Run `20261118-234758_baseline` (see `results/summary.csv`): answers by `groq/openai/gpt-oss-120b`, graded by `groq/openai/gpt-oss-20b`.

| Metric | Value |
|---|---|
| Retrieval hit rate @5 | **93.5%** (58 / 62 answerable questions) |
| MRR | **0.78** |
| Answer accuracy (LLM judge, 65 questions) | **92.3%** |
| Citation rate (cites the expected doc) | **71%** |
| Cost per request | **$0.00050** ($0.033 for the full 65-question run) |
| p50 / p95 end-to-end latency | **1.40 s / 2.39 s** |

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
| LLM | Any LiteLLM model, default `groq/openai/gpt-oss-120b` (judge: `gpt-oss-20b`) |
| API | FastAPI: `GET /health`, `POST /ask`, `POST /search` |
| Eval | Hit rate, MRR, citation rate, LLM-judge correctness, cost, p50/p95 latency to CSV |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.10+
pip install -r requirements.txt
git config core.hooksPath .githooks                   # strip AI attribution from commits
cp .env.example .env                                   # add GROQ_API_KEY (or another provider)
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
python eval/run_eval.py --label baseline --rpm 4            # full run (throttled for Groq free-tier limits)
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
