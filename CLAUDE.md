# Project context for Claude Code

Baseline RAG over the FastAPI docs (120 markdown files in `data/docs/`) plus a repeatable eval.
This baseline is "Step 0" for later work: semantic caching + cost-aware routing (Project 1).

## Layout
- `app/config.py`  - all knobs, env-overridable (.env)
- `app/ingest.py`  - clean markdown -> ~500-token chunks (75 overlap, tiktoken) -> MiniLM embeddings -> local Qdrant (`qdrant_data/`)
- `app/rag.py`     - retrieve top-k -> LiteLLM answer with [n] citations; returns tokens, cost, latency
- `app/api.py`     - FastAPI: GET /health, POST /ask, POST /search
- `eval/build_eval_set.py` - source of truth for the 65-question eval set; verifies each `evidence` string exists in its source doc
- `eval/run_eval.py` - scores hit rate, MRR, citation rate, LLM-judge accuracy, cost, p50/p95 -> `results/`

## Rules
- Git attribution: never add `Co-Authored-By`, `Claude-Session`, "Generated with Claude Code", or any other Claude/AI attribution to commit messages or PR descriptions. Commits are authored solely by the repo owner.
- Do not edit `eval/eval_set.jsonl` by hand: edit `eval/build_eval_set.py` and re-run it.
- Local Qdrant is file-locked: stop the API server before running `eval/run_eval.py` (or set QDRANT_URL to a Qdrant server).
- After any change to chunking/retrieval/prompting: bump CORPUS_VERSION if docs changed, re-ingest, re-run the eval, and compare against `results/summary.csv`.
- Never commit `.env`.

## Commands
```
git config core.hooksPath .githooks
pip install -r requirements.txt
python -m app.ingest
pytest -q
python eval/run_eval.py --label baseline --retrieval-only
python eval/run_eval.py --label baseline [--rpm 10]
uvicorn app.api:app --reload
```
