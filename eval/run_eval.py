"""Run every eval question through the pipeline and score it.

Scores per question:
  - retrieval_hit : expected source doc appears in the top-k retrieved chunks
  - reciprocal_rank: 1/rank of the first chunk from the expected doc (0 if missing)
  - citation_hit  : the answer cites the expected doc
  - correctness   : LLM judge vs the reference answer (1 / 0.5 / 0)
Plus tokens, cost and latency. Writes results/<run>.csv and appends a row to results/summary.csv.

Usage:
  python eval/run_eval.py --label baseline                 # full run (needs an LLM API key)
  python eval/run_eval.py --label baseline --retrieval-only  # no LLM calls, free
  python eval/run_eval.py --limit 10 --rpm 10              # quick smoke test, throttled for free tiers
"""
import argparse
import csv
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import litellm  # noqa: E402

from app import config  # noqa: E402
from app.rag import answer, retrieve  # noqa: E402

RESULTS = ROOT / "results"

JUDGE_PROMPT = """You are grading an answer from a documentation Q&A bot.

Question: {question}
Reference answer: {reference}
Bot answer: {answer}

Grade the bot answer against the reference:
- "correct": contains the key facts of the reference and nothing that contradicts it
- "partial": some key facts present but important ones missing or vague
- "incorrect": wrong, contradicts the reference, or does not answer
If the reference says "I don't know", the bot answer is "correct" only if it declines to answer.

Respond with JSON only: {{"grade": "correct" | "partial" | "incorrect", "reason": "<one sentence>"}}"""

SUMMARY_FIELDS = [
    "run_id", "label", "n", "top_k", "corpus_version", "embed_model", "chunk_tokens",
    "hit_rate", "mrr", "p50_ms", "p95_ms",
    "llm_model", "judge_model", "accuracy", "citation_rate", "cost_total_usd", "cost_per_req_usd",
]

GRADE_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def with_backoff(fn, *args, max_tries=30, **kwargs):
    """Retry on provider rate limits, sleeping as long as the provider asks (capped at 65s)."""
    import re
    for attempt in range(1, max_tries + 1):
        try:
            return fn(*args, **kwargs)
        except litellm.RateLimitError as e:
            m = re.search(r"try again in ([0-9.]+)s", str(e))
            wait = min(65.0, float(m.group(1)) + 1 if m else 20.0)
            print(f"   rate limited (attempt {attempt}/{max_tries}), sleeping {wait:.0f}s")
            time.sleep(wait)
    return fn(*args, **kwargs)


def judge(question: str, reference: str, bot_answer: str) -> tuple[float, str]:
    resp = litellm.completion(
        model=config.JUDGE_MODEL,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, reference=reference, answer=bot_answer)}],
        temperature=0, max_tokens=1000, num_retries=3, timeout=60,
    )
    text = (resp.choices[0].message.content or "").strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(text)
        return GRADE_SCORE.get(str(data.get("grade", "")).lower(), 0.0), data.get("reason", "")
    except json.JSONDecodeError:
        return 0.0, f"unparseable judge output: {text[:120]}"


def reciprocal_rank(expected: str | None, retrieved_docs: list[str]) -> float:
    if expected is None:
        return 0.0
    for i, d in enumerate(retrieved_docs, start=1):
        if d == expected:
            return 1.0 / i
    return 0.0


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run")
    ap.add_argument("--eval-file", default=str(ROOT / "eval" / "eval_set.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--rpm", type=float, default=0, help="max LLM calls per minute (0 = no throttle)")
    ap.add_argument("--k", type=int, default=config.TOP_K)
    args = ap.parse_args()

    items = [json.loads(line) for line in open(args.eval_file, encoding="utf-8") if line.strip()]
    if args.limit:
        items = items[: args.limit]
    gap = 60.0 / args.rpm if args.rpm else 0.0

    RESULTS.mkdir(exist_ok=True)
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{args.label}"
    rows = []
    for n, it in enumerate(items, start=1):
        expected = it["source_doc"]
        if args.retrieval_only:
            t0 = time.perf_counter()
            docs = [c["doc_id"] for c in retrieve(it["question"], args.k)]
            row = {"latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
        else:
            t_call = time.perf_counter()
            res = with_backoff(answer, it["question"], k=args.k)
            docs = [r["doc_id"] for r in res["retrieved"]]
            if gap:
                time.sleep(max(0.0, gap - (time.perf_counter() - t_call)))
            score, reason = with_backoff(judge, it["question"], it["reference_answer"], res["answer"])
            if gap:
                time.sleep(gap)
            row = {
                "correctness": score, "judge_reason": reason,
                "citation_hit": int(expected in res["citations"]) if expected else "",
                "model": res["model"], "tokens_in": res["tokens_in"], "tokens_out": res["tokens_out"],
                "cost_usd": res["cost_usd"], "latency_ms": res["latency_ms"],
                "answer": res["answer"].replace("\n", " "),
            }
        rr = reciprocal_rank(expected, docs)
        rows.append({
            "id": it["id"], "category": it["category"], "question": it["question"],
            "source_doc": expected or "",
            "retrieval_hit": int(rr > 0) if expected else "",
            "reciprocal_rank": round(rr, 3) if expected else "",
            "retrieved_docs": "|".join(docs),
            **row,
        })
        status = f"hit={rows[-1]['retrieval_hit']}"
        if not args.retrieval_only:
            status += f" score={rows[-1]['correctness']}"
        print(f"[{n}/{len(items)}] {it['id']} {status}")

    fields = list(dict.fromkeys(k for r in rows for k in r))
    out = RESULTS / f"{run_id}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    answerable = [r for r in rows if r["source_doc"]]
    lat = [float(r["latency_ms"]) for r in rows]
    summary = {
        "run_id": run_id, "label": args.label, "n": len(rows), "top_k": args.k,
        "corpus_version": config.CORPUS_VERSION, "embed_model": config.EMBED_MODEL,
        "chunk_tokens": config.CHUNK_TOKENS,
        "hit_rate": round(statistics.mean(r["retrieval_hit"] for r in answerable), 3) if answerable else "",
        "mrr": round(statistics.mean(r["reciprocal_rank"] for r in answerable), 3) if answerable else "",
        "p50_ms": round(pct(lat, 50), 1), "p95_ms": round(pct(lat, 95), 1),
    }
    if not args.retrieval_only:
        cost = [float(r["cost_usd"]) for r in rows]
        summary.update({
            "llm_model": config.LLM_MODEL, "judge_model": config.JUDGE_MODEL,
            "accuracy": round(statistics.mean(r["correctness"] for r in rows), 3),
            "citation_rate": round(statistics.mean(r["citation_hit"] for r in answerable), 3) if answerable else "",
            "cost_total_usd": round(sum(cost), 6),
            "cost_per_req_usd": round(statistics.mean(cost), 8),
        })
    summary_file = RESULTS / "summary.csv"
    new = not summary_file.exists()
    with summary_file.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore", restval="")
        if new:
            w.writeheader()
        w.writerow(summary)

    print("\n" + json.dumps(summary, indent=2))
    print(f"\nPer-question results: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
