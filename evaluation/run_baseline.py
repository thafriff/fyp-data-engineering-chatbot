"""
Baseline runner: answer all evaluation questions with the dev prototype's RAG pipeline and save the raw results.

This is the "before" measurement for the report. It calls the real pipeline (src/core/rag.py ->
RagPipeline.answer), so retrieval, spelling correction of the search query, the SIMILARITY_THRESHOLD fallback,
the prompt builder and the runaway-answer guards in llm.stream_generate are exactly the chatbot's. The only
difference: the dev-only intent routing (small talk / unclear / emotional classifier) is skipped, because every
evaluation question is a real course question. It is skipped by replacing intent.route with a stub for this
process only; no project code is changed. No answer is scored here (no LLM judging); the only metric computed
is retrieval hit@5 against relevant_chunk_ids.

The output file is rewritten after every question, and an interrupted run resumes: questions already saved
without an error are skipped (questions that errored are retried).

Usage (from the project root):
    venv\\Scripts\\python evaluation\\run_baseline.py [--model llama3.2:3b] [--limit N] [--out PATH]
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make the project root importable (config/, src/) when the script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings  # noqa: E402
from src.core import intent, llm  # noqa: E402
from src.core.intent import Intent  # noqa: E402
from src.core.rag import RagPipeline  # noqa: E402

BASELINE_DIR = settings.TEST_QUESTIONS_FILE.parent / "baselines"
# Run settings that must match when resuming, otherwise results from different setups would be mixed.
RESUME_MUST_MATCH = ["model", "top_k", "similarity_threshold", "embedding_model", "max_answer_tokens",
                     "llm_temperature", "num_ctx", "spell_correction"]


def parse_args() -> argparse.Namespace:
    """Command-line options: model, how many questions, output path."""
    parser = argparse.ArgumentParser(description="Run the evaluation questions through the RAG pipeline.")
    parser.add_argument("--model", default=settings.DEV_LLM_MODEL, help="local Ollama model for the answers")
    parser.add_argument("--limit", type=int, default=None, help="run only the first N questions")
    parser.add_argument("--out", type=Path, default=None,
                        help="output JSON (default evaluation/baselines/baseline_<model>.json)")
    args = parser.parse_args()
    if args.out is None:
        args.out = BASELINE_DIR / f"baseline_{args.model.replace(':', '_')}.json"
    return args


def git_commit() -> dict:
    """Current commit hash and whether tracked files have uncommitted changes."""
    def run(*cmd):
        return subprocess.run(["git", *cmd], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
    try:
        return {"commit": run("rev-parse", "HEAD") or None,
                "tracked_changes": bool(run("status", "--porcelain", "--untracked-files=no"))}
    except OSError:
        return {"commit": None, "tracked_changes": None}


def run_settings(model: str) -> dict:
    """Everything needed to reproduce or compare this run."""
    return {
        "model": model,
        "top_k": settings.TOP_K,
        "similarity_threshold": settings.SIMILARITY_THRESHOLD,
        "embedding_model": settings.EMBEDDING_MODEL,
        "llm_temperature": settings.LLM_TEMPERATURE,
        "num_ctx": settings.DEV_LLM_NUM_CTX,
        "max_answer_tokens": settings.DEV_LLM_MAX_ANSWER_TOKENS,
        "spell_correction": settings.SPELL_CORRECTION_ENABLED,
        "intent_routing": "skipped (all evaluation questions are real course questions)",
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_commit(),
        "questions_file": str(settings.TEST_QUESTIONS_FILE.relative_to(PROJECT_ROOT)),
    }


def load_existing(out: Path, settings_now: dict) -> list[dict]:
    """Results already saved by an earlier (interrupted) run with the same settings."""
    if not out.exists():
        return []
    previous = json.loads(out.read_text(encoding="utf-8"))
    old = previous.get("run_settings", {})
    mismatched = [k for k in RESUME_MUST_MATCH if old.get(k) != settings_now[k]]
    if mismatched:
        sys.exit(f"{out} was made with different settings ({', '.join(mismatched)}); "
                 f"use another --out or delete the file.")
    return previous.get("results", [])


def stop_reason(answer: str) -> str:
    """Why generation stopped: a runaway-answer guard, or the model finished by itself."""
    if answer.endswith(settings.ANSWER_REPEAT_NOTE):
        return "repetition_guard"
    if answer.endswith(settings.ANSWER_CUT_NOTE):
        return "length_cap"
    return "finished"


def answer_question(pipeline: RagPipeline, q: dict, model: str) -> dict:
    """Run one question through the pipeline and return its result record."""
    record = {"id": q["id"], "question": q["question"], "answer": None, "fallback": None, "stopped_by": None,
              "search_query": None, "retrieved": [], "hit_at_5": None, "seconds": None, "error": None}
    start = time.perf_counter()
    try:
        result = pipeline.answer(q["question"], model=model)
        record["retrieved"] = [{"chunk_id": c.chunk_id, "score": round(c.score, 4)} for c in result.chunks]
        record["search_query"] = result.search_query
        record["fallback"] = result.used_fallback
        record["answer"] = "".join(result.answer_stream)        # generation happens while this is consumed
        record["stopped_by"] = "fallback" if result.used_fallback else stop_reason(record["answer"])
        retrieved_ids = {c["chunk_id"] for c in record["retrieved"]}
        record["hit_at_5"] = bool(retrieved_ids & set(q["relevant_chunk_ids"]))
    except Exception as e:  # record the failure and keep going; it is retried on the next run
        record["error"] = f"{type(e).__name__}: {e}"
    record["seconds"] = round(time.perf_counter() - start, 2)
    return record


def summarize(results: list[dict]) -> dict:
    """Counts and means over the saved results (questions that errored are excluded from the means)."""
    ok = [r for r in results if not r["error"]]
    return {
        "questions_run": len(results),
        "errors": len(results) - len(ok),
        "fallback_count": sum(1 for r in ok if r["fallback"]),
        "mean_seconds": round(statistics.mean(r["seconds"] for r in ok), 2) if ok else None,
        "hit_at_5": round(sum(r["hit_at_5"] for r in ok) / len(ok), 3) if ok else None,
        "stopped_by": {k: sum(1 for r in ok if r["stopped_by"] == k)
                       for k in ("finished", "fallback", "repetition_guard", "length_cap")},
    }


def save(out: Path, settings_now: dict, results: list[dict]) -> None:
    """Write the whole file atomically (temp file + rename) so an interruption cannot corrupt it."""
    out.parent.mkdir(parents=True, exist_ok=True)
    order = {f"Q{i:02d}": i for i in range(1, 1000)}
    results = sorted(results, key=lambda r: order.get(r["id"], 10**6))
    data = {"run_settings": settings_now, "summary": summarize(results), "results": results}
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(out)


def main() -> None:
    """Run the questions, saving after each one, then print a summary."""
    args = parse_args()
    ok, message = llm.check_backend(args.model)
    if not ok:
        sys.exit(message)

    questions = json.loads(settings.TEST_QUESTIONS_FILE.read_text(encoding="utf-8"))["questions"]
    if args.limit is not None:
        questions = questions[:args.limit]

    settings_now = run_settings(args.model)
    results = {r["id"]: r for r in load_existing(args.out, settings_now)}
    todo = [q for q in questions if q["id"] not in results or results[q["id"]]["error"]]
    print(f"Model {args.model} | {len(questions)} questions selected, {len(questions) - len(todo)} already done, "
          f"{len(todo)} to run | output {args.out}")

    # Skip the dev-only intent routing for this process: every evaluation question is a real course question.
    intent.route = lambda message, top_score=None: Intent("question", path="baseline_no_routing")
    pipeline = RagPipeline()

    for n, q in enumerate(todo, 1):
        record = answer_question(pipeline, q, args.model)
        results[q["id"]] = record
        save(args.out, settings_now, list(results.values()))
        status = f"ERROR {record['error']}" if record["error"] else (
            f"fallback={'yes' if record['fallback'] else 'no'} hit@5={'yes' if record['hit_at_5'] else 'no'} "
            f"stopped_by={record['stopped_by']}")
        print(f"[{n}/{len(todo)}] {q['id']} {record['seconds']:.1f}s {status}", flush=True)

    s = summarize(list(results.values()))
    print(f"\nSummary: {s['questions_run']} questions | errors {s['errors']} | fallback {s['fallback_count']} | "
          f"mean {s['mean_seconds']} s | hit@5 {s['hit_at_5']} | stopped_by {s['stopped_by']}")


if __name__ == "__main__":
    main()
