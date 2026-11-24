"""
Baseline runner: answer all evaluation questions with the dev prototype's RAG pipeline and save the raw results.

This is the "before" measurement for the report. It calls the real pipeline (src/core/rag.py ->
RagPipeline.answer), so retrieval, spelling correction of the search query, the SIMILARITY_THRESHOLD fallback,
the prompt builder and the runaway-answer guards in llm.stream_generate are exactly the chatbot's. The only
difference: the dev-only intent routing (small talk / unclear / emotional classifier) is skipped, because every
evaluation question is a real course question. It is skipped by replacing intent.route with a stub for this
process only; no project code is changed. No answer is scored here (no LLM judging); the only metric computed
is retrieval hit@5 against relevant_chunk_ids.

Each request has a wall-clock limit (settings.NB04_REQUEST_TIMEOUT_SECONDS). A timed-out request is abandoned, recorded
as an error with stopped_by "timeout" and its elapsed seconds, left out of the time statistics, and retried on the next run.

The output file is rewritten after every question, and an interrupted run resumes: questions already saved
without an error are skipped (questions that errored are retried).

With --contexts, the same runner reads the frozen chunks and prompts of notebook 04 instead of searching again, and also
answers the guardrail probes (no hit@5 for those). Each prompt is rebuilt by the real prompt builder and must match its
frozen SHA-256, otherwise the run stops. The SYSTEM_PROMPT hash and version are in the run settings, and a run cannot
resume or overwrite a file made under another prompt.

Usage (from the project root):
    venv\\Scripts\\python evaluation\\run_baseline.py [--model llama3.2:3b] [--limit N] [--out PATH]
    venv\\Scripts\\python evaluation\\run_baseline.py --contexts evaluation\\notebook04\\contexts.json --out PATH
"""

import argparse
import hashlib
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
from src.core.search_text import spelling_corrections  # noqa: E402
from src.core.rag import RagPipeline  # noqa: E402
from src.core.retriever import RetrievedChunk  # noqa: E402
from src.core.timeouts import RequestTimedOut, collect_stream  # noqa: E402

BASELINE_DIR = settings.TEST_QUESTIONS_FILE.parent / "baselines"
# Run settings that must match when resuming, otherwise results from different setups would be mixed.
RESUME_MUST_MATCH = ["model", "top_k", "similarity_threshold", "embedding_model", "max_answer_tokens",
                     "llm_temperature", "num_ctx", "spell_correction", "system_prompt_sha256", "contexts_file",
                     "request_timeout_seconds"]


def parse_args() -> argparse.Namespace:
    """Command-line options: model, how many questions, output path."""
    parser = argparse.ArgumentParser(description="Run the evaluation questions through the RAG pipeline.")
    parser.add_argument("--model", default=settings.DEV_LLM_MODEL, help="local Ollama model for the answers")
    parser.add_argument("--limit", type=int, default=None, help="run only the first N questions")
    parser.add_argument("--out", type=Path, default=None,
                        help="output JSON (default evaluation/baselines/baseline_<model>.json)")
    parser.add_argument("--contexts", type=Path, default=None,
                        help="frozen contexts file: answer its items from its chunks instead of searching again")
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


def run_settings(model: str, contexts: Path | None = None) -> dict:
    """Everything needed to reproduce or compare this run."""
    return {
        "model": model,
        "top_k": settings.TOP_K,
        "similarity_threshold": settings.SIMILARITY_THRESHOLD,
        "embedding_model": settings.EMBEDDING_MODEL,
        "llm_temperature": settings.LLM_TEMPERATURE,
        "num_ctx": settings.DEV_LLM_NUM_CTX,
        "max_answer_tokens": settings.DEV_LLM_MAX_ANSWER_TOKENS,
        "request_timeout_seconds": settings.NB04_REQUEST_TIMEOUT_SECONDS,
        "spell_correction": settings.SPELL_CORRECTION_ENABLED,
        "ollama_think": False if model in settings.DEV_LLM_THINK_OFF_MODELS else "not sent (model default)",
        "intent_routing": "skipped (all evaluation questions are real course questions)",
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_commit(),
        "questions_file": str(settings.TEST_QUESTIONS_FILE.relative_to(PROJECT_ROOT)),
        "system_prompt_version": settings.SYSTEM_PROMPT_VERSION,
        "system_prompt_sha256": hashlib.sha256(settings.SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "contexts_file": contexts.resolve().relative_to(PROJECT_ROOT).as_posix() if contexts else None,
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
        # generation happens while this is consumed
        record["answer"] = collect_stream(lambda: result.answer_stream, settings.NB04_REQUEST_TIMEOUT_SECONDS)
        record["stopped_by"] = "fallback" if result.used_fallback else stop_reason(record["answer"])
        retrieved_ids = {c["chunk_id"] for c in record["retrieved"]}
        record["hit_at_5"] = bool(retrieved_ids & set(q["relevant_chunk_ids"]))
    except RequestTimedOut as e:    # abandoned and retried on the next run; seconds stays None
        record.update(stopped_by="timeout", error=str(e), elapsed_seconds=round(e.elapsed_seconds, 2))
        return record
    except Exception as e:  # record the failure and keep going; it is retried on the next run
        record["error"] = f"{type(e).__name__}: {e}"
    record["seconds"] = round(time.perf_counter() - start, 2)
    return record


def answer_frozen_item(item: dict, relevant_ids: set | None, model: str) -> dict:
    """Answer one frozen item from its stored chunks, with the same generator as the chatbot. Stops the whole run if the
    rebuilt prompt is not the frozen one, because the results would then not belong to these contexts."""
    chunks = [RetrievedChunk(**c) for c in item["chunks"]]
    corrections = spelling_corrections(item["text"], item["search_query"], [c.text for c in chunks])   # the hint, as frozen
    rebuilt = json.dumps(llm.build_messages(item["text"], chunks, corrections=corrections), ensure_ascii=False, sort_keys=True)
    if hashlib.sha256(rebuilt.encode("utf-8")).hexdigest() != item["prompt_sha256"]:
        sys.exit(f"{item['id']}: the rendered prompt differs from the frozen one. Re-run freeze_contexts.py and "
                 f"check_parity.py first.")
    ids = {c.chunk_id for c in chunks}
    record = {"id": item["id"], "kind": item["kind"], "question": item["text"], "answer": None, "fallback": False,
              "stopped_by": None, "search_query": item["search_query"], "prompt_sha256": item["prompt_sha256"],
              "retrieved": [{"chunk_id": c.chunk_id, "score": c.score} for c in chunks],
              "hit_at_5": bool(ids & relevant_ids) if relevant_ids is not None else None,
              "seconds": None, "error": None}
    start = time.perf_counter()
    try:
        record["answer"] = collect_stream(lambda: llm.stream_generate(item["text"], chunks, model=model, corrections=corrections),
                                          settings.NB04_REQUEST_TIMEOUT_SECONDS)
        record["stopped_by"] = stop_reason(record["answer"])
    except RequestTimedOut as e:    # abandoned and retried on the next run; seconds stays None
        record.update(stopped_by="timeout", error=str(e), elapsed_seconds=round(e.elapsed_seconds, 2))
        return record
    except Exception as e:  # record the failure and keep going; it is retried on the next run
        record["error"] = f"{type(e).__name__}: {e}"
    record["seconds"] = round(time.perf_counter() - start, 2)
    return record


def run_frozen(args: argparse.Namespace, settings_now: dict) -> None:
    """--contexts mode: every frozen item, answered from its frozen chunks, saved after each one."""
    items = json.loads(args.contexts.read_text(encoding="utf-8"))["items"]
    if args.limit is not None:
        items = items[:args.limit]
    relevant = {q["id"]: set(q["relevant_chunk_ids"])
                for q in json.loads(settings.TEST_QUESTIONS_FILE.read_text(encoding="utf-8"))["questions"]}
    results = {r["id"]: r for r in load_existing(args.out, settings_now)}
    todo = [i for i in items if i["id"] not in results or results[i["id"]]["error"]]
    print(f"Model {args.model} | {len(items)} frozen items, {len(items) - len(todo)} already done, {len(todo)} to run "
          f"| output {args.out}")
    for n, item in enumerate(todo, 1):
        record = answer_frozen_item(item, relevant.get(item["id"]), args.model)
        results[item["id"]] = record
        save(args.out, settings_now, list(results.values()))
        status = f"ERROR {record['error']}" if record["error"] else f"stopped_by={record['stopped_by']}"
        print(f"[{n}/{len(todo)}] {item['id']} {record.get('elapsed_seconds', record['seconds'])}s {status}", flush=True)
    s = summarize(list(results.values()))
    print(f"\nSummary: {s['items_run']} items | errors {s['errors']} | fallback {s['fallback_count']} | "
          f"mean {s['mean_seconds']} s | hit@5 {s['hit_at_5']} | stopped_by {s['stopped_by']}")


def summarize(results: list[dict]) -> dict:
    """Counts and means over the saved results (questions that errored are excluded from the means)."""
    ok = [r for r in results if not r["error"]]
    scored = [r for r in ok if r["hit_at_5"] is not None]    # guardrail probes have no relevant chunks to hit
    return {
        "items_run": len(results),
        "questions_run": sum(r.get("kind", "question") == "question" for r in results),
        "probes_run": sum(r.get("kind") == "probe" for r in results),
        "errors": len(results) - len(ok),
        "timeouts": sum(r.get("stopped_by") == "timeout" for r in results),
        "fallback_count": sum(1 for r in ok if r["fallback"]),
        "mean_seconds": round(statistics.mean(r["seconds"] for r in ok), 2) if ok else None,
        "hit_at_5": round(sum(r["hit_at_5"] for r in scored) / len(scored), 3) if scored else None,
        "stopped_by": {k: sum(1 for r in ok if r["stopped_by"] == k)
                       for k in ("finished", "fallback", "repetition_guard", "length_cap")},
    }


def save(out: Path, settings_now: dict, results: list[dict]) -> None:
    """Write the whole file atomically (temp file + rename) so an interruption cannot corrupt it."""
    out.parent.mkdir(parents=True, exist_ok=True)
    order = {f"Q{i:02d}": i for i in range(1, 1000)}
    results = sorted(results, key=lambda r: (order.get(r["id"], 10**6), r["id"]))
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

    if args.contexts:
        run_frozen(args, run_settings(args.model, args.contexts))
        return

    questions = json.loads(settings.TEST_QUESTIONS_FILE.read_text(encoding="utf-8"))["questions"]
    if args.limit is not None:
        questions = questions[:args.limit]

    settings_now = run_settings(args.model)
    results = {r["id"]: r for r in load_existing(args.out, settings_now)}
    todo = [q for q in questions if q["id"] not in results or results[q["id"]]["error"]]
    print(f"Model {args.model} | {len(questions)} questions selected, {len(questions) - len(todo)} already done, "
          f"{len(todo)} to run | output {args.out}")

    # Skip the dev-only intent routing for this process: every evaluation question is a real course question.
    intent.route = lambda message, top_score=None, clean=None: Intent("question", path="baseline_no_routing")
    pipeline = RagPipeline()

    for n, q in enumerate(todo, 1):
        record = answer_question(pipeline, q, args.model)
        results[q["id"]] = record
        save(args.out, settings_now, list(results.values()))
        status = f"ERROR {record['error']}" if record["error"] else (
            f"fallback={'yes' if record['fallback'] else 'no'} hit@5={'yes' if record['hit_at_5'] else 'no'} "
            f"stopped_by={record['stopped_by']}")
        print(f"[{n}/{len(todo)}] {q['id']} {record.get('elapsed_seconds', record['seconds'])}s {status}", flush=True)

    s = summarize(list(results.values()))
    print(f"\nSummary: {s['items_run']} questions | errors {s['errors']} | fallback {s['fallback_count']} | "
          f"mean {s['mean_seconds']} s | hit@5 {s['hit_at_5']} | stopped_by {s['stopped_by']}")


if __name__ == "__main__":
    main()
