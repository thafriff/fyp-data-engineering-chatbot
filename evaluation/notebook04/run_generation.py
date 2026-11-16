"""
Generation runner for notebook 04: sends each frozen prompt to one model and caches the answers. No scoring happens here.

Rules from docs/notebook04_plan.md:
- check_parity.py runs first; any failure aborts before a single call;
- one cache per model in evaluation/notebook04/raw/, rewritten atomically after every answer; a rerun resumes, skipping
  saved answers, and refuses to resume under different settings. Prompts are checked per item: an answer is reused only
  while its prompt hash equals its item's current one; a stale answer is moved to "superseded" and generated again;
- every answer is capped at 800 tokens and passed through the repetition guard (same rule as llm.stream_generate,
  applied to the returned text because these calls do not stream); the guard's note is not added to the answer,
  because stopped_by records it and the judges should see only what the model wrote;
- a model that cannot turn thinking off gets a reasoning allowance on top of the 800, so reasoning cannot take answer
  tokens; its answer is still cut to the 800-token cap, and cost still uses every token the API returns;
- --report prints each model's truncation count and rate, flags a model far above the others, and prints the timing evidence
  for Response Time: successful-request latency (mean and p95), timed-out attempts, total time including those attempts,
  the true elapsed total (every attempt and every wait), and each model's run window in UTC;
- latency covers only the successful request; 429 and temporary-failure (5xx, connection) waits are recorded apart from it
  and from each other. A 429 uses the host's own requested wait when it gives one (a Retry-After header, or Google's retryDelay
  in the error body), otherwise a longer back-off than a 5xx;
- every call, successful or not, is one entry in the item's call_log: UTC start and finish, seconds, outcome, the wait that
  followed it and, for a failure, the error type, HTTP status and text. The log is kept across reruns;
- each request has a wall-clock limit (NB04_REQUEST_TIMEOUT_SECONDS). A timed-out request is abandoned, recorded as an error
  with stopped_by "timeout" and its elapsed seconds, kept out of latency, and retried on the next run. Every timed-out attempt
  stays in the item's timed_out_attempts list, even after a later attempt succeeds, so that a stall cannot vanish from the record;
- each probe runs NB04_PROBE_REPEATS times, under result keys such as "G01#r2"; contexts.json still holds it once;
- tokens come from the API; cost is tokens x list price, also on a free tier (the free_tier flag says so);
- hard stop: a model above 120% of its estimate, or all spend reaching settings.NB04_TOTAL_STOP_MYR. State is saved first,
  and the run exits.

Usage (from the project root; needs the model's key in .env):
    venv\\Scripts\\python evaluation\\notebook04\\run_generation.py --model gpt-4o [--only Q01,Q02,Q03]
    venv\\Scripts\\python evaluation\\notebook04\\run_generation.py --report
Exit codes: 0 finished, 1 parity or setup failure, 2 hard stop, 3 rate limit not cleared or access blocked.
"""

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import check_parity  # noqa: E402
import providers  # noqa: E402
from config import settings  # noqa: E402
from src.core.timeouts import RequestTimedOut, call_with_timeout  # noqa: E402

EXIT_SETUP, EXIT_HARD_STOP, EXIT_BLOCKED = 1, 2, 3
# Settings that must match to resume; date and git commit are recorded but may differ between sittings.
# The prompts are NOT in this list: they are checked per item (answer_is_current), so changing one item's prompt makes only
# that item's cached answer stale. prompt_set_sha256 is still recorded in the run settings, for information.
RESUME_MUST_MATCH = ["model_key", "model_id", "api", "base_url", "temperature", "max_answer_tokens", "request_options",
                     "reasoning_allowance_tokens", "request_timeout_seconds", "probe_repeats",
                     "input_usd_per_m", "output_usd_per_m", "top_k", "similarity_threshold", "embedding_model",
                     "repeat_line_limit", "repeat_min_chars"]


def estimated_spend_usd(candidate: dict) -> float:
    """List-price estimate for a model's full run (plan section 8); the base of its 120% stop."""
    allowance = candidate.get("token_allowance", 1.0)
    output_tokens = candidate.get("est_output_tokens", settings.NB04_EST_OUTPUT_TOKENS)
    per_answer = allowance * (settings.NB04_EST_INPUT_TOKENS * candidate["input_usd_per_m"]
                              + output_tokens * candidate["output_usd_per_m"]) / 1_000_000
    return per_answer * settings.NB04_ANSWERS_PER_MODEL


def apply_repetition_guard(text: str) -> tuple[str, bool]:
    """Cut the answer just after a line of at least DEV_LLM_REPEAT_MIN_CHARS characters appears for the
    DEV_LLM_REPEAT_LINE_LIMIT-th time (whitespace-insensitive). Returns the text and whether it was cut."""
    counts: dict[str, int] = {}
    position = 0
    for line in text.splitlines(keepends=True):
        position += len(line)
        if not line.endswith(("\n", "\r")):      # same as the streamed guard: only finished lines count
            continue
        key = " ".join(line.split())
        if len(key) >= settings.DEV_LLM_REPEAT_MIN_CHARS:
            counts[key] = counts.get(key, 0) + 1
            if counts[key] >= settings.DEV_LLM_REPEAT_LINE_LIMIT:
                return text[:position].rstrip(), True
    return text, False


def cut_to_answer_cap(text: str, answer_tokens: int) -> str:
    """Cut an answer that went past the cap, in proportion to its token count, at the last space. Only a model with a
    reasoning allowance can get here. It is approximate, because the calls do not stream and tokens cannot be counted
    locally for every model."""
    keep = int(len(text) * settings.NB04_MAX_ANSWER_TOKENS / answer_tokens)
    return text[:keep].rsplit(" ", 1)[0].rstrip()


def stop_label(finish: str, repeated: bool, capped: bool, answer_tokens: int, reasoning_tokens: int) -> str:
    """Why the answer ended. reasoning_budget_exhausted means reasoning used up the allowance before the answer
    reached its cap, which the allowance is meant to prevent, so it counts as truncation and is reported."""
    if repeated:
        return "repetition_guard"
    if capped:
        return "length_cap"
    if finish == "length":
        if reasoning_tokens and answer_tokens < settings.NB04_MAX_ANSWER_TOKENS:
            return "reasoning_budget_exhausted"
        return "length_cap"
    return {"stop": "finished", "refusal": "provider_refusal"}.get(finish, finish)


def truncation_report() -> list[dict]:
    """Per model: answers, truncated answers, rate, and mean reasoning tokens, from the caches on disk. A model is flagged
    when its rate exceeds the median rate of the other models by NB04_TRUNCATION_FLAG_POINTS."""
    rows = []
    for path in sorted(settings.NB04_RAW_DIR.glob("*.json")) if settings.NB04_RAW_DIR.exists() else []:
        cache = json.loads(path.read_text(encoding="utf-8"))
        done = [r for r in cache["results"].values() if not r["error"]]
        timeouts = sum(r["stopped_by"] == "timeout" for r in cache["results"].values())
        length_cap = sum(r["stopped_by"] == "length_cap" for r in done)
        exhausted = sum(r["stopped_by"] == "reasoning_budget_exhausted" for r in done)
        rows.append({"model": path.stem, "answers": len(done), "length_cap": length_cap,
                     "reasoning_budget_exhausted": exhausted, "unresolved_timeouts": timeouts,
                     "truncation_rate": round((length_cap + exhausted) / len(done), 4) if done else 0.0,
                     "mean_reasoning_tokens": round(statistics.mean(r["reasoning_tokens"] or 0 for r in done), 1)
                     if done else 0.0,
                     "reasoning_allowance_tokens": cache["run_settings"]["reasoning_allowance_tokens"]})
    for row in rows:
        others = [r["truncation_rate"] for r in rows if r is not row]
        row["flagged"] = bool(others) and (row["truncation_rate"]
                                           > statistics.median(others) + settings.NB04_TRUNCATION_FLAG_POINTS)
    return rows


def percentile_95(values: list[float]) -> float:
    """Nearest-rank 95th percentile."""
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _waited_and_failed_seconds(record: dict) -> float:
    """Seconds in failed attempts and back-off waits, from a record's summed fields."""
    return sum(record.get(field, 0) or 0 for field in
               ("failed_attempt_seconds", "rate_limit_wait_seconds", "temporary_failure_wait_seconds"))


def _call_log_seconds(log: list[dict]) -> float:
    """Every call in a log and the wait that followed it."""
    return sum(call["elapsed_seconds"] + call["waited_after_seconds"] for call in log)


def true_elapsed_seconds(record: dict) -> float:
    """All the time one item took across every run: each attempt, successful or not, and every wait. Records saved before
    call_log existed are added up from their summed fields, which give the same total."""
    log = record.get("call_log")
    seconds = _call_log_seconds(log or [])
    for attempt in record.get("timed_out_attempts", []):
        if "started_at" not in attempt:      # saved before call_log existed, so not in the log above
            seconds += attempt["elapsed_seconds"] + _waited_and_failed_seconds(attempt.get("waits", {}))
    if log is None and not record["error"] and record.get("latency_seconds") is not None:
        seconds += record["latency_seconds"] + _waited_and_failed_seconds(record)
    return seconds


def timing_summary(cache: dict) -> dict:
    """The Response Time evidence for one model, from its question answers only (the probes are not in that criterion).
    Latency is the successful request only, as the plan defines it. Everything spent on attempts that produced no answer is
    reported beside it: timed-out attempts, failed (429 or 5xx) attempts, and the back-off waits, 429 apart from 5xx.
    true_elapsed_seconds is what the answers really took, with every attempt and wait included."""
    questions = [r for r in cache["results"].values() if r.get("kind") == "question"]
    answered = [r for r in questions if not r["error"] and r.get("latency_seconds") is not None]
    seconds = [r["latency_seconds"] for r in answered]
    timed_out = [a for r in questions for a in r.get("timed_out_attempts", [])]
    exhausted = [e for e in cache.get("retries_exhausted", []) if e.get("kind") == "question"]
    # A timed-out attempt's waits are kept in its own entry, so unanswered records are not counted a second time.
    wait_records = answered + [a.get("waits", {}) for a in timed_out + exhausted]
    def added(field: str) -> float:
        return round(sum(r.get(field, 0) or 0 for r in wait_records), 2)
    timed_out_seconds = round(sum(a["elapsed_seconds"] for a in timed_out), 2)
    true_elapsed = (sum(true_elapsed_seconds(r) for r in questions)
                    + sum(_call_log_seconds(e["call_log"]) if "call_log" in e else _waited_and_failed_seconds(e.get("waits", {}))
                          for e in exhausted))
    return {"question_answers": len(answered),
            "latency_mean_seconds": round(statistics.mean(seconds), 2) if seconds else None,
            "latency_p95_seconds": round(percentile_95(seconds), 2) if seconds else None,
            "timed_out_attempts": len(timed_out), "timed_out_seconds": timed_out_seconds,
            "successful_request_seconds": round(sum(seconds), 2),
            "total_seconds_including_abandoned": round(sum(seconds) + timed_out_seconds, 2),
            "true_elapsed_seconds": round(true_elapsed, 2),
            "failed_attempt_seconds": added("failed_attempt_seconds"),
            "rate_limit_retries": added("rate_limit_retries"), "rate_limit_wait_seconds": added("rate_limit_wait_seconds"),
            "temporary_failure_retries": added("temporary_failure_retries"),
            "temporary_failure_wait_seconds": added("temporary_failure_wait_seconds"),
            "retries_exhausted": len(exhausted)}


def timing_report() -> list[dict]:
    """timing_summary for every model cache on disk."""
    return [{"model": path.stem, **timing_summary(json.loads(path.read_text(encoding="utf-8")))}
            for path in (sorted(settings.NB04_RAW_DIR.glob("*.json")) if settings.NB04_RAW_DIR.exists() else [])]


def run_window(cache: dict) -> dict:
    """When a model ran: the start of its first call and the end of its last, in UTC, over every call (questions, probes and
    attempts that failed). Chapter 4 states this beside Response Time, because latency also reflects the host's load at the
    time. Results saved before calls were timestamped are counted, not guessed."""
    logs = [r["call_log"] for r in cache["results"].values() if r.get("call_log")]
    logs += [e.get("call_log", []) for e in cache.get("retries_exhausted", []) + cache.get("access_blocked", [])]
    calls = [call for log in logs for call in log]
    return {"calls_timestamped": len(calls),
            "first_call_utc": min((c["started_at"] for c in calls), default=None),
            "last_call_utc": max((c["finished_at"] for c in calls), default=None),
            "results_without_timestamps": sum("call_log" not in r for r in cache["results"].values())}


def run_window_report() -> list[dict]:
    """run_window for every model cache on disk."""
    return [{"model": path.stem, **run_window(json.loads(path.read_text(encoding="utf-8")))}
            for path in (sorted(settings.NB04_RAW_DIR.glob("*.json")) if settings.NB04_RAW_DIR.exists() else [])]


def prompt_set_hash(items: list[dict]) -> str:
    """One hash over all frozen prompts, so a cache can never be resumed against different prompts."""
    return hashlib.sha256("".join(i["prompt_sha256"] for i in items).encode("ascii")).hexdigest()


def git_state() -> dict:
    """Current commit and whether tracked files have uncommitted changes."""
    def run(*cmd):
        return subprocess.run(["git", *cmd], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD") or None,
            "tracked_changes": bool(run("status", "--porcelain", "--untracked-files=no"))}


def run_settings(model_key: str, candidate: dict, prompt_set_sha256: str) -> dict:
    """Everything needed to reproduce this run."""
    return {
        "model_key": model_key, "slot": candidate["slot"], "model_id": candidate["model_id"],
        "api": candidate["api"], "base_url": candidate["base_url"],
        "temperature": candidate["temperature"], "max_answer_tokens": settings.NB04_MAX_ANSWER_TOKENS,
        "request_options": candidate["request_options"], "thinking_note": candidate["thinking_note"],
        "reasoning_allowance_tokens": candidate["reasoning_allowance_tokens"],
        "request_timeout_seconds": settings.NB04_REQUEST_TIMEOUT_SECONDS, "probe_repeats": settings.NB04_PROBE_REPEATS,
        "backoff": {"max_retries": settings.NB04_BACKOFF_MAX_RETRIES,
                    "temporary_failure_base_and_cap_seconds": [settings.NB04_BACKOFF_BASE_SECONDS, settings.NB04_BACKOFF_MAX_SECONDS],
                    "rate_limit_base_and_cap_seconds": [settings.NB04_RATE_LIMIT_BACKOFF_BASE_SECONDS,
                                                        settings.NB04_RATE_LIMIT_BACKOFF_MAX_SECONDS]},
        "system_prompt_version": settings.SYSTEM_PROMPT_VERSION,
        "input_usd_per_m": candidate["input_usd_per_m"], "output_usd_per_m": candidate["output_usd_per_m"],
        "price_status": candidate["price_status"], "prices_checked": settings.NB04_PRICES_CHECKED,
        "free_tier": candidate["free_tier"],
        "top_k": settings.TOP_K, "similarity_threshold": settings.SIMILARITY_THRESHOLD,
        "embedding_model": settings.EMBEDDING_MODEL,
        "repeat_line_limit": settings.DEV_LLM_REPEAT_LINE_LIMIT, "repeat_min_chars": settings.DEV_LLM_REPEAT_MIN_CHARS,
        "prompt_set_sha256": prompt_set_sha256,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git": git_state(),
    }


def load_cache(path: Path, settings_now: dict) -> dict:
    """The saved cache for this model, or an empty one. Exits if it was made with different settings."""
    if not path.exists():
        return {"run_settings": settings_now, "spend": {"list_usd_total": 0.0, "billed_calls": 0}, "results": {}}
    cache = json.loads(path.read_text(encoding="utf-8"))
    mismatched = [k for k in RESUME_MUST_MATCH if cache["run_settings"].get(k) != settings_now[k]]
    if mismatched:
        print(f"ABORTED: {path} was made with different settings ({', '.join(mismatched)}). No model was called. "
              f"Use another file or remove this one deliberately.")
        sys.exit(EXIT_SETUP)
    return cache


def save_cache(path: Path, cache: dict) -> None:
    """Atomic write (temp file + rename), so an interruption cannot corrupt the cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def total_spend_usd() -> float:
    """List-price spend across all model caches and judge caches, as saved on disk."""
    total = 0.0
    for folder in (settings.NB04_RAW_DIR, settings.NB04_JUDGE_DIR):
        for path in sorted(folder.glob("*.json")) if folder.exists() else []:
            total += json.loads(path.read_text(encoding="utf-8")).get("spend", {}).get("list_usd_total", 0.0)
    return total


def hard_stop_reason(model_spend: float, estimate: float) -> str | None:
    """Why the run must stop now, or None. Plan section 8: 120% of a model's estimate, or the total stop line."""
    limit = estimate * settings.NB04_MODEL_STOP_FACTOR
    if model_spend > limit:
        return (f"model spend ${model_spend:.4f} is above {settings.NB04_MODEL_STOP_FACTOR:.0%} "
                f"of its estimate (${limit:.4f})")
    total_myr = total_spend_usd() * settings.NB04_USD_TO_MYR
    if total_myr >= settings.NB04_TOTAL_STOP_MYR:
        return f"total spend RM{total_myr:.2f} reached the RM{settings.NB04_TOTAL_STOP_MYR} stop line"
    return None


def utc_now() -> str:
    """The current time in UTC, in the format every saved timestamp uses."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_call(log: list[dict], started_at: str, elapsed: float, outcome: str, error: Exception | None = None, **details) -> dict:
    """Add one call to the log and return its entry."""
    entry = {"started_at": started_at, "finished_at": utc_now(), "elapsed_seconds": round(elapsed, 3), "outcome": outcome,
             "waited_after_seconds": 0.0, **details, **(providers.describe_error(error) if error else {})}
    log.append(entry)
    return entry


def call_with_backoff(candidate: dict, request: dict) -> tuple[providers.GenerationResult, float, dict]:
    """One answer: retry 429s and temporary failures with back-off. Latency is the successful request only;
    all waiting and failed attempts are counted in the returned wait record, 429s apart from temporary failures, and
    waits["call_log"] holds one entry per call. A 429 uses the wait the host asks for, when it gives one, otherwise the
    longer rate-limit back-off."""
    log = []
    waits = {"rate_limit_retries": 0, "rate_limit_wait_seconds": 0.0, "temporary_failure_retries": 0,
             "temporary_failure_wait_seconds": 0.0, "failed_attempt_seconds": 0.0, "call_log": log}
    for attempt in range(settings.NB04_BACKOFF_MAX_RETRIES + 1):
        started_at, start = utc_now(), time.perf_counter()
        try:
            result = call_with_timeout(lambda: providers.send(candidate, request),
                                       settings.NB04_REQUEST_TIMEOUT_SECONDS)
            latency = time.perf_counter() - start
            log_call(log, started_at, latency, "success")
            return result, latency, waits
        except RequestTimedOut as error:
            error.elapsed_seconds = time.perf_counter() - start
            log_call(log, started_at, error.elapsed_seconds, "timeout", error)
            error.waits = waits
            raise
        except providers.RateLimited as error:
            prefix, retry_after, last_error = "rate_limit", error.retry_after, error
            base, cap = settings.NB04_RATE_LIMIT_BACKOFF_BASE_SECONDS, settings.NB04_RATE_LIMIT_BACKOFF_MAX_SECONDS
            entry = log_call(log, started_at, time.perf_counter() - start, prefix, error, retry_after_seconds=retry_after,
                             retry_after_source=error.retry_after_source)
        except providers.TemporaryFailure as error:
            prefix, retry_after, last_error = "temporary_failure", None, error
            base, cap = settings.NB04_BACKOFF_BASE_SECONDS, settings.NB04_BACKOFF_MAX_SECONDS
            entry = log_call(log, started_at, time.perf_counter() - start, prefix, error)
        except Exception as error:      # access blocked, or an error such as a 400: not retried here
            log_call(log, started_at, time.perf_counter() - start, "not_retried", error)
            error.waits = waits
            raise
        waits["failed_attempt_seconds"] = round(waits["failed_attempt_seconds"] + entry["elapsed_seconds"], 2)
        if attempt == settings.NB04_BACKOFF_MAX_RETRIES:
            break
        # Each kind backs off from its own count, so a 429 after several 503s still starts at the 429 base.
        wait = retry_after if retry_after is not None else min(base * 2 ** waits[f"{prefix}_retries"], cap)
        waits[f"{prefix}_retries"] += 1
        waits[f"{prefix}_wait_seconds"] = round(waits[f"{prefix}_wait_seconds"] + wait, 2)
        entry["waited_after_seconds"] = wait
        print(f"  {prefix}: waiting {wait:.0f} s before attempt {attempt + 2} of {settings.NB04_BACKOFF_MAX_RETRIES + 1}", flush=True)
        time.sleep(wait)
    last_error.waits = waits
    raise last_error


def is_stale(record: dict | None, item: dict) -> bool:
    """True when a cached answer was made from an earlier prompt of its item."""
    return record is not None and record.get("prompt_sha256") != item["prompt_sha256"]


def answer_is_current(record: dict | None, item: dict) -> bool:
    """The per-item resume rule: a cached answer is reused only if it has no error and was made from its item's CURRENT prompt.
    A change to another item's prompt (Q100, say) never invalidates this one (Q01)."""
    return record is not None and not record["error"] and not is_stale(record, item)


def waits_and_call_log(error: Exception) -> tuple[dict, list[dict]]:
    """Split what a failed answer carried: the summed wait fields, and its calls."""
    waits = dict(getattr(error, "waits", {}))
    return waits, waits.pop("call_log", [])


def answer_item(candidate: dict, item: dict, cache: dict) -> dict:
    """Generate one answer and return its record. Billed tokens count as spend even if the answer is unusable."""
    request = providers.build_request(candidate, item["messages"])
    result, latency, waits = call_with_backoff(candidate, request)
    cost = (result.input_tokens * candidate["input_usd_per_m"]
            + result.output_tokens * candidate["output_usd_per_m"]) / 1_000_000
    cache["spend"]["list_usd_total"] = round(cache["spend"]["list_usd_total"] + cost, 6)
    cache["spend"]["billed_calls"] += 1
    answer, repeated = apply_repetition_guard(result.text)
    answer_tokens = result.output_tokens - result.reasoning_tokens
    capped = answer_tokens > settings.NB04_MAX_ANSWER_TOKENS and not repeated
    if capped:
        answer = cut_to_answer_cap(answer, answer_tokens)
    empty = not answer.strip() and result.finish != "refusal"
    return {"id": item["id"], "repeat": item["repeat"], "kind": item["kind"], "prompt_sha256": item["prompt_sha256"],
            "answer": answer,
            "stopped_by": stop_label(result.finish, repeated, capped, answer_tokens, result.reasoning_tokens),
            "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
            "answer_tokens": answer_tokens, "reasoning_tokens": result.reasoning_tokens,
            "latency_seconds": round(latency, 3), **waits,
            "cost_usd_list": round(cost, 6), "served_model": result.served_model,
            "error": f"empty answer (finish: {result.finish})" if empty else None}    # errors are retried on rerun


def work_units(items: list[dict]) -> list[dict]:
    """One unit per call: each question once, each probe NB04_PROBE_REPEATS times with a repeat index and its own key."""
    units = []
    for item in items:
        if item["kind"] == "probe":
            units += [{**item, "repeat": n, "key": f"{item['id']}#r{n}"}
                      for n in range(1, settings.NB04_PROBE_REPEATS + 1)]
        else:
            units.append({**item, "repeat": 1, "key": item["id"]})
    return units


def select_items(items: list[dict], kinds: str, only: str | None) -> list[dict]:
    """Which frozen items this run covers."""
    if only:
        wanted = {x.strip() for x in only.split(",")}
        unknown = wanted - {i["id"] for i in items}
        if unknown:
            print(f"ABORTED: unknown item ids {sorted(unknown)}.")
            sys.exit(EXIT_SETUP)
        return [i for i in items if i["id"] in wanted]
    return [i for i in items if kinds == "all" or i["kind"] == kinds]


def main() -> None:
    """Check parity and setup, then generate, saving after every answer and stopping when a rule says so."""
    models = check_parity.all_models()
    parser = argparse.ArgumentParser(description="Generate notebook 04 answers for one model.")
    parser.add_argument("--model", choices=sorted(models))
    parser.add_argument("--report", action="store_true", help="print the truncation and timing reports and exit; calls no model")
    parser.add_argument("--items", default="all", choices=["all", "question", "probe"])
    parser.add_argument("--only", default=None, help="comma-separated ids, e.g. Q01,Q02,Q03 for the integration trial")
    args = parser.parse_args()
    if args.report:
        print(json.dumps({"truncation": truncation_report(), "timing_question_answers": timing_report(),
                          "run_window_all_calls": run_window_report()}, indent=1))
        return
    if not args.model:
        parser.error("--model is required unless --report is given")
    candidate = models[args.model]

    # Stale answers (made from an earlier prompt of their item) are not a failure here: this run replaces them.
    failures = check_parity.run_checks(allow_stale_answers=True)
    if failures:
        print(f"ABORTED: {len(failures)} parity failure(s). No model was called. First ten:")
        print("\n".join(failures[:10]))
        sys.exit(EXIT_SETUP)

    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    if not providers.api_key(candidate):
        print(f"ABORTED: {settings.API_KEY_ENV_VARS[candidate['key']]} is not set in .env. No model was called.")
        sys.exit(EXIT_SETUP)

    items = json.loads(settings.NB04_CONTEXTS_FILE.read_text(encoding="utf-8"))["items"]
    settings_now = run_settings(args.model, candidate, prompt_set_hash(items))
    path = settings.NB04_RAW_DIR / f"{args.model}.json"
    cache = load_cache(path, settings_now)
    estimate = estimated_spend_usd(candidate)

    selected = work_units(select_items(items, args.items, args.only))
    todo = [i for i in selected if not answer_is_current(cache["results"].get(i["key"]), i)]
    stale = [i["key"] for i in todo if is_stale(cache["results"].get(i["key"]), i)]
    print(f"{args.model}: {len(selected)} selected, {len(selected) - len(todo)} already done, {len(todo)} to run"
          f"{f' ({len(stale)} made from an earlier prompt: {stale[:5]})' if stale else ''}. "
          f"Estimate ${estimate:.2f}, spent ${cache['spend']['list_usd_total']:.4f}.")

    for n, item in enumerate(todo, 1):
        reason = hard_stop_reason(cache["spend"]["list_usd_total"], estimate)
        if reason:
            save_cache(path, cache)
            print(f"HARD STOP before {item['id']}: {reason}. State saved. Waiting for the author's decision.")
            sys.exit(EXIT_HARD_STOP)
        if is_stale(cache["results"].get(item["key"]), item):     # kept for the record, never reused or carried over
            cache.setdefault("superseded", []).append(
                {**cache["results"].pop(item["key"]), "superseded_at": utc_now(), "reason": "its item's prompt changed"})
        timed_out = None
        try:
            record = answer_item(candidate, item, cache)
        except RequestTimedOut as error:     # abandoned; retried on the next run, and never counted as latency
            waits, log = waits_and_call_log(error)
            timed_out = {"at": utc_now(), "started_at": log[-1]["started_at"],
                         "elapsed_seconds": round(error.elapsed_seconds, 2), "waits": waits}
            record = {"id": item["id"], "repeat": item["repeat"], "kind": item["kind"],
                      "prompt_sha256": item["prompt_sha256"], "answer": None, "stopped_by": "timeout",
                      "latency_seconds": None, "elapsed_seconds": timed_out["elapsed_seconds"],
                      **waits, "call_log": log, "error": str(error)}
        except (providers.RateLimited, providers.TemporaryFailure) as error:
            waits, log = waits_and_call_log(error)
            cache.setdefault("retries_exhausted", []).append(
                {"key": item["key"], "kind": item["kind"], "at": utc_now(),
                 "error_type": type(error).__name__, "waits": waits, "call_log": log})
            save_cache(path, cache)
            print(f"STOPPED at {item['id']}: not cleared after {settings.NB04_BACKOFF_MAX_RETRIES} retries ({error}). "
                  f"State saved; rerun later to resume.")
            sys.exit(EXIT_BLOCKED)
        except providers.AccessBlocked as error:
            cache.setdefault("access_blocked", []).append(
                {"key": item["key"], "kind": item["kind"], "at": utc_now(), "call_log": waits_and_call_log(error)[1]})
            save_cache(path, cache)
            print(f"STOPPED at {item['id']}: access blocked ({error}). Log it in ease_log.json as an access blocker.")
            sys.exit(EXIT_BLOCKED)
        except Exception as error:     # e.g. a 400 for a rejected setting: recorded, retried on the next run
            waits, log = waits_and_call_log(error)
            record = {"id": item["id"], "repeat": item["repeat"], "kind": item["kind"],
                      "prompt_sha256": item["prompt_sha256"], "answer": None, "stopped_by": None,
                      **waits, "call_log": log, "error": f"{type(error).__name__}: {error}"}
        # Carried over from earlier runs, so that a stall or failure stays on the record after a later attempt succeeds.
        previous = cache["results"].get(item["key"], {})
        record["timed_out_attempts"] = previous.get("timed_out_attempts", []) + ([timed_out] if timed_out else [])
        record["call_log"] = previous.get("call_log", []) + record.get("call_log", [])
        cache["results"][item["key"]] = record
        save_cache(path, cache)
        print(f"[{n}/{len(todo)}] {item['key']} {record.get('latency_seconds')}s stopped_by={record['stopped_by']} "
              f"cost=${record.get('cost_usd_list')} error={record['error']}", flush=True)

    reason = hard_stop_reason(cache["spend"]["list_usd_total"], estimate)
    if reason:
        print(f"HARD STOP after the last answer: {reason}.")
        sys.exit(EXIT_HARD_STOP)
    print(f"Done. {args.model} spend at list price: ${cache['spend']['list_usd_total']:.4f}.")


if __name__ == "__main__":
    main()
