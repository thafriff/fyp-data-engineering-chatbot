"""
Planned diagnostic from docs/notebook04_plan.md section 2.10: "what is olpa", sent with the typo exactly as the dev model
received it, to the two FREE candidates. NOT a Table 3.9 criterion: it enters no score and no weight.

Section 11.4 recorded that llama3.2:3b answered "I couldn't find the term 'OLPA' in the course materials..." although
retrieval was fully correct. If a candidate answers about OLAP from the same prompt, that separates model from pipeline for
this input. The author reads each answer: answered about OLAP; noted the typo and answered; or refused.

How it stays apart from the comparison:
- the prompt is built here from the five chunks fixed by ID and order in section 2.10, and must match the registered SHA-256;
  it is not one of the 110 frozen items, and contexts.json, the manifest and settings.py are not touched;
- answers go to olpa_diagnostic.json, not raw/ or judge/, so they are outside the running spend total (free tier);
- each call goes through run_generation.answer_item, so the request, timeout, back-off, 800-token cap and repetition guard
  are exactly those of the real run.
The constants live here, not in settings.py, for the same reason as in prompt_wording_check.py: this side experiment must not
change that file. The local llama3.2:3b arm registered in section 2.10 (5 + 5 runs) is not built yet.

It saves after every answer, resumes after an interruption, retries errors on the next run, and refuses to resume under
different settings.

Usage (from the project root; needs GOOGLE_API_KEY in .env):
    venv\\Scripts\\python evaluation\\notebook04\\olpa_diagnostic.py            run or resume, then print the answers
    venv\\Scripts\\python evaluation\\notebook04\\olpa_diagnostic.py --report   print the answers; calls no model
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for extra_path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

import providers  # noqa: E402
import run_generation  # noqa: E402
from config import settings  # noqa: E402
from freeze_contexts import prompt_sha256  # noqa: E402
from src.core import llm  # noqa: E402
from src.core.retriever import RetrievedChunk  # noqa: E402
from src.core.timeouts import RequestTimedOut  # noqa: E402

# Registered in plan section 2.10 on 2026-11-04 and 2026-11-05; do not change.
QUESTION = "what is olpa"
CHUNKS = [("S05-0138", 0.7084), ("S04-0210", 0.692), ("S06-0204", 0.6831), ("S05-0128", 0.6819), ("S06-0205", 0.6701)]
REGISTERED_SHA256 = "2d280feb76113d35b625c2ef8e1f0bf9248f6eb12c4dd5ca2fc69bcd874297db"
CANDIDATES = ["gemini-3.8-flash", "gemma-4-31b-it"]
REPEATS = 3
OUTPUT_FILE = settings.NB04_DIR / "olpa_diagnostic.json"
RESUME_MUST_MATCH = ["question", "chunk_ids", "prompt_sha256", "candidates", "repeats", "candidate_settings"]


def build_messages() -> list[dict]:
    """The registered prompt: the typo as typed, the five fixed chunks, the applied SYSTEM_PROMPT. Exits if the hash differs."""
    chunks = {c["chunk_id"]: c for c in json.loads(settings.CHUNKS_FILE.read_text(encoding="utf-8"))}
    fields = RetrievedChunk.__dataclass_fields__
    retrieved = [RetrievedChunk(**{k: (score if k == "score" else chunks[cid].get(k)) for k in fields}) for cid, score in CHUNKS]
    messages = llm.build_messages(QUESTION, retrieved)
    if prompt_sha256(messages) != REGISTERED_SHA256:
        sys.exit(f"ABORTED: the prompt hash {prompt_sha256(messages)} is not the registered {REGISTERED_SHA256}. No model was called.")
    return messages


def run_settings() -> dict:
    """What must match to resume, plus the date and commit."""
    return {"question": QUESTION, "chunk_ids": [c for c, _ in CHUNKS], "prompt_sha256": REGISTERED_SHA256,
            "candidates": CANDIDATES, "repeats": REPEATS,
            "candidate_settings": {m: {k: settings.NB04_CANDIDATES[m][k] for k in
                                       ("model_id", "temperature", "request_options", "reasoning_allowance_tokens")}
                                   for m in CANDIDATES},
            "max_answer_tokens": settings.NB04_MAX_ANSWER_TOKENS, "request_timeout_seconds": settings.NB04_REQUEST_TIMEOUT_SECONDS,
            "date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git": run_generation.git_state()}


def load_cache(now: dict) -> dict:
    """The saved results, or an empty file. Exits if they were made under different settings."""
    if not OUTPUT_FILE.exists():
        return {"run_settings": now, "spend": {"list_usd_total": 0.0, "billed_calls": 0}, "results": {}}
    cache = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
    mismatched = [k for k in RESUME_MUST_MATCH if cache["run_settings"].get(k) != now[k]]
    if mismatched:
        sys.exit(f"ABORTED: {OUTPUT_FILE.name} was made with different settings ({', '.join(mismatched)}). No model was called.")
    return cache


def print_answers(cache: dict) -> None:
    """Every saved answer, for the author to read."""
    for key, record in sorted(cache["results"].items()):
        print(f"\n--- {key}: {record.get('latency_seconds')} s, in {record.get('input_tokens')}, out {record.get('output_tokens')}, "
              f"reasoning {record.get('reasoning_tokens')}, stopped_by {record.get('stopped_by')}, error {record.get('error')}")
        print(record.get("answer"))


def main() -> None:
    """Send each repeat once, alternating the models, saving after every answer."""
    parser = argparse.ArgumentParser(description="The 'what is olpa' diagnostic (plan section 2.10).")
    parser.add_argument("--report", action="store_true", help="print the saved answers; calls no model")
    args = parser.parse_args()
    if args.report:
        print_answers(json.loads(OUTPUT_FILE.read_text(encoding="utf-8")))
        return

    messages = build_messages()
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    for model in CANDIDATES:
        if not providers.api_key(settings.NB04_CANDIDATES[model]):
            sys.exit(f"ABORTED: the key for {model} is not set in .env. No model was called.")

    cache = load_cache(run_settings())
    units = [(model, repeat) for repeat in range(1, REPEATS + 1) for model in CANDIDATES]
    todo = [u for u in units if f"{u[0]}#r{u[1]}" not in cache["results"] or cache["results"][f"{u[0]}#r{u[1]}"]["error"]]
    print(f"{len(units)} answers planned, {len(units) - len(todo)} already saved, {len(todo)} to run.")
    for model, repeat in todo:
        key = f"{model}#r{repeat}"
        item = {"id": "OLPA", "repeat": repeat, "kind": "diagnostic", "prompt_sha256": REGISTERED_SHA256, "messages": messages}
        try:
            record = run_generation.answer_item(settings.NB04_CANDIDATES[model], item, cache)
        except RequestTimedOut as error:
            record = {"id": "OLPA", "repeat": repeat, "answer": None, "stopped_by": "timeout", "latency_seconds": None,
                      "elapsed_seconds": round(error.elapsed_seconds, 2), "error": str(error)}
        except (providers.RateLimited, providers.TemporaryFailure, providers.AccessBlocked) as error:
            run_generation.save_cache(OUTPUT_FILE, cache)
            sys.exit(f"STOPPED at {key}: {type(error).__name__} ({error}). Saved; rerun later to resume.")
        except Exception as error:
            record = {"id": "OLPA", "repeat": repeat, "answer": None, "stopped_by": None, "error": f"{type(error).__name__}: {error}"}
        record["author_reading"] = None       # the author fills: answered / noted_typo_and_answered / refused
        cache["results"][key] = record
        run_generation.save_cache(OUTPUT_FILE, cache)
        print(f"{key}: {record.get('latency_seconds')} s, stopped_by={record['stopped_by']}, error={record['error']}", flush=True)
    print_answers(cache)


if __name__ == "__main__":
    main()
