"""
Reruns run_generation.py for one model until its 130 answers are all saved without errors (author's STEP 2, 2026-11-27: "keep retrying
503s with backoff"). run_generation.py already backs off and retries each call (NB04_BACKOFF_MAX_RETRIES); when an answer's retries are used
up it saves state and exits with code 3. This loop waits and starts it again. Saved answers are never re-sent or re-paid (the per-item
resume rule), so a rerun only sends what is missing or failed.

Stops, without retrying, on:
- exit 2: a hard stop (120% of the model's estimate, or the RM96 total): the author decides;
- exit 3 with "access blocked": a 401/403/404 cannot clear by waiting;
- exit 1 (setup or parity failure), except Windows' transient torch DLL error (WinError 1114) at import, which is retried;
- three complete passes in a row that add no answer (e.g. a request setting the host keeps rejecting). An overloaded host (exit 3 after
  its retries) is not a stall: it is waited out, up to --max-runs.
While evaluation/notebook04/raw/PAUSE exists, no new run is started (used while files in the working tree are swapped for a commit).

Usage (from the project root):  venv\\Scripts\\python evaluation\\notebook04\\rerun_until_done.py --model gemini-3.8-flash
Log: evaluation/notebook04/raw/logs/<model>.log (Git ignored; the runner prints keys, timings, costs and errors, never answer text).
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from config import settings  # noqa: E402

RUNNER = PROJECT_ROOT / "evaluation" / "notebook04" / "run_generation.py"
PAUSE_FILE = settings.NB04_RAW_DIR / "PAUSE"


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def answers_done(model: str) -> tuple[int, int]:
    """(answers saved without an error, answers saved with one) in the model's cache."""
    path = settings.NB04_RAW_DIR / f"{model}.json"
    if not path.exists():
        return 0, 0
    results = json.loads(path.read_text(encoding="utf-8"))["results"].values()
    return sum(not r["error"] for r in results), sum(bool(r["error"]) for r in results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun run_generation.py for one model until it is complete.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--wait", type=float, default=600, help="seconds between reruns (default 10 minutes)")
    parser.add_argument("--max-runs", type=int, default=48)
    args = parser.parse_args()
    log_path = settings.NB04_RAW_DIR / "logs" / f"{args.model}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stalls = 0
    with log_path.open("a", encoding="utf-8") as log:
        def note(text: str) -> None:
            log.write(f"[{stamp()}] {text}\n")
            log.flush()
        for run in range(1, args.max_runs + 1):
            while PAUSE_FILE.exists():
                time.sleep(15)
            before, _ = answers_done(args.model)
            note(f"run {run}: starting ({before} answers saved)")
            proc = subprocess.run([sys.executable, str(RUNNER), "--model", args.model], cwd=PROJECT_ROOT,
                                  capture_output=True, text=True, encoding="utf-8", errors="replace")
            log.write(proc.stdout + proc.stderr)
            after, errors = answers_done(args.model)
            note(f"run {run}: exit {proc.returncode}; {after} answers saved, {errors} with an error")
            output = proc.stdout + proc.stderr
            if proc.returncode == 0 and after >= settings.NB04_ANSWERS_PER_MODEL and errors == 0:
                note("DONE: all answers saved without errors")
                return
            if proc.returncode == 2:
                note("STOP: hard stop; the author decides")
                sys.exit(2)
            if proc.returncode == 3 and "access blocked" in output:
                note("STOP: access blocked")
                sys.exit(3)
            if proc.returncode == 1 and "WinError 1114" not in output:
                note("STOP: setup or parity failure")
                sys.exit(1)
            # Only a pass that ran to the end yet left errors counts as a stall: an exit 3 (an overloaded or rate-limited host whose
            # retries ran out) is waited out for as long as --max-runs allows.
            stalls = stalls + 1 if (proc.returncode == 0 and after <= before) else 0
            if stalls >= 3:
                note("STOP: three reruns in a row added no answer")
                sys.exit(4)
            wait = 60 if proc.returncode == 1 else args.wait
            note(f"waiting {wait:.0f} s before the next run")
            time.sleep(wait)
        note("STOP: max runs reached")
        sys.exit(5)


if __name__ == "__main__":
    main()
