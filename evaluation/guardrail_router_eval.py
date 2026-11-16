"""
Detection evaluation of the guardrail router (src/core/guardrail.py), no model called.

Runs the router's rule over three sets and prints caught/missed per set:
- G01-G10 from evaluation/notebook04/guardrail_probes.json: IN-SAMPLE (they were used to design the rules);
- the held-out recall and near-miss sets in evaluation/guardrail_router_cases.json, whose SHA-256 is checked against the
  value recorded before the rule code was written (evaluation/guardrail_router_cases.sha256);
- all 100 frozen evaluation questions, which must all pass through.
The message goes through the pipeline's real cleaning step (greeting removal, then spelling correction) first, as in the app.

Usage (from the project root):  venv\\Scripts\\python evaluation\\guardrail_router_eval.py [--json OUT]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings  # noqa: E402
from src.core import guardrail  # noqa: E402

CASES = PROJECT_ROOT / "evaluation" / "guardrail_router_cases.json"
CASES_SHA = PROJECT_ROOT / "evaluation" / "guardrail_router_cases.sha256"


def check_cases_unchanged() -> str:
    """The held-out set must be the one hashed before the rules existed."""
    recorded = CASES_SHA.read_text(encoding="utf-8").split()[0]
    actual = hashlib.sha256(CASES.read_bytes()).hexdigest()
    if actual != recorded:
        sys.exit(f"ABORTED: {CASES.name} has changed since it was hashed ({actual[:12]} != {recorded[:12]}).")
    return actual


def run_set(name: str, items: list[dict], clean, in_sample: bool) -> dict:
    """Route each item and compare with its expected decision."""
    rows = []
    for item in items:
        decision = guardrail.detect_task_request(clean(item["text"]))
        got = "task" if decision.is_task else "question"
        rows.append({"id": item["id"], "text": item["text"], "expect": item["expect"], "got": got,
                     "pass": got == item["expect"], "verb": decision.verb, "deliverable": decision.deliverable,
                     "assessed": decision.assessed, "complete": decision.complete, "example": decision.example,
                     "reasons": decision.reasons})
    return {"set": name, "in_sample": in_sample, "n": len(rows), "passed": sum(r["pass"] for r in rows),
            "failed": [r["id"] for r in rows if not r["pass"]], "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Guardrail router detection evaluation (no model called).")
    parser.add_argument("--json", type=Path, default=None, help="also write the full rows to this file")
    args = parser.parse_args()

    sha = check_cases_unchanged()
    from src.core.rag import RagPipeline
    pipeline = RagPipeline()
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    probes = json.loads(settings.NB04_PROBES_FILE.read_text(encoding="utf-8"))["probes"]
    questions = json.loads(settings.TEST_QUESTIONS_FILE.read_text(encoding="utf-8"))["questions"]

    results = [
        run_set("G01-G10 (in-sample)", [{"id": p["id"], "text": p["request"], "expect": "task"} for p in probes],
                pipeline.search_text, True),
        run_set("recall (held out)", cases["recall"], pipeline.search_text, False),
        run_set("near-miss (held out)", cases["near_miss"], pipeline.search_text, False),
        run_set("100 test questions", [{"id": q["id"], "text": q["question"], "expect": "question"} for q in questions],
                pipeline.search_text, False),
    ]
    print(f"held-out set SHA-256 {sha[:12]} unchanged since it was written")
    for r in results:
        print(f"{r['set']:24} {r['passed']:3} of {r['n']:3} pass{'  (in-sample)' if r['in_sample'] else ''}"
              f"{'  failed: ' + ', '.join(r['failed']) if r['failed'] else ''}")
    for r in results:
        for row in r["rows"]:
            if not row["pass"]:
                why = ", ".join(row["reasons"]) if row["reasons"] else (
                    f"verb={row['verb']!r} deliverable={row['deliverable']!r} assessed={row['assessed']!r} complete={row['complete']!r}")
                print(f"  FAIL {row['id']}: expected {row['expect']}, got {row['got']} [{why}] {row['text'][:80]!r}")
    if args.json:
        args.json.write_text(json.dumps({"cases_sha256": sha, "results": results}, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"rows written to {args.json}")


if __name__ == "__main__":
    main()
