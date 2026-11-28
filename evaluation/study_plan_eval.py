"""
Evaluation of the study plan (feature A; docs/feature_design_2026-11-28.md) on the hashed design set written before the code
(evaluation/study_plan_cases.json; its SHA-256 must match, or nothing runs). The author's 6 held-out cases, when written, go in
evaluation/study_plan_cases_holdout.json (hashed the same way) and are run and reported apart.

Through the app's own pipeline (RagPipeline.answer, exactly what the chat does; the intent classifier is replaced by REAL_QUESTION,
because a plan is decided before it):
- ROUTING and fixed replies for every case;
- for each in-scope and Malay case, --runs plans with --model: the deterministic checks (budget within 10%, real section titles,
  citations in the retrieved set and in their own section, difficulty never decreasing, a checkpoint at the end) and the grounding of
  the suggested questions before replacement (target >= 90%), with cost and time.
Raw results go to evaluation/features/runs/ (Git ignored); --summary writes evaluation/features/study_plan_eval_summary.json (numbers
only) and the author's review sheet of 5 plans (Git ignored).

Usage (from the project root):
    venv\\Scripts\\python evaluation\\study_plan_eval.py --model deepseek-flash --runs 3
    venv\\Scripts\\python evaluation\\study_plan_eval.py --model qwen3.5:2b --runs 1
    venv\\Scripts\\python evaluation\\study_plan_eval.py --summary
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.stdout.reconfigure(encoding="utf-8")

from config import settings  # noqa: E402
from src.core import intent, study_plan  # noqa: E402
from src.core.rag import RagPipeline  # noqa: E402

EVAL = ROOT / "evaluation"
RUNS = EVAL / "features" / "runs"
SUMMARY = EVAL / "features" / "study_plan_eval_summary.json"
SHEET = RUNS / "study_plan_review_sheet.md"
FILES = [name for name in ("study_plan_cases", "study_plan_cases_holdout") if (EVAL / f"{name}.json").exists()]
EXPECT = {"in_scope": "plan", "malay": "plan", "off_topic": "plan_declined", "too_broad": "plan_broad", "too_short": "plan_too_short",
          "missing": None, "not_plan": "not_plan"}


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_sets() -> dict:
    sets = {}
    for name in FILES:
        data = (EVAL / f"{name}.json").read_bytes()
        if hashlib.sha256(data).hexdigest() != (EVAL / f"{name}.sha256").read_text(encoding="utf-8").split()[0]:
            sys.exit(f"ABORTED: {name}.json does not match its recorded SHA-256.")
        sets[name] = json.loads(data)
    return sets


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def outcome(pipeline: RagPipeline, text: str, model: str):
    with mock.patch.object(intent, "classify", lambda message: "REAL_QUESTION"):
        return pipeline.answer(text, model=model)


def checks(plan: dict, retrieved: set[str], chunks: dict, budget: int) -> dict:
    """The deterministic checks of the hashed set's scoring."""
    titles = {study_plan.section_title(c["section"])[0] for c in chunks.values()}
    items = [i for s in plan["sessions"] for i in s["items"] if not i.get("review")]
    levels = [max(study_plan._DIFFICULTY[i["difficulty"]] for i in s["items"] if not i.get("review"))
              for s in plan["sessions"] if any(not i.get("review") for i in s["items"])]
    return {"budget_ok": abs(study_plan.total_minutes(plan) - budget) <= 0.1 * budget,
            "titles_real": all(i["title"] in titles for i in items),
            "citations_ok": all(c in retrieved and study_plan.section_title(chunks[c]["section"])[0].lower().split()
                                == i["title"].lower().split() for i in items for c in i["chunk_ids"]),
            "difficulty_ok": levels == sorted(levels), "checkpoint_at_end": plan["sessions"][-1]["checkpoint"]}


def run(model: str, runs: int, tag: str = "") -> None:
    sets = load_sets()
    pipeline = RagPipeline()
    path = RUNS / f"study_plan_eval_{model.replace(':', '_')}{'_' + tag if tag else ''}.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "meta": {"model": model, "tag": tag or "round1", "started": stamp(), "finished": False,
                 "case_sha256": {n: (EVAL / f"{n}.sha256").read_text().split()[0] for n in FILES}}, "routing": [], "plans": []}
    if not data["routing"]:
        for name, cases in sets.items():
            for group, items in cases.items():
                if group == "meta":
                    continue
                for case in items:
                    result = outcome(pipeline, case["text"], model)
                    got = ("plan" if result.plan_call else result.intent if result.intent.startswith("plan_") else "not_plan")
                    want = EXPECT[group] or f"plan_ask_{case.get('missing')}"
                    data["routing"].append({"file": name, "set": group, "id": case["id"], "expect": want, "got": got,
                                            "passed": got == want})
        save(path, data)
    done = {(p["id"], p["run"]) for p in data["plans"]}
    for number in range(1, runs + 1):
        for name, cases in sets.items():
            for case in cases.get("in_scope", []) + cases.get("malay", []):
                if (case["id"], number) in done:
                    continue
                result = outcome(pipeline, case["text"], model)
                record = {"id": case["id"], "file": name, "run": number, "at": stamp(), "text": case["text"]}
                try:
                    plan = result.plan_call()
                except Exception as error:
                    record["error"] = f"{type(error).__name__}: {error}"
                else:
                    record.update(plan=plan, checks=checks(plan, {c.chunk_id for c in result.chunks} | set(plan["chunk_ids"]),
                                                           pipeline.retriever.chunks, case["budget_minutes"]),
                                  sessions=len(plan["sessions"]), subtopics_kept=sum(1 for s in plan["sessions"] for i in s["items"]
                                                                                     if not i.get("review") and not i.get("continued")))
                data["plans"].append(record)
                save(path, data)
                stats = record.get("plan", {}).get("question_stats", {})
                print(f"[run {number}] {case['id']} sessions {record.get('sessions')} checks "
                      f"{all(record.get('checks', {}).values()) if record.get('checks') else None} questions "
                      f"{stats.get('grounded_before_replacement')}/{stats.get('generated')} error {record.get('error')}", flush=True)
    data["meta"].update(finished=True, ended=stamp())
    save(path, data)
    print(f"saved {path}")


def file_summary(data: dict) -> dict:
    plans = [p for p in data["plans"] if p.get("plan")]
    stats = [p["plan"]["question_stats"] for p in plans]
    generated = sum(s["generated"] for s in stats)
    grounded = sum(s["grounded_before_replacement"] for s in stats)
    return {"model": data["meta"]["model"], "round": data["meta"]["tag"],
            "routing": {"cases": len(data["routing"]), "as_expected": sum(r["passed"] for r in data["routing"]),
                        "misses": [f"{r['id']}: expected {r['expect']}, got {r['got']}" for r in data["routing"] if not r["passed"]]},
            "plans": len(data["plans"]), "plan_errors": sum(1 for p in data["plans"] if p.get("error")),
            "deterministic_all_pass": sum(all(p["checks"].values()) for p in plans),
            "deterministic_failures": {k: sum(not p["checks"][k] for p in plans) for k in (plans[0]["checks"] if plans else {})},
            "questions_generated": generated, "questions_grounded_before_replacement": grounded,
            "grounding_rate": round(grounded / generated, 3) if generated else None,
            "questions_replaced": sum(s["replaced"] for s in stats), "questions_dropped": sum(s["dropped"] for s in stats),
            "sessions_with_template_questions_only": sum(s["template_only_sessions"] for s in stats),
            "write_errors": sum(len(p["plan"]["write_errors"]) for p in plans),
            "seconds_median": statistics.median([p["plan"]["write_seconds"] for p in plans]) if plans else None,
            "cost_usd_list_total": round(sum((p["plan"].get("usage") or {}).get("cost_usd_list", 0) for p in plans), 6)}


def write_summary() -> None:
    files = sorted(RUNS.glob("study_plan_eval_*.json"))
    summary = {"meta": {"feature": "A Study plan", "generated": stamp(), "contents": "aggregate numbers only (no plan text)"},
               "runs": {}}
    sheet = None
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        summary["runs"][path.stem.replace("study_plan_eval_", "")] = file_summary(data)
        if data["meta"]["model"] == settings.DEV_DEFAULT_ANSWER_MODEL:
            sheet = data
    save(SUMMARY, summary)
    if sheet:
        _write_sheet(sheet)
    print(json.dumps(summary, indent=1))


def _write_sheet(data: dict) -> None:
    chunks = {c["chunk_id"]: c for c in json.loads(settings.CHUNKS_FILE.read_text(encoding="utf-8"))}
    picks = [p for p in data["plans"] if p["run"] == 1 and p.get("plan")][:5]
    lines = ["# Study plan review sheet (feature A)", "",
             f"Generated {stamp()}. Five plans from the {settings.DEV_DEFAULT_ANSWER_MODEL} evaluation, run 1. Read each plan AGAINST ITS "
             "SOURCE SECTIONS (verification against the material, not expert judgement): are the sub-topics the right sections for the "
             "request, in a sensible order, and does every suggested question fit its session's material? Write a verdict after **My "
             "verdict**. Git ignored: it quotes licensed course text.", ""]
    for n, record in enumerate(picks, 1):
        plan = record["plan"]
        lines += ["---", "", f"## {n}. {record['id']}: \"{record['text']}\"", "", study_plan.to_markdown(plan), "", "**Source sections:**", ""]
        for cid in list(dict.fromkeys(c for s in plan["sessions"] for i in s["items"] for c in i["chunk_ids"]))[:12]:
            text = chunks[cid]["text"].replace("\n", " ")
            lines.append(f"- `{cid}` {chunks[cid]['source_name']} · {chunks[cid]['section']}: {text[:240]}...")
        lines += ["", "**My verdict:** ", ""]
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    SHEET.write_text("\n".join(lines), encoding="utf-8")
    print(f"review sheet: {SHEET}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the study plan on the hashed test set.")
    parser.add_argument("--model")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--tag", default="")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.summary:
        write_summary()
        return
    if not args.model:
        parser.error("--model is required unless --summary is given")
    run(args.model, args.runs, args.tag)


if __name__ == "__main__":
    main()
