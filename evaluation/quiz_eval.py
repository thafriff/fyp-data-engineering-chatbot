"""
Evaluation of "Quiz me" (feature B; docs/feature_design_2026-11-28.md) on the hashed test sets written before the code
(evaluation/quiz_cases.json, quiz_cases_holdout.json; their SHA-256 must match, or nothing runs).

What it does, through the app's own pipeline (RagPipeline.answer, exactly what the chat does):
- ROUTING for every case (deterministic). The intent classifier is replaced by REAL_QUESTION, because a quiz is decided before the
  classifier and a not-quiz case only has to avoid the quiz route; this keeps the cloud run independent of Ollama.
- QUIZZES for the in-scope cases (design and held-out), the near-domain case and the multi-turn follow-up, --runs times with --model,
  recording every round's check results, the items shown, cost and time. Off-topic cases must get the course fallback (no model call).
- The follow-up QF01 gets its first turn as a recorded exchange (a fixed answer, no model call), then "quiz me on that".
Raw results (quiz text quotes course material) go to evaluation/features/runs/ (Git ignored), saved after every quiz so a stopped run
keeps its data; --summary writes evaluation/features/quiz_eval_summary.json (numbers only, committable) and the author's review sheet
of 15 items (Git ignored).

Usage (from the project root):
    venv\\Scripts\\python evaluation\\quiz_eval.py --model deepseek-flash --runs 3      # about RM0.4 at list price
    venv\\Scripts\\python evaluation\\quiz_eval.py --model qwen3.5:2b --runs 1         # local; Ollama must be running
    venv\\Scripts\\python evaluation\\quiz_eval.py --summary
"""

import argparse
import collections
import hashlib
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.stdout.reconfigure(encoding="utf-8")

from config import settings  # noqa: E402
from src.core import intent, llm, quiz  # noqa: E402
from src.core.conversation import Conversation  # noqa: E402
from src.core.rag import RagPipeline  # noqa: E402

EVAL = ROOT / "evaluation"
RUNS = EVAL / "features" / "runs"
SUMMARY = EVAL / "features" / "quiz_eval_summary.json"
SHEET = RUNS / "quiz_review_sheet.md"
FILES = ("quiz_cases", "quiz_cases_holdout")


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_sets() -> dict:
    """Both case files, refusing to run if either changed since it was hashed."""
    sets = {}
    for name in FILES:
        data = (EVAL / f"{name}.json").read_bytes()
        recorded = (EVAL / f"{name}.sha256").read_text(encoding="utf-8").split()[0]
        if hashlib.sha256(data).hexdigest() != recorded:
            sys.exit(f"ABORTED: {name}.json does not match its recorded SHA-256; the test set must not change.")
        sets[name] = json.loads(data)
    return sets


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def routing_report(pipeline: RagPipeline, sets: dict) -> list[dict]:
    """Every case's route, against its set's expectation."""
    expect = {"in_scope": "quiz", "off_topic": "quiz", "near_domain": "quiz", "task": "coach", "not_quiz": "not_quiz"}
    rows = []
    with mock.patch.object(intent, "classify", lambda message: "REAL_QUESTION"):
        for name, cases in sets.items():
            for group, items in cases.items():
                if group not in expect:
                    continue
                for case in items:
                    decision = intent.route(case["text"], top_score=lambda m: 0.0, clean=pipeline.search_text)
                    got = ("quiz" if decision.kind == "quiz" else "coach" if decision.path == "task_request" else "not_quiz")
                    want = expect[group]
                    rows.append({"file": name, "set": group, "id": case["id"], "expect": want, "got": got,
                                 "topic": decision.quiz_topic, "passed": got == want if group != "near_domain" else None})
    return rows


def quiz_cases(sets: dict) -> list[dict]:
    """The cases that get a quiz attempt: in scope (design, held out), near domain, off topic and the follow-up."""
    out = []
    for name, cases in sets.items():
        for group in ("in_scope", "near_domain", "off_topic"):
            for case in cases.get(group, []):
                out.append({**case, "file": name, "set": group})
    for case in sets["quiz_cases"]["follow_up"]:
        if case["expect"] == "quiz":
            out.append({"id": case["id"], "turns": case["turns"], "file": "quiz_cases", "set": "follow_up",
                        "topic_terms": case["topic_terms"]})
    return out


def run_case(pipeline: RagPipeline, case: dict, model: str) -> dict:
    """One case through the real pipeline: route, retrieve, threshold, then write and check the quiz (when one is due)."""
    convo = None
    text = case.get("text")
    with mock.patch.object(intent, "classify", lambda message: "REAL_QUESTION"):
        if case["set"] == "follow_up":
            convo = Conversation()
            with mock.patch.object(llm, "cloud_generate", lambda *a, **k: SimpleNamespace(text="(recorded first answer)")):
                first = pipeline.answer(case["turns"][0], model="deepseek-flash", conversation=convo)
                convo.record(first, first.cloud_call().text)
            text = case["turns"][1]
        result = pipeline.answer(text, model=model, conversation=convo)
    record = {"id": case["id"], "file": case["file"], "set": case["set"], "text": text, "at": stamp(), "intent": result.intent,
              "topic": result.quiz_topic, "top1": result.chunks[0].score if result.chunks else None,
              "fallback": result.used_fallback, "retrieved": [c.chunk_id for c in result.chunks], "error": None}
    if case["set"] == "follow_up":
        record["topic_has_terms"] = all(t in (result.quiz_topic or "") for t in case["topic_terms"])
    if result.quiz_call is None:
        return record
    start = time.perf_counter()
    try:
        made = result.quiz_call()
    except (llm.CloudError, quiz.QuizError) as error:
        record.update(error=f"{type(error).__name__}: {error}", rounds=getattr(error, "rounds", []),
                      seconds=round(time.perf_counter() - start, 1))
        return record
    record.update(items=made["items"], rounds=made["rounds"], usage=made["usage"], seconds=made["seconds"],
                  outside=[i for item in made["items"] for i in item["chunk_ids"] if i not in made["chunk_ids"]])
    return record


def run(model: str, runs: int, tag: str = "") -> None:
    sets = load_sets()
    pipeline = RagPipeline()
    path = RUNS / f"quiz_eval_{model.replace(':', '_')}{'_' + tag if tag else ''}.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "meta": {"model": model, "tag": tag or "round1", "started": stamp(), "case_sha256": {n: (EVAL / f"{n}.sha256").read_text().split()[0] for n in FILES},
                 "finished": False}, "routing": routing_report(pipeline, sets), "quizzes": []}
    done = {(q["id"], q["run"]) for q in data["quizzes"]}
    for number in range(1, runs + 1):
        for case in quiz_cases(sets):
            if (case["id"], number) in done:
                continue
            record = run_case(pipeline, case, model)
            record["run"] = number
            data["quizzes"].append(record)
            save(path, data)
            shown = len(record.get("items", []))
            print(f"[run {number}] {case['id']:5s} {record['intent']:9s} top1 {record['top1'] or 0:.3f} fallback {record['fallback']!s:5s} "
                  f"items {shown} error {record['error']}", flush=True)
    data["meta"].update(finished=True, ended=stamp())
    save(path, data)
    print(f"saved {path}")


def model_summary(data: dict) -> dict:
    """Aggregate numbers for one model's run file."""
    rows = data["quizzes"]
    out = {"runs": max((q["run"] for q in rows), default=0), "groups": {}}
    for group in ("design", "held_out"):
        wanted = [q for q in rows if q["set"] in ("in_scope", "follow_up")
                  and (q["file"] == "quiz_cases") == (group == "design")]
        made = [q for q in wanted if q.get("items")]
        first = [i for q in wanted for r in q.get("rounds", []) if r["attempt"] == 1 for i in r["items"]]
        reasons = collections.Counter(i["reason"].split(":")[0] for q in wanted for r in q.get("rounds", []) for i in r["items"]
                                      if not i["passed"])
        out["groups"][group] = {
            "quiz_attempts": len(wanted),
            "shown_with_2_or_more_items": len(made),
            "shown_with_3_items": sum(len(q["items"]) == 3 for q in made),
            "fallback_below_threshold": sum(q["fallback"] for q in wanted),
            "errors": collections.Counter((q["error"] or "").split(":")[0] for q in wanted if q["error"]),
            "first_try_items_generated": len(first),
            "first_try_items_passed": sum(i["passed"] for i in first),
            "first_try_pass_rate": round(sum(i["passed"] for i in first) / len(first), 3) if first else None,
            "quizzes_needing_the_retry": sum(len(q.get("rounds", [])) > 1 for q in wanted),
            "failure_reasons": dict(reasons),
            "outside_set_citations_shown": sum(len(q.get("outside", [])) for q in made),
            "follow_up_topic_ok": [q.get("topic_has_terms") for q in wanted if q["set"] == "follow_up"],
            "seconds_median": statistics.median([q["seconds"] for q in made]) if made else None,
            "cost_usd_list_total": round(sum((q.get("usage") or {}).get("cost_usd_list", 0) for q in wanted), 6),
        }
    off = [q for q in rows if q["set"] == "off_topic"]
    out["off_topic"] = {"cases": len(off), "fallback_no_quiz": sum(q["fallback"] and not q.get("items") for q in off)}
    near = [q for q in rows if q["set"] == "near_domain"]
    out["near_domain"] = [{"id": q["id"], "run": q["run"], "top1": round(q["top1"] or 0, 3), "fallback": q["fallback"],
                           "items": len(q.get("items", []))} for q in near]
    return out


def write_summary() -> None:
    files = sorted(RUNS.glob("quiz_eval_*.json"))
    if not files:
        sys.exit("no run files yet")
    summary = {"meta": {"feature": "B Quiz me", "generated": stamp(), "contents": "aggregate numbers only (no quiz text)",
                        "case_sha256": {n: (EVAL / f"{n}.sha256").read_text().split()[0] for n in FILES}}, "models": {}}
    review = []
    for path in files:                       # one entry per run file (model and round), e.g. quiz_eval_deepseek-flash_v2
        data = json.loads(path.read_text(encoding="utf-8"))
        model = data["meta"]["model"]
        routing = data["routing"]
        summary["models"][path.stem.replace("quiz_eval_", "")] = {"model": model, "round": data["meta"].get("tag", "round1"),
            "routing": {group: {"cases": len(rows), "as_expected": sum(r["passed"] for r in rows if r["passed"] is not None),
                                "misses": [r["id"] for r in rows if r["passed"] is False]}
                        for group, rows in _group(routing).items()},
            **model_summary(data)}
        if model == settings.DEV_DEFAULT_ANSWER_MODEL:   # files are sorted, so the latest round of the default model wins
            review = _review_items(data)
    save(SUMMARY, summary)
    if review:
        _write_sheet(review)
    print(json.dumps(summary, indent=1))


def _group(rows: list[dict]) -> dict:
    groups = collections.defaultdict(list)
    for r in rows:
        groups[f"{r['file']}:{r['set']}"].append(r)
    return groups


def _review_items(data: dict) -> list[dict]:
    """15 shown items for the author's check: one per design in-scope case from run 1, rotating through the item positions."""
    pipeline_chunks = {}
    chosen = []
    design = [q for q in data["quizzes"] if q["run"] == 1 and q["set"] == "in_scope" and q["file"] == "quiz_cases" and q.get("items")]
    # Round-robin over the cases and item positions, so 15 items come from as many cases as exist (14 design quizzes -> one case
    # gives a second item) and the multiple-choice and short-answer positions are all covered.
    pairs = [(q, q["items"][(n + offset) % len(q["items"])]) for offset in range(3) for n, q in enumerate(design)]
    for q, item in pairs[:15]:
        chosen.append({"case": q["id"], "topic": q["topic"], "item": item})
        pipeline_chunks.update({i: None for i in item["chunk_ids"]})
    chunks = {c["chunk_id"]: c for c in json.loads(settings.CHUNKS_FILE.read_text(encoding="utf-8"))
              if c["chunk_id"] in pipeline_chunks}
    for entry in chosen:
        entry["cited"] = [(i, chunks[i]["text"]) for i in entry["item"]["chunk_ids"] if i in chunks]
    return chosen


def _write_sheet(entries: list[dict]) -> None:
    lines = ["# Quiz review sheet (feature B)", "",
             f"Generated {stamp()}. {len(entries)} shown items from the {settings.DEV_DEFAULT_ANSWER_MODEL} evaluation, run 1, one per "
             "design case. Check each item AGAINST ITS CITED CHUNKS only (verification against the material, not expert judgement): "
             "is the marked answer correct and supported, and is exactly one option correct? Write correct or wrong after "
             "**My verdict**. Git ignored: the chunks are licensed course text.", ""]
    for n, entry in enumerate(entries, 1):
        item = entry["item"]
        lines += ["---", "", f"## {n}. {entry['case']} · {entry['topic']} · {item['type']}", "", f"**Question:** {item['question']}", ""]
        if item["type"] == "mcq":
            lines += [f"- {k}. {v}{'  <- marked correct' if k == item['correct'] else ''}" for k, v in item["options"].items()] + [""]
        else:
            lines += [f"**Model answer:** {item['answer']}", "", "**Key points:** " + "; ".join(item["key_points"]), ""]
        lines += [f"**Explanation:** {item['explanation']}", "", "**Cited chunks:**", ""]
        for chunk_id, text in entry["cited"]:
            lines += [f"`{chunk_id}`", "", *[f"> {t}" if t.strip() else ">" for t in text.splitlines()], ""]
        lines += ["**My verdict:** ", ""]
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    SHEET.write_text("\n".join(lines), encoding="utf-8")
    print(f"review sheet: {SHEET}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Quiz me on the hashed test sets.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--tag", default="", help="a round name for a new run file, e.g. v2 (the earlier rounds are kept)")
    args = parser.parse_args()
    if args.summary:
        write_summary()
        return
    if not args.model:
        parser.error("--model is required unless --summary is given")
    run(args.model, args.runs, args.tag)


if __name__ == "__main__":
    main()
