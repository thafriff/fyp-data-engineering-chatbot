"""
Evaluation of multi-turn chat on the fixed, hashed sets evaluation/multiturn_cases.json and multiturn_cases_holdout2.json
(author's approved design, 2026-11-26). Each file is checked against its recorded hash; a file's "regroup" entry moves cases
between groups (H01-H03 became tuning cases on 2026-11-26).
DEV PROTOTYPE, a system feature evaluated in notebook 05; not part of notebook 04.

Each conversation is played --runs times, each time from a new, empty conversation, through the real chat pipeline
(RagPipeline.answer with a Conversation, the default local model), the way the app does it: the answer is collected, a
coach-mode answer gets the app's code cap, and the turn is recorded with conversation.record(). Every expectation in the case
file is checked:
- deterministic checks (routing, search mode, flags, history size, retrieval) must pass in EVERY run;
- answer checks (answered with valid citations, a table) must each pass in at least 2 of 3 runs.
Results are saved after every conversation run (atomic), so a stopped evaluation keeps its data. Design cases (C) and held-out
cases (H) are reported separately.

Usage (from the project root, Ollama running):
    venv\\Scripts\\python evaluation\\multiturn_eval.py OUT.json [--runs 3] [--only C01,C02]
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from config import settings  # noqa: E402
from src.core import citations, guardrail  # noqa: E402
from src.core import conversation as conversation_rules  # noqa: E402
from src.core.conversation import Conversation  # noqa: E402
from src.core.rag import RagPipeline  # noqa: E402

CASE_FILES = [ROOT / "evaluation" / "multiturn_cases.json", ROOT / "evaluation" / "multiturn_cases_holdout2.json",
              ROOT / "evaluation" / "multiturn_cases_v3.json"]
OLD_EXAMPLE = "OLTP databases are normalized"      # the v2b opening-rule example, replaced on 2026-11-26
TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")


def has_table(answer: str) -> bool:
    """A markdown table: a header line, a separator line, then at least 2 data lines."""
    lines = [l for l in answer.splitlines() if l.strip()]
    for i in range(1, len(lines)):
        if TABLE_SEPARATOR.match(lines[i]) and lines[i - 1].strip().startswith("|"):
            return sum(1 for l in lines[i + 1:i + 3] if l.strip().startswith("|")) >= 2
    return False


def observe(result, answer: str, previous_top5: list[str] | None, previous_answer: str | None = None) -> dict:
    """Everything the case files can check for one turn."""
    top5 = [c.chunk_id for c in result.chunks]
    check = citations.check(answer, top5) if result.model else None
    return {
        "kind": result.intent, "path": result.routing.path if result.routing else None,
        "search": result.search_mode or "none", "search_text": result.search_query, "fallback": result.used_fallback,
        "coach": result.coach_mode, "comparison": result.comparison_format,
        "offer": bool(result.routing and result.routing.offer_question), "resolved_question": result.resolved_question,
        "model_called": result.model is not None, "history_exchanges": result.history_exchanges,
        "top5": top5, "top1_score": round(result.chunks[0].score, 3) if result.chunks else None,
        "overlap_with_previous_top5": len(set(top5) & set(previous_top5)) if previous_top5 is not None and top5 else None,
        "top_texts_lower": [c.text.lower() for c in result.chunks],
        "valid_citations": check.counts["valid"] if check else 0,
        "outside_set": check.outside_set if check else [], "malformed": check.malformed if check else [],
        "is_fallback_sentence": answer.strip() == settings.FALLBACK_MESSAGE, "table": has_table(answer),
        "repeat_share_of_previous": (round(conversation_rules.repeat_share(answer, previous_answer), 3)
                                     if result.model and previous_answer else None),
        "app_instruction_text": bool(result.model) and conversation_rules.contains_app_instruction(answer),
        "regenerated": result.regenerated,
        "answer": answer,
    }


def deterministic_failures(expected: dict, seen: dict) -> list[str]:
    """One message per deterministic expectation that does not hold."""
    out = []
    for key, want in expected.items():
        if key == "min_overlap_with_previous_top5":
            got = seen["overlap_with_previous_top5"]
            ok = got is not None and got >= want
        elif key == "top1_contains":
            got = seen["top_texts_lower"][0][:80] if seen["top_texts_lower"] else None
            ok = bool(seen["top_texts_lower"]) and want.lower() in seen["top_texts_lower"][0]
        elif key == "top5_contains_all":
            got = [w for w in want if any(w.lower() in t for t in seen["top_texts_lower"])]
            ok = len(got) == len(want)
        else:
            got = seen[key]
            ok = got == want
        if not ok:
            out.append(f"{key}: expected {want!r}, got {got!r}")
    return out


def answer_results(expected: dict, seen: dict) -> dict[str, bool]:
    """Pass/fail of each answer expectation for one turn in one run."""
    out = {}
    for key, want in expected.items():
        if key == "answered":
            ok = (seen["model_called"] and not seen["is_fallback_sentence"] and seen["valid_citations"] > 0
                  and not seen["outside_set"] and not seen["malformed"])
        elif key == "table":
            ok = seen["table"]
        elif key == "fresh":           # multiturn_cases_v3.json: under 50% of its word 5-grams repeat the previous answer
            ok = seen["model_called"] and (seen["repeat_share_of_previous"] or 0.0) < 0.5
        elif key == "no_app_instruction":
            ok = seen["model_called"] and not seen["app_instruction_text"]
        else:
            raise ValueError(f"unknown answer check {key}")
        out[key] = ok == want
    return out


def play(pipeline: RagPipeline, conversation_case: dict) -> list[dict]:
    """One run of one conversation from an empty conversation; returns one record per user turn."""
    conversation = Conversation()
    previous_top5, previous_answer, records = None, None, []
    for number, turn in enumerate(conversation_case["turns"], 1):
        if turn.get("action") == "new_chat":
            conversation.clear()
            previous_top5 = previous_answer = None
            continue
        start = time.perf_counter()
        result = pipeline.answer(turn["user"], conversation=conversation)
        answer = "".join(result.answer_stream)
        if result.coach_mode:                       # the app caps long code blocks in coach mode before showing them
            answer, _ = guardrail.cap_code_blocks(answer)
        conversation.record(result, answer)
        seen = observe(result, answer, previous_top5, previous_answer)
        if seen["model_called"]:
            previous_top5, previous_answer = seen["top5"], answer
        seen["seconds"] = round(time.perf_counter() - start, 1)
        det = deterministic_failures(turn.get("deterministic", {}), seen)
        ans = answer_results(turn.get("answer", {}), seen)
        records.append({"turn": number, "user": turn["user"], "deterministic_failures": det, "answer_checks": ans,
                        **{k: v for k, v in seen.items() if k != "top_texts_lower"}})
    return records


def verdict(runs: list[list[dict]]) -> tuple[bool, list[str]]:
    """The author's pass rule: every deterministic check in all runs, every answer check in at least 2 of 3 runs."""
    reasons = []
    for r, records in enumerate(runs, 1):
        for rec in records:
            for failure in rec["deterministic_failures"]:
                reasons.append(f"run {r} turn {rec['turn']} [deterministic] {failure}")
    by_check = {}
    for records in runs:
        for rec in records:
            for key, ok in rec["answer_checks"].items():
                by_check.setdefault((rec["turn"], key), []).append(ok)
    needed = 2 if len(runs) >= 3 else len(runs)
    for (turn, key), oks in sorted(by_check.items()):
        if sum(oks) < needed:
            reasons.append(f"turn {turn} [answer] {key}: passed in {sum(oks)} of {len(oks)} runs")
    return not reasons, reasons


def save(path: Path, data: dict) -> None:
    """Atomic write, so an interrupted evaluation never leaves a broken file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-turn evaluation on the hashed case set.")
    parser.add_argument("out", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--only", default=None, help="comma-separated conversation ids")
    args = parser.parse_args()
    cases, regroup, hashes = [], {}, {}
    for path in CASE_FILES:
        recorded = path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
        if hashlib.sha256(path.read_bytes()).hexdigest() != recorded:
            sys.exit(f"{path.name} does not match its recorded hash; it must not be edited.")
        content = json.loads(path.read_text(encoding="utf-8"))
        cases += content["conversations"]
        regroup.update(content.get("regroup", {}))
        hashes[path.name] = recorded
    for case in cases:
        case["group"] = regroup.get(case["id"], case["group"])
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    pipeline = RagPipeline()
    data = {"meta": {"model": settings.DEV_LLM_MODEL, "runs": args.runs, "case_sha256": hashes, "finished": False,
                     "regroup": regroup, "opening_instruction": settings.ANSWER_OPENING_INSTRUCTION,
                     "history_max_exchanges": settings.HISTORY_MAX_EXCHANGES, "system_prompt_version": settings.SYSTEM_PROMPT_VERSION},
            "conversations": {}}
    for case in cases:
        runs = []
        for run in range(1, args.runs + 1):
            runs.append(play(pipeline, case))
            data["conversations"][case["id"]] = {"group": case["group"], "case": case["case"], "runs": runs}
            save(args.out, data)
        ok, reasons = verdict(runs)
        data["conversations"][case["id"]].update(passed=ok, reasons=reasons)
        save(args.out, data)
        print(f"{'PASS' if ok else 'FAIL'} {case['id']} ({case['group']}) {case['case']}", flush=True)
        for reason in reasons:
            print(f"      {reason}", flush=True)
    data["meta"]["finished"] = True
    save(args.out, data)
    for group in ("design", "tuning", "held_out"):
        ids = [k for k, v in data["conversations"].items() if v["group"] == group]
        passed = [k for k in ids if data["conversations"][k]["passed"]]
        print(f"{group}: {len(passed)} of {len(ids)} passed; failed: {[k for k in ids if k not in passed]}")
    # Grounding leak (the author's target after the 2026-11-26 swap: S06-0205 never cited unless retrieved).
    answers = [t for c in data["conversations"].values() for run in c["runs"] for t in run if t["model_called"]]
    leak = {"model_answers": len(answers),
            "S06-0205_cited_outside_retrieved": sum("S06-0205" in t["outside_set"] for t in answers),
            "old_example_sentence": sum(OLD_EXAMPLE in t["answer"] for t in answers),
            "S06-0322_cited_outside_retrieved": sum("S06-0322" in t["outside_set"] for t in answers),
            "any_outside_set": sum(bool(t["outside_set"]) for t in answers),
            # multiturn_cases_v3.json global check: no model answer may contain app-layer instruction text (target 0)
            "app_instruction_text": sum(bool(t.get("app_instruction_text")) for t in answers),
            "regenerated": {reason: sum(t.get("regenerated") == reason for t in answers)
                            for reason in ("repeats_earlier_answer", "app_instruction_text")}}
    data["meta"]["leak"] = leak
    save(args.out, data)
    print("leak:", leak)
    print("saved", args.out)


if __name__ == "__main__":
    main()
