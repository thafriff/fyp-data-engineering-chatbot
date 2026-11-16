"""
Side experiment: does the applied SYSTEM_PROMPT make false refusals more likely than the wording it replaced, or is the
Q43 and G10 behaviour within llama3.2:3b's run-to-run variance? A third arm tests a proposed request-triggered wording, and
gate() checks it against the author's conditions for applying it.

The arms: "current" is the wording that was applied when this experiment began (kept in the saved file, so its name and
meaning do not change if settings.py later does); "previous" is the wording that one replaced; "proposed" is the wording in
PROPOSED_PASSAGE, built in this process only.

This is a diagnosis only. It is not part of the notebook 04 comparison or the evaluation. It reads contexts.json and writes
only prompt_wording_check.json; settings.py, contexts.json, the manifest and the baseline files are not modified. The
previous wording is recovered from Git and used in this process only. Answers go through the chatbot's own
llm.stream_generate, so temperature 0.2, the 800-token cap and the repetition guard are the same in both arms. The two arms
differ only in the system message.

Constants live here, not in settings.py, because this script must not change that file for its own sake.

The run alternates which arm goes first, so slow periods do not fall on one arm. It saves after every answer, resumes
after an interruption, and retries errors and timeouts on the next run.

Usage (from the project root; the Ollama app must be running):
    venv\\Scripts\\python evaluation\\notebook04\\prompt_wording_check.py            run or resume, then print the report
    venv\\Scripts\\python evaluation\\notebook04\\prompt_wording_check.py --report   print the report; calls no model
"""

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scipy.stats import fisher_exact  # noqa: E402

from config import settings  # noqa: E402
from src.core import llm  # noqa: E402
from src.core.timeouts import RequestTimedOut, collect_stream  # noqa: E402

OUTPUT_FILE = settings.NB04_DIR / "prompt_wording_check.json"
MODEL = "llama3.2:3b"
REPEATS = 5
PREVIOUS_COMMIT = "d7a3cae"      # the commit whose settings.py held the wording that was replaced

CODE_IDENTIFIER_ITEMS = ["Q43", "Q56", "Q58", "Q78", "Q79"]
CONTROL_ITEMS = {"Q17": "definitional", "Q08": "conceptual", "Q22": "procedural"}
PROBE_ITEMS = ["G10"]
ITEMS = CODE_IDENTIFIER_ITEMS + list(CONTROL_ITEMS) + PROBE_ITEMS
ARMS = ["current", "previous", "proposed"]
FIRST_ARMS = ["current", "previous"]
PROPOSED_ITEMS = ["Q43", "Q56", "Q58", "Q78", "Q79", "G10", "Q22"]
PROPOSED_PASSAGE = (
    "Act as a coach, not an answer machine. If the student asks you to do a task they have been set, such as an "
    "exercise, assignment or other assessed work, or to write a complete query, program or written answer for "
    "them, guide them with hints instead: explain the concept and the reasoning, quote and explain only the "
    "fragments of the excerpts needed to teach each step, and do not assemble them into a finished, runnable or "
    "submittable whole, even if the excerpts contain one. Answer every other question fully, including how a "
    "concept, process or piece of course code works, and quote code or configuration from the excerpts where "
    "it helps."
)

FALLBACK_TEXT = "could not find this in the course materials"
CITATION = re.compile(r"\[(S0[1-7]-\d{4})\]")
INLINE_CODE = re.compile(r"`([^`\n]+)`")
CODE_LIKE = re.compile(r"[_(=.{/]|--")
# Whether the answer names the model Q43 asks for: the old baseline did, and the new one did not.
NAMES_TARGET = {"Q43": "fct_monthly_zone_revenue"}


def previous_system_prompt() -> str:
    """Rebuild the replaced SYSTEM_PROMPT from settings.py at PREVIOUS_COMMIT, without writing it anywhere."""
    source = subprocess.run(["git", "show", f"{PREVIOUS_COMMIT}:config/settings.py"], cwd=PROJECT_ROOT,
                            capture_output=True, check=True).stdout.decode("utf-8")
    values = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") in ("FALLBACK_MESSAGE", "SYSTEM_PROMPT"):
            values[node.targets[0].id] = eval(compile(ast.Expression(node.value), "settings", "eval"), {}, values)
    return values["SYSTEM_PROMPT"]


def proposed_system_prompt(current: str) -> str:
    """The current prompt with its coaching passage replaced by PROPOSED_PASSAGE. Nothing else differs."""
    start, end = current.index("Act as a coach"), current.index("After each statement")
    return current[:start] + PROPOSED_PASSAGE + " " + current[end:]


def arm_items(arm: str) -> list[str]:
    """Which items an arm covers."""
    return PROPOSED_ITEMS if arm == "proposed" else ITEMS


def sha256(text: str) -> str:
    """Hash of a prompt, recorded so that each arm's exact text can be verified."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_arm_messages(saved_current: str | None) -> tuple[dict, dict, dict]:
    """Per item and arm, the exact chat messages, plus each arm's system prompt. All arms share the frozen user message,
    which holds the same chunks and ANSWER_REMINDER. On a first run the frozen prompts must be the applied wording."""
    frozen = json.loads(settings.NB04_CONTEXTS_FILE.read_text(encoding="utf-8"))["items"]
    by_id = {i["id"]: i for i in frozen}
    current, previous = saved_current or settings.SYSTEM_PROMPT, previous_system_prompt()
    if current == previous or "do NOT give the finished answer" not in previous:
        sys.exit("The previous wording could not be recovered from Git.")
    prompts = {"current": current, "previous": previous, "proposed": proposed_system_prompt(current)}
    messages = {}
    for item_id in ITEMS:
        frozen_messages = by_id[item_id]["messages"]
        if saved_current is None and frozen_messages[0]["content"] != current:
            sys.exit(f"{item_id}: the frozen prompt is not the wording in settings.py. Nothing was run.")
        messages[item_id] = {arm: [{"role": "system", "content": text}, frozen_messages[1]]
                             for arm, text in prompts.items()}
    return messages, prompts, by_id


def analyse_answer(answer: str, retrieved_ids: set, item_id: str) -> dict:
    """The measurements for one answer. Code detection is a heuristic: a fenced block, or an inline code span holding
    code-like characters, so naming an identifier in backticks counts. has_fenced_block is the stricter measure."""
    spans = INLINE_CODE.findall(answer)
    cited = CITATION.findall(answer)
    lowered = answer.lower()
    return {
        "has_fallback_sentence": FALLBACK_TEXT in lowered,
        "fallback_only": FALLBACK_TEXT in lowered and len(answer.strip()) <= len(settings.FALLBACK_MESSAGE) + 20,
        "has_fenced_block": "```" in answer,
        "has_code": "```" in answer or any(CODE_LIKE.search(s) for s in spans),
        "has_valid_citation": any(c in retrieved_ids for c in cited),
        "citations_outside_set": sorted({c for c in cited if c not in retrieved_ids}),
        "names_target": NAMES_TARGET[item_id] in answer if item_id in NAMES_TARGET else None,
        "estimated_tokens": round(len(answer) / 4),
    }


def run_settings(prompts: dict) -> dict:
    """What must match for a resumed run to be comparable."""
    return {"model": MODEL, "temperature": settings.LLM_TEMPERATURE, "max_answer_tokens": settings.DEV_LLM_MAX_ANSWER_TOKENS,
            "num_ctx": settings.DEV_LLM_NUM_CTX, "request_timeout_seconds": settings.NB04_REQUEST_TIMEOUT_SECONDS,
            "repeats": REPEATS, "items": ITEMS,
            "system_prompt_sha256": {arm: sha256(prompts[arm]) for arm in FIRST_ARMS}}


def proposed_settings(prompts: dict) -> dict:
    """What must match for the third arm to resume. It is kept apart so the first two arms' saved results stay valid."""
    return {"items": PROPOSED_ITEMS, "repeats": REPEATS, "system_prompt_sha256": sha256(prompts["proposed"])}


def load_results(settings_now: dict, prompts: dict) -> dict:
    """The saved file, or a new one. Stops if it was made under different settings."""
    if OUTPUT_FILE.exists():
        data = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
        third = data.setdefault("run_settings_proposed", proposed_settings(prompts))
        if data["run_settings"] != settings_now or third != proposed_settings(prompts):
            sys.exit(f"{OUTPUT_FILE} was made with different settings. Nothing was run.")
        data["meta"]["system_prompts"]["proposed"] = prompts["proposed"]
        return data
    return {
        "meta": {"purpose": "Side experiment on SYSTEM_PROMPT wording; not part of the model comparison or the evaluation.",
                 "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "controls": CONTROL_ITEMS, "code_identifier_items": CODE_IDENTIFIER_ITEMS, "probe_items": PROBE_ITEMS,
                 "previous_wording_source": f"git show {PREVIOUS_COMMIT}:config/settings.py",
                 "system_prompts": prompts, "contains_chunk_text": False},
        "run_settings": settings_now, "run_settings_proposed": proposed_settings(prompts), "results": {}}


def save(data: dict) -> None:
    """Atomic write (temp file + rename), so an interruption cannot corrupt the file."""
    data["report"] = build_report(data["results"])
    tmp = OUTPUT_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(OUTPUT_FILE)


def planned_runs() -> list[tuple]:
    """Every (repeat, item, arm). The first arm alternates by repeat, so neither arm always runs first."""
    runs = []
    for repeat in range(1, REPEATS + 1):
        order = FIRST_ARMS if repeat % 2 else list(reversed(FIRST_ARMS))
        for item_id in ITEMS:
            runs += [(repeat, item_id, arm) for arm in order]
    return runs + [(repeat, item_id, "proposed") for repeat in range(1, REPEATS + 1) for item_id in PROPOSED_ITEMS]


def stop_reason(answer: str) -> str:
    """Why generation stopped, from the note the chatbot's guards add to the answer."""
    if answer.endswith(settings.ANSWER_REPEAT_NOTE):
        return "repetition_guard"
    if answer.endswith(settings.ANSWER_CUT_NOTE):
        return "length_cap"
    return "finished"


def generate(messages: list[dict], question: str) -> tuple[str, float]:
    """One answer through the chatbot's generator, with these exact messages. The prompt builder is replaced for this
    process only, which is how the previous arm's wording reaches the model without touching settings.py."""
    llm.build_messages = lambda _question, _chunks, **_hint: messages     # the hint argument is ignored: messages are fixed
    start = time.perf_counter()
    answer = collect_stream(lambda: llm.stream_generate(question, [], model=MODEL),
                            settings.NB04_REQUEST_TIMEOUT_SECONDS)
    return answer, round(time.perf_counter() - start, 2)


def run_all(data: dict, messages: dict, frozen: dict) -> None:
    """Generate whatever is missing or failed, saving after each answer."""
    todo = [r for r in planned_runs()
            if f"{r[1]}|{r[2]}|{r[0]}" not in data["results"] or data["results"][f"{r[1]}|{r[2]}|{r[0]}"]["error"]]
    print(f"{len(planned_runs())} answers planned, {len(todo)} to run.")
    for n, (repeat, item_id, arm) in enumerate(todo, 1):
        record = {"item": item_id, "arm": arm, "repeat": repeat, "answer": None, "stopped_by": None, "seconds": None,
                  "error": None}
        try:
            answer, seconds = generate(messages[item_id][arm], frozen[item_id]["text"])
            retrieved = {c["chunk_id"] for c in frozen[item_id]["chunks"]}
            record.update(answer=answer, seconds=seconds, stopped_by=stop_reason(answer),
                          **analyse_answer(answer, retrieved, item_id))
        except RequestTimedOut as error:
            record.update(stopped_by="timeout", error=str(error), elapsed_seconds=round(error.elapsed_seconds, 2))
        except Exception as error:  # recorded and retried on the next run
            record["error"] = f"{type(error).__name__}: {error}"
        data["results"][f"{item_id}|{arm}|{repeat}"] = record
        save(data)
        print(f"[{n}/{len(todo)}] {item_id} {arm} r{repeat} {record['seconds']}s {record['stopped_by']} "
              f"fallback={record.get('has_fallback_sentence')} error={record['error']}", flush=True)


def build_report(results: dict) -> dict:
    """Counts per item and arm, and fallback comparisons between arms. Failed answers are left out and counted."""
    table, totals = [], {arm: {"fallback": 0, "fallback_only": 0, "answers": 0} for arm in ARMS}
    for item_id in ITEMS:
        for arm in (a for a in ARMS if item_id in arm_items(a)):
            done = [r for r in results.values() if r["item"] == item_id and r["arm"] == arm and not r["error"]]
            count = lambda field: sum(bool(r.get(field)) for r in done)    # noqa: E731
            seconds = sorted(r["seconds"] for r in done)
            table.append({"item": item_id, "arm": arm, "answers": len(done), "fallback": count("has_fallback_sentence"),
                          "fallback_only": count("fallback_only"), "code": count("has_code"),
                          "fenced": count("has_fenced_block"), "citation": count("has_valid_citation"),
                          "outside_set_answers": count("citations_outside_set"), "names_target": count("names_target"),
                          "mean_tokens": round(sum(r["estimated_tokens"] for r in done) / len(done)) if done else None,
                          "median_seconds": seconds[len(seconds) // 2] if seconds else None})
            totals[arm]["fallback"] += count("has_fallback_sentence")
            totals[arm]["fallback_only"] += count("fallback_only")
            totals[arm]["answers"] += len(done)

    def fisher(rows: list[dict], first: str, second: str) -> float | None:
        """Two-sided Fisher exact test on fallback answers between two arms."""
        a = {arm: [sum(r[k] for r in rows if r["arm"] == arm) for k in ("fallback", "answers")]
             for arm in (first, second)}
        if not all(a[arm][1] for arm in a):
            return None
        grid = [[a[arm][0], a[arm][1] - a[arm][0]] for arm in a]
        return round(float(fisher_exact(grid)[1]), 4)

    shared = [r for r in table if r["item"] in PROPOSED_ITEMS]
    return {"table": table, "totals": totals, "fisher_p_pooled": fisher(table, "current", "previous"),
            "fisher_p_per_item": {i: fisher([r for r in table if r["item"] == i], "current", "previous") for i in ITEMS},
            "fisher_p_proposed_vs_current_shared_items": fisher(shared, "proposed", "current"),
            "gate": gate(table),
            "failed_answers": sum(bool(r["error"]) for r in results.values())}


def gate(table: list[dict]) -> dict:
    """The author's conditions for applying the proposed wording, each with its measured value and threshold."""
    row = {r["item"]: r for r in table if r["arm"] == "proposed"}
    if set(row) != set(PROPOSED_ITEMS) or any(row[i]["answers"] != REPEATS for i in PROPOSED_ITEMS):
        return {"complete": False, "passed": False, "conditions": []}
    only = sum(r["fallback_only"] for r in row.values())
    contains = sum(r["fallback"] for r in row.values())
    conditions = [("No answer is only the fallback sentence", f"{only} of 35", "0", only == 0)]
    conditions += [(f"{i} fenced code", f"{row[i]['fenced']} of 5", "at least 4", row[i]["fenced"] >= 4) for i in ("Q78", "Q79")]
    conditions += [(f"{i} code", f"{row[i]['code']} of 5", "at least 4", row[i]["code"] >= 4) for i in ("Q56", "Q58")]
    conditions += [("Q22 fallback sentence", f"{row['Q22']['fallback']} of 5", "0", row["Q22"]["fallback"] == 0),
                   ("Q22 valid citation", f"{row['Q22']['citation']} of 5", "at least 4", row["Q22"]["citation"] >= 4),
                   ("Answers containing the fallback sentence", f"{contains} of 35", "at most 2", contains <= 2)]
    return {"complete": True, "passed": all(c[3] for c in conditions),
            "conditions": [{"condition": c[0], "result": c[1], "threshold": c[2], "passed": c[3]} for c in conditions]}


def print_report(report: dict) -> None:
    """The table per item and arm, then the comparison between arms."""
    print("\n| Item | Arm | Answers | Fallback | Fallback only | Code | Fenced | Citation | Outside set | Mean tokens |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in report["table"]:
        code = (f"{r['code']}", f"{r['fenced']}") if r["item"] not in CONTROL_ITEMS else ("n/a", "n/a")
        print(f"| {r['item']} | {r['arm']} | {r['answers']} | {r['fallback']} | {r['fallback_only']} | {code[0]} | "
              f"{code[1]} | {r['citation']} | {r['outside_set_answers']} | {r['mean_tokens']} |")
    print("\nQ43 names fct_monthly_zone_revenue:",
          {r["arm"]: r["names_target"] for r in report["table"] if r["item"] == "Q43"})
    print("Totals:", report["totals"], "| Fisher p, current vs previous:", report["fisher_p_pooled"],
          "| proposed vs current on shared items:", report["fisher_p_proposed_vs_current_shared_items"],
          "| failed answers:", report["failed_answers"])
    print(f"\nGATE: {'PASSED' if report['gate']['passed'] else 'NOT PASSED'} (complete: {report['gate']['complete']})")
    for c in report["gate"]["conditions"]:
        print(f"  {'pass' if c['passed'] else 'FAIL'} | {c['condition']}: {c['result']} (threshold {c['threshold']})")


def main() -> None:
    """Run or resume the experiment, then print the report."""
    parser = argparse.ArgumentParser(description="SYSTEM_PROMPT wording check on llama3.2:3b.")
    parser.add_argument("--report", action="store_true", help="print the report from the saved file; calls no model")
    args = parser.parse_args()
    if args.report:
        print_report(json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))["report"])
        return
    ok, message = llm.check_backend(MODEL)
    if not ok:
        sys.exit(message)
    saved = json.loads(OUTPUT_FILE.read_text(encoding="utf-8")) if OUTPUT_FILE.exists() else None
    messages, prompts, frozen = load_arm_messages(saved["meta"]["system_prompts"]["current"] if saved else None)
    data = load_results(run_settings(prompts), prompts)
    run_all(data, messages, frozen)
    save(data)
    print_report(data["report"])


if __name__ == "__main__":
    main()
