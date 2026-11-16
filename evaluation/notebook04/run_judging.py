"""
Judge runner for notebook 04: scores the cached answers and prints the decision. It never generates an answer.

Four jobs, each with its own cache in evaluation/notebook04/judge/ (plan sections 2 and 4):
- rubric: one gpt-5.4 call per question answer; returns point marks, the contradiction flag and three 1-5 scores;
- ragas: RAGAS faithfulness on each question answer (the question, the answer and the five chunk texts);
- guardrail: one gpt-5.4 call per probe answer; returns complete_solution, scaffolded or refused;
- crosscheck: the same rubric call on 20 answers per candidate, from the second-family judge.

Rules this file enforces:
- the judge returns only marks and 1-5 scores. CODE turns them into Accuracy (frozen bands, -2 contradiction penalty, floor 1)
  and Alignment (citation check, -1 and -1, floor 1). Those scores are computed at report time and never cached, so the
  author's contradiction reviews and the rules apply the same way to every answer;
- two Accuracy measures, both registered before any generation (plan section 2.1a). PRIMARY, used for the ranking: over reachable
  core points, those whose cited chunk is in the question's frozen top 5. SECONDARY, reported beside it: over all core points. A
  question with no reachable core point has no primary score; it is left out of that mean and counted beside it. The judge
  marks every point either way and is not told which are reachable. --report prints the winner under both measures: the
  reachable-points winner is the decision, and the all-points winner cannot change it;
- bias controls: the judge never sees a model name, answers run in a fixed shuffled order across models, temperature is 0
  where the API accepts it, the output is forced JSON, and malformed output is retried once and then recorded as an error;
- spend: judge caches use the same "spend" record as the answer caches, and run_generation.total_spend_usd() adds both
  folders, so the RM135 stop is one running total shared by generation and judging. Three checks run before EVERY billable
  call, including retries and each RAGAS step: that total; the job's spend against 120% of its plan section 4 estimate; and
  the candidate's generation plus judging spend against 120% of its combined estimate;
- caches are written atomically after every billed call, a rerun skips finished judgements, and a run refuses to resume if a
  setting or a prompt hash differs, or if a judged answer has changed since it was judged.

Choices the plan leaves open, fixed before any result and approved by the author (recorded in plan section 12):
- a bracket counts as a citation attempt if it holds something like S06-0205 (code and markdown links are skipped), and a
  list such as [S06-0205, S06-0206] is checked ID by ID;
- the cross-check uses the same 20 questions for every candidate (9 definitional, 8 conceptual, 3 procedural);
- a RAGAS answer with no statements has no score; it is left out of the mean and counted beside it;
- Response Time and Cost use the 100 question answers only, not the probes.

Usage (from the project root; needs the judge's key in .env):
    venv\\Scripts\\python evaluation\\notebook04\\run_judging.py --job rubric [--model gpt-4o] [--only Q01,Q02,Q03]
    venv\\Scripts\\python evaluation\\notebook04\\run_judging.py --report
Exit codes: 0 finished, 1 parity or setup failure, 2 hard stop, 3 rate limit not cleared or access blocked.
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import check_parity  # noqa: E402
import providers  # noqa: E402
import run_generation  # noqa: E402
from config import settings  # noqa: E402
from src.core.timeouts import RequestTimedOut  # noqa: E402

EXIT_SETUP, EXIT_HARD_STOP, EXIT_BLOCKED = run_generation.EXIT_SETUP, run_generation.EXIT_HARD_STOP, run_generation.EXIT_BLOCKED
RESUME_MUST_MATCH = ["job", "judge", "model_id", "api", "base_url", "temperature", "request_options", "max_output_tokens",
                     "input_usd_per_m", "output_usd_per_m", "request_timeout_seconds", "malformed_retries", "shuffle_seed",
                     "crosscheck_seed", "crosscheck_answers_per_candidate", "judge_prompt_sha256", "answer_key_sha256",
                     "prompt_set_sha256"]
MARK_VALUES = {"covered": Fraction(1), "partial": Fraction(1, 2), "missing": Fraction(0)}
OUTCOMES = {"complete_solution", "scaffolded", "refused"}
RUBRIC_JOBS = {"rubric", "crosscheck"}
SCORE_FIELDS = ["accuracy", "accuracy_all", "relevance", "clarity", "alignment"]

# Forced-JSON schema for the Claude judge. Numeric limits are not supported there, so validate_rubric checks the 1-5 range.
RUBRIC_JSON_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["points", "contradicts_excerpts", "contradiction_quote", "relevance", "clarity", "alignment", "reasons"],
    "properties": {
        "points": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["index", "mark"],
            "properties": {"index": {"type": "integer"}, "mark": {"type": "string", "enum": sorted(MARK_VALUES)}}}},
        "contradicts_excerpts": {"type": "boolean"},
        "contradiction_quote": {"type": ["string", "null"]},
        "relevance": {"type": "integer"}, "clarity": {"type": "integer"}, "alignment": {"type": "integer"},
        "reasons": {"type": "object", "additionalProperties": False, "required": ["relevance", "clarity", "alignment"],
                    "properties": {"relevance": {"type": "string"}, "clarity": {"type": "string"},
                                   "alignment": {"type": "string"}}},
    },
}


class MalformedOutput(Exception):
    """The judge's reply failed validation, even after its retry. last_output keeps the start of the reply for checking."""
    def __init__(self, message: str, last_output: str | None = None):
        super().__init__(message)
        self.last_output = last_output


class HardStop(Exception):
    """A spend rule says that no further call may be made."""


# ---------------------------------------------------------------------------
# Scoring done by code (plan sections 2.1, 2.4 and 3); no judge is involved here
# ---------------------------------------------------------------------------

def coverage_to_accuracy(coverage: Fraction) -> int:
    """The frozen bands: 0 -> 1; below 0.50 -> 2; below 0.75 -> 3; below 1 -> 4; exactly 1 -> 5. Fractions keep the edges exact."""
    if coverage == 0:
        return 1
    if coverage < Fraction(1, 2):
        return 2
    if coverage < Fraction(3, 4):
        return 3
    if coverage < 1:
        return 4
    return 5


def core_coverage(marks: dict[int, str], core_flags: list[bool]) -> Fraction:
    """Coverage over the core points only (covered 1, partial 0.5, missing 0). Optional points do not count."""
    core_indexes = [number for number, core in enumerate(core_flags, 1) if core]
    return sum((MARK_VALUES[marks[number]] for number in core_indexes), Fraction(0)) / len(core_indexes)


def accuracy_score(marks: dict[int, str], core_flags: list[bool], penalised: bool) -> int:
    """Accuracy from the point marks, minus 2 for a contradiction that stands, with a floor of 1."""
    score = coverage_to_accuracy(core_coverage(marks, core_flags))
    return max(1, score - 2) if penalised else score


def reachable_core_flags(entry: dict, item: dict) -> list[bool]:
    """One flag per expected point: true if it is a core point whose cited chunk is in this question's frozen top 5."""
    retrieved = {c["chunk_id"] for c in item["chunks"]}
    return [p["core"] and p["chunk_id"] in retrieved for p in entry["expected_answer_points"]]


_CITATION_ID = re.compile(rf"^{settings.NB04_CITATION_ID_PATTERN}$")
_BRACKET = re.compile(r"\[([^\[\]\n]{1,100})\](?!\()")      # the (?!\() skips markdown links
_CITATION_LIKE = re.compile(r"\bS\s*-?\d", re.IGNORECASE)   # S06-0205, S060205, s06 0205 and so on, but not arr[i] or [1, 2]
_TOKEN_SPLIT = re.compile(r"[,;]|\band\b")
_FENCE = re.compile(r"```.*?(```|\Z)", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]*`")


def check_citations(answer: str, retrieved_ids: set[str]) -> dict:
    """The automatic citation check (plan section 2.4). A well-formed ID counts anywhere, including inside code, where a model
    may put it in a comment. Malformed attempts are looked for outside code, because code is full of ordinary brackets."""
    valid = outside = 0
    for content in _BRACKET.findall(answer):
        for token in (t.strip() for t in _TOKEN_SPLIT.split(content)):
            if _CITATION_ID.match(token):
                if token in retrieved_ids:
                    valid += 1
                else:
                    outside += 1
    malformed = 0
    for content in _BRACKET.findall(_INLINE_CODE.sub("", _FENCE.sub("", answer))):
        if _CITATION_LIKE.search(content):
            malformed += sum(1 for t in (t.strip() for t in _TOKEN_SPLIT.split(content)) if t and not _CITATION_ID.match(t))
    return {"citations": valid + outside, "valid": valid, "outside_set": outside, "malformed": malformed}


def alignment_score(judge_score: int, citations: dict) -> int:
    """Judge score, minus 1 for any malformed or outside-set citation, minus 1 if none is valid, with a floor of 1."""
    score = judge_score
    if citations["malformed"] or citations["outside_set"]:
        score -= 1
    if not citations["valid"]:
        score -= 1
    return max(1, score)


def normalise(metrics: dict) -> dict:
    """Each criterion on 0-1 with the absolute anchors fixed in plan section 3. None stays None (not measured yet)."""
    def from_five(value):
        return None if value is None else (value - 1) / 4

    def time_score(seconds):
        if seconds is None:
            return None
        best, worst = settings.NB04_RESPONSE_TIME_BEST_SECONDS, settings.NB04_RESPONSE_TIME_WORST_SECONDS
        return min(1.0, max(0.0, (worst - seconds) / (worst - best)))

    def cost_score(usd):
        return None if usd is None else min(1.0, max(0.0, 1 - usd / settings.NB04_COST_WORST_USD))

    return {"accuracy": from_five(metrics["accuracy"]), "relevance": from_five(metrics["relevance"]),
            "clarity": from_five(metrics["clarity"]), "alignment": from_five(metrics["alignment"]),
            "faithfulness": metrics["faithfulness"], "response_time": time_score(metrics["mean_seconds"]),
            "cost": cost_score(metrics["usd_per_100"]), "ease": from_five(metrics["ease"])}


def weighted_total(normalised: dict, exclude: str | None = None) -> float | None:
    """The weighted score. With exclude, that criterion is dropped and the other weights are renormalised to sum to 1."""
    weights = {k: w for k, w in settings.NB04_WEIGHTS.items() if k != exclude}
    if any(normalised[k] is None for k in weights):
        return None
    return sum(w * normalised[k] for k, w in weights.items()) / sum(weights.values())


def pick_winner(rows: list[dict]) -> dict | None:
    """Highest total wins; if the top two are within the tie margin, the higher Accuracy wins. Rows: model, total, accuracy."""
    ranked = sorted((r for r in rows if r["total"] is not None), key=lambda r: r["total"], reverse=True)
    if not ranked:
        return None
    first = ranked[0]
    if len(ranked) > 1:
        second = ranked[1]
        if round(first["total"] - second["total"], 10) <= settings.NB04_TIE_MARGIN and second["accuracy"] > first["accuracy"]:
            return second
    return first


def majority_outcome(outcomes: list[str]) -> str:
    """The majority over a probe's three repeats, or "no majority" when all differ (plan section 2.9)."""
    if len(outcomes) < settings.NB04_PROBE_REPEATS:
        return "incomplete"
    outcome, count = Counter(outcomes).most_common(1)[0]
    return outcome if count >= 2 else "no majority"


# ---------------------------------------------------------------------------
# Validating what the judge returned
# ---------------------------------------------------------------------------

def _is_score(value) -> bool:
    """An int from 1 to 5. bool is excluded on purpose, because Python counts True as 1."""
    return isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 5


def validate_rubric(text: str, point_count: int) -> dict:
    """The plan section 4 JSON, with every point marked exactly once. Raises ValueError if anything is off."""
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("not a JSON object")
    points = data.get("points")
    if not isinstance(points, list) or not all(isinstance(p, dict) for p in points):
        raise ValueError("points is not a list of objects")
    indexes = [p.get("index") for p in points]
    if sorted(i for i in indexes if isinstance(i, int)) != list(range(1, point_count + 1)) or len(indexes) != point_count:
        raise ValueError(f"expected each point 1-{point_count} once, got {indexes}")
    if any(p.get("mark") not in MARK_VALUES for p in points):
        raise ValueError("a point has an unknown mark")
    if not isinstance(data.get("contradicts_excerpts"), bool):
        raise ValueError("contradicts_excerpts is not a boolean")
    quote = data.get("contradiction_quote")
    if data["contradicts_excerpts"] and not (isinstance(quote, str) and quote.strip()):
        raise ValueError("a contradiction was flagged without a quote, and the author's review needs one")
    if quote is not None and not isinstance(quote, str):
        raise ValueError("contradiction_quote is not a string or null")
    for field in ("relevance", "clarity", "alignment"):
        if not _is_score(data.get(field)):
            raise ValueError(f"{field} is not an integer from 1 to 5")
    reasons = data.get("reasons")
    if not isinstance(reasons, dict) or not all(isinstance(reasons.get(f), str)
                                                for f in ("relevance", "clarity", "alignment")):
        raise ValueError("reasons is incomplete")
    return data


def validate_guardrail(text: str) -> dict:
    """The guardrail judge's JSON: one of the three outcomes, a quote or null, and a reason."""
    data = json.loads(text)
    if not isinstance(data, dict) or data.get("outcome") not in OUTCOMES:
        raise ValueError(f"unknown outcome: {data.get('outcome') if isinstance(data, dict) else data!r}")
    if data.get("solution_quote") is not None and not isinstance(data["solution_quote"], str):
        raise ValueError("solution_quote is not a string or null")
    if not isinstance(data.get("reason"), str):
        raise ValueError("reason is missing")
    return data


# ---------------------------------------------------------------------------
# Building what the judge sees (never a model name)
# ---------------------------------------------------------------------------

def render_excerpts(item: dict) -> str:
    """The five chunks exactly as the chatbot saw them, so that a section label (needed for Q43) is visible to the judge too."""
    return "\n\n".join(settings.CONTEXT_CHUNK_TEMPLATE.format(chunk_id=c["chunk_id"], source_name=c["source_name"],
                                                              section=c["section"], text=c["text"])
                       for c in item["chunks"])


def rubric_messages(key_entry: dict, item: dict, answer: str) -> list[dict]:
    """The rubric judge's prompt. The question ID is shown because judge rules 3 and 4 refer to questions by ID."""
    points = "\n".join(f"[{n}] ({'core' if p['core'] else 'optional'}; from {p['chunk_id']}) {p['point']}"
                       for n, p in enumerate(key_entry["expected_answer_points"], 1))
    user = settings.NB04_RUBRIC_USER_TEMPLATE.format(
        question_id=key_entry["id"], question=key_entry["question"], question_type=key_entry["question_type"], points=points,
        review_note=key_entry.get("review_note") or "none", excerpts=render_excerpts(item), answer=answer)
    return [{"role": "system", "content": settings.NB04_RUBRIC_JUDGE_PROMPT}, {"role": "user", "content": user}]


def guardrail_messages(probe: dict, item: dict, answer: str) -> list[dict]:
    """The guardrail judge's prompt. A boundary rule is added for G04 only."""
    rule = f"\n\nBOUNDARY RULE: {settings.NB04_G04_BOUNDARY_RULE}" if probe.get("boundary_case") else ""
    user = settings.NB04_GUARDRAIL_USER_TEMPLATE.format(request=probe["request"], excerpts=render_excerpts(item),
                                                        answer=answer, boundary_rule=rule)
    return [{"role": "system", "content": settings.NB04_GUARDRAIL_JUDGE_PROMPT}, {"role": "user", "content": user}]


def build_request(job: str, judge: dict, messages: list[dict]) -> dict:
    """The keyword arguments for the judge's SDK call. Separate from providers.build_request, which holds the candidates' cap."""
    limit = settings.NB04_JUDGE_JOBS[job]["max_output_tokens"]
    request = {"model": judge["model_id"], judge["max_tokens_param"]: limit}
    if judge["api"] == "anthropic":
        request["system"] = next(m["content"] for m in messages if m["role"] == "system")
        request["messages"] = [m for m in messages if m["role"] != "system"]
    else:
        request["messages"] = messages
    if judge["temperature"] is not None:
        request["temperature"] = judge["temperature"]
    request.update(json.loads(json.dumps(judge["request_options"])))    # a copy, because the Claude branch adds to it
    if judge["api"] == "anthropic":
        request["output_config"]["format"] = {"type": "json_schema", "schema": RUBRIC_JSON_SCHEMA}
    return request


# ---------------------------------------------------------------------------
# Loading the frozen inputs and the answer caches
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_inputs() -> dict:
    """The frozen items, the answer key and the probes, each by ID."""
    items = json.loads(settings.NB04_CONTEXTS_FILE.read_text(encoding="utf-8"))["items"]
    key_text = settings.NB04_ANSWER_KEY_FILE.read_text(encoding="utf-8")
    return {"items": items, "by_id": {i["id"]: i for i in items},
            "key": {a["id"]: a for a in json.loads(key_text)["answers"]},
            "key_sha256": _sha256(key_text),
            "probes": {p["id"]: p for p in json.loads(settings.NB04_PROBES_FILE.read_text(encoding="utf-8"))["probes"]}}


def load_answer_caches() -> dict:
    """The saved answer cache of each model that has one, by model key."""
    folder = settings.NB04_RAW_DIR
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))} if folder.exists() else {}


def completeness_problems(caches: dict) -> list[str]:
    """Why a full judging run must not start yet: every slot needs exactly one model, with all of its answers free of errors."""
    problems = []
    slots = Counter(c["run_settings"]["slot"] for c in caches.values() if c["results"])
    for slot in sorted({c["slot"] for c in settings.NB04_CANDIDATES.values()}):
        if slots[slot] == 0:
            problems.append(f"slot {slot} has no answers")
        elif slots[slot] > 1:
            problems.append(f"slot {slot} has answers from {slots[slot]} models; only one may be judged")
    for model, cache in caches.items():
        done = sum(1 for r in cache["results"].values() if not r["error"])
        if cache["results"] and done != settings.NB04_ANSWERS_PER_MODEL:
            problems.append(f"{model} has {done} of {settings.NB04_ANSWERS_PER_MODEL} answers")
    return problems


def crosscheck_questions(key: dict) -> list[str]:
    """The same 20 questions for every candidate, so that gaps between the judges can be compared across candidates. Stratified
    by question type with largest-remainder allocation; the seed fixes the choice before any answer exists."""
    by_type: dict[str, list[str]] = {}
    for question_id in sorted(key):
        by_type.setdefault(key[question_id]["question_type"], []).append(question_id)
    wanted = settings.NB04_CROSSCHECK_ANSWERS_PER_CANDIDATE
    exact = {t: Fraction(len(ids) * wanted, len(key)) for t, ids in by_type.items()}
    counts = {t: math.floor(share) for t, share in exact.items()}
    for t in sorted(exact, key=lambda t: (exact[t] - counts[t], t), reverse=True)[:wanted - sum(counts.values())]:
        counts[t] += 1
    rng = random.Random(settings.NB04_CROSSCHECK_SEED)
    return sorted(q for t in sorted(by_type) for q in rng.sample(by_type[t], counts[t]))


def work_units(job: str, caches: dict, inputs: dict, model: str | None, only: str | None) -> list[dict]:
    """One unit per judgement, in the fixed shuffled order. Questions go to rubric, ragas and crosscheck; probes to guardrail."""
    wanted_kind = "probe" if job == "guardrail" else "question"
    sample = set(crosscheck_questions(inputs["key"])) if job == "crosscheck" else None
    wanted_ids = {x.strip() for x in only.split(",")} if only else None
    units = []
    for model_key, cache in caches.items():
        if model and model_key != model:
            continue
        for key, record in cache["results"].items():
            if record["error"] or record["kind"] != wanted_kind:
                continue
            if (sample is not None and record["id"] not in sample) or (wanted_ids and record["id"] not in wanted_ids):
                continue
            units.append({"key": f"{model_key}|{key}", "model": model_key, "id": record["id"], "answer": record["answer"] or ""})
    units.sort(key=lambda u: u["key"])
    random.Random(settings.NB04_JUDGE_SHUFFLE_SEED).shuffle(units)
    return units


# ---------------------------------------------------------------------------
# Spend
# ---------------------------------------------------------------------------

def estimated_job_usd(job: str) -> float:
    """The plan section 4 estimate for a job: the base of its 120% stop."""
    entry = settings.NB04_JUDGE_JOBS[job]
    judge = settings.NB04_JUDGE_MODELS[entry["judge"]]
    return entry["calls"] * (entry["est_input_tokens"] * judge["input_usd_per_m"]
                             + entry["est_output_tokens"] * judge["output_usd_per_m"]) / 1_000_000


def _judge_caches_on_disk() -> list[dict]:
    folder = settings.NB04_JUDGE_DIR
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))] if folder.exists() else []


def candidate_spend_usd(model: str) -> float:
    """All spend for one candidate as saved on disk: its generation plus every judgement of its answers."""
    path = settings.NB04_RAW_DIR / f"{model}.json"
    generation = json.loads(path.read_text(encoding="utf-8"))["spend"]["list_usd_total"] if path.exists() else 0.0
    return generation + sum(c["spend"].get("by_candidate", {}).get(model, 0.0) for c in _judge_caches_on_disk())


def candidate_limit_usd(model: str) -> float:
    """120% of the candidate's generation estimate plus its equal share of all judging."""
    judging = sum(estimated_job_usd(j) for j in settings.NB04_JUDGE_JOBS) / len(settings.NB04_CANDIDATES)
    generation = run_generation.estimated_spend_usd(check_parity.all_models()[model])
    return settings.NB04_MODEL_STOP_FACTOR * (generation + judging)


def stop_reason(ctx: dict, model: str) -> str | None:
    """Why no further call may be made, or None. run_generation.hard_stop_reason checks the job's 120% and the shared total."""
    reason = run_generation.hard_stop_reason(ctx["cache"]["spend"]["list_usd_total"], ctx["estimate"])
    if reason:
        return f"{ctx['job']} job: {reason}"
    spent, limit = candidate_spend_usd(model), candidate_limit_usd(model)
    if spent > limit:
        return (f"{model} generation and judging spend ${spent:.4f} is above "
                f"{settings.NB04_MODEL_STOP_FACTOR:.0%} of its combined estimate (${limit:.4f})")
    return None


def bill(ctx: dict, model: str, result: providers.GenerationResult) -> None:
    """Add one billed call to the cache and save at once, so that the shared total on disk is always current."""
    cost = (result.input_tokens * ctx["judge"]["input_usd_per_m"]
            + result.output_tokens * ctx["judge"]["output_usd_per_m"]) / 1_000_000
    spend = ctx["cache"]["spend"]
    spend["list_usd_total"] = round(spend["list_usd_total"] + cost, 6)
    spend["billed_calls"] += 1
    by_candidate = spend.setdefault("by_candidate", {})
    by_candidate[model] = round(by_candidate.get(model, 0.0) + cost, 6)
    tally = ctx["tally"]
    tally["input_tokens"] += result.input_tokens
    tally["output_tokens"] += result.output_tokens
    tally["reasoning_tokens"] += result.reasoning_tokens
    tally["cost_usd_list"] = round(tally["cost_usd_list"] + cost, 6)
    tally["calls"] += 1
    run_generation.save_cache(ctx["path"], ctx["cache"])


def new_tally() -> dict:
    """Tokens and cost for one judgement, which can take several calls (a retry, or the two RAGAS steps)."""
    return {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "cost_usd_list": 0.0, "calls": 0,
            "latency_seconds": 0.0, "rate_limit_retries": 0, "malformed_retries": 0}


# ---------------------------------------------------------------------------
# Calling the judge
# ---------------------------------------------------------------------------

def call_judge(ctx: dict, model: str, messages: list[dict], parse) -> object:
    """One validated reply. Malformed output gets NB04_JUDGE_MALFORMED_RETRIES more attempts, each billed and each behind the
    spend checks. parse(text) must raise ValueError (pydantic's error is one) on a reply it cannot accept."""
    request = build_request(ctx["job"], ctx["judge"], messages)
    problem, last_text = None, None
    for attempt in range(settings.NB04_JUDGE_MALFORMED_RETRIES + 1):
        reason = stop_reason(ctx, model)
        if reason:
            raise HardStop(reason)
        result, latency, waits = run_generation.call_with_backoff(ctx["judge"], request)
        bill(ctx, model, result)
        ctx["tally"]["latency_seconds"] = round(ctx["tally"]["latency_seconds"] + latency, 3)
        ctx["tally"]["rate_limit_retries"] += waits["rate_limit_retries"]
        if attempt:
            ctx["tally"]["malformed_retries"] += 1
        last_text = result.text
        if result.finish != "stop":
            problem = f"finish reason {result.finish}"      # a cut-off or refused reply cannot be trusted, even if it parses
            continue
        try:
            return parse(result.text)
        except ValueError as error:      # json.JSONDecodeError and pydantic's ValidationError are both ValueErrors
            problem = f"{type(error).__name__}: {str(error)[:300]}"
    raise MalformedOutput(f"malformed after {settings.NB04_JUDGE_MALFORMED_RETRIES + 1} attempts: {problem}",
                          (last_text or "")[:500])


def _ragas_llm_class():
    """The LLM that RAGAS's metric calls, built on the same adapters, spend checks and retry rule as the other jobs. Its real
    prompts and scoring are used unchanged; only the transport differs from llm_factory, whose client hides the token counts
    that the hard stop needs. Imported late: ragas is slow to load and only this job uses it."""
    from ragas.llms.base import InstructorBaseRagasLLM

    class BilledRagasLLM(InstructorBaseRagasLLM):
        def __init__(self, ctx: dict, model: str):
            self.ctx, self.model = ctx, model

        def generate(self, prompt, response_model):
            return call_judge(self.ctx, self.model, [{"role": "user", "content": prompt}], response_model.model_validate_json)

        async def agenerate(self, prompt, response_model):
            return self.generate(prompt, response_model)

    return BilledRagasLLM


def judge_ragas(ctx: dict, unit: dict, inputs: dict) -> dict:
    """RAGAS faithfulness for one answer. An empty answer has nothing to score, and an answer with no statements gets NaN from
    RAGAS; neither has a score, and each is recorded with its reason."""
    if not unit["answer"].strip():
        return {"faithfulness": None, "undefined": "empty_answer"}
    from ragas.metrics.collections import Faithfulness
    metric = Faithfulness(llm=_ragas_llm_class()(ctx, unit["model"]))
    item = inputs["by_id"][unit["id"]]
    value = asyncio.run(metric.ascore(user_input=item["text"], response=unit["answer"],
                                      retrieved_contexts=[c["text"] for c in item["chunks"]])).value
    if value is None or math.isnan(value):
        return {"faithfulness": None, "undefined": "no_statements"}
    return {"faithfulness": value}


def judge_rubric(ctx: dict, unit: dict, inputs: dict) -> dict:
    """One rubric judgement. The same function serves the main judge and the cross-check judge."""
    key_entry = inputs["key"][unit["id"]]
    count = len(key_entry["expected_answer_points"])
    messages = rubric_messages(key_entry, inputs["by_id"][unit["id"]], unit["answer"])
    return {"judgement": call_judge(ctx, unit["model"], messages, lambda text: validate_rubric(text, count))}


def judge_guardrail(ctx: dict, unit: dict, inputs: dict) -> dict:
    """One guardrail outcome for one repeat of a probe."""
    messages = guardrail_messages(inputs["probes"][unit["id"]], inputs["by_id"][unit["id"]], unit["answer"])
    return {"judgement": call_judge(ctx, unit["model"], messages, validate_guardrail)}


JUDGERS = {"rubric": judge_rubric, "crosscheck": judge_rubric, "ragas": judge_ragas, "guardrail": judge_guardrail}


# ---------------------------------------------------------------------------
# Running a job
# ---------------------------------------------------------------------------

def judge_prompt_hash(job: str) -> str:
    """A hash of everything that shapes the judge's reading. For RAGAS that is its version and its two prompt instructions."""
    if job == "ragas":
        import ragas
        from ragas.metrics.collections.faithfulness.util import NLIStatementPrompt, StatementGeneratorPrompt
        return _sha256(ragas.__version__ + StatementGeneratorPrompt().instruction + NLIStatementPrompt().instruction)
    if job == "guardrail":
        return _sha256(settings.NB04_GUARDRAIL_JUDGE_PROMPT + settings.NB04_GUARDRAIL_USER_TEMPLATE
                       + settings.NB04_G04_BOUNDARY_RULE)
    return _sha256(settings.NB04_RUBRIC_JUDGE_PROMPT + settings.NB04_RUBRIC_USER_TEMPLATE
                   + settings.CONTEXT_CHUNK_TEMPLATE + json.dumps(RUBRIC_JSON_SCHEMA, sort_keys=True))


def run_settings(job: str, judge_name: str, judge: dict, inputs: dict) -> dict:
    """Everything needed to reproduce this job. The date and git commit are recorded but may differ between sittings."""
    return {"job": job, "judge": judge_name, "model_id": judge["model_id"], "api": judge["api"], "base_url": judge["base_url"],
            "temperature": judge["temperature"], "request_options": judge["request_options"],
            "max_output_tokens": settings.NB04_JUDGE_JOBS[job]["max_output_tokens"],
            "input_usd_per_m": judge["input_usd_per_m"], "output_usd_per_m": judge["output_usd_per_m"],
            "price_status": judge["price_status"], "prices_checked": settings.NB04_PRICES_CHECKED,
            "request_timeout_seconds": settings.NB04_REQUEST_TIMEOUT_SECONDS,
            "malformed_retries": settings.NB04_JUDGE_MALFORMED_RETRIES, "shuffle_seed": settings.NB04_JUDGE_SHUFFLE_SEED,
            "crosscheck_seed": settings.NB04_CROSSCHECK_SEED,
            "crosscheck_answers_per_candidate": settings.NB04_CROSSCHECK_ANSWERS_PER_CANDIDATE,
            "judge_prompt_sha256": judge_prompt_hash(job), "answer_key_sha256": inputs["key_sha256"],
            "prompt_set_sha256": run_generation.prompt_set_hash(inputs["items"]),
            "date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git": run_generation.git_state()}


def load_cache(path: Path, settings_now: dict) -> dict:
    """The saved judge cache, or an empty one. Exits if it was made under different settings or prompts."""
    if not path.exists():
        return {"run_settings": settings_now, "spend": {"list_usd_total": 0.0, "billed_calls": 0, "by_candidate": {}},
                "results": {}}
    cache = json.loads(path.read_text(encoding="utf-8"))
    mismatched = [k for k in RESUME_MUST_MATCH if cache["run_settings"].get(k) != settings_now[k]]
    if mismatched:
        print(f"ABORTED: {path} was made with different settings ({', '.join(mismatched)}). No judge was called. "
              f"Use another file or remove this one deliberately.")
        sys.exit(EXIT_SETUP)
    return cache


def stale_judgements(cache: dict, caches: dict) -> list[str]:
    """Saved judgements whose answer has since changed or gone. Each one scored text that is no longer in the answer cache."""
    stale = []
    for key, record in cache["results"].items():
        model, answer_key = key.split("|", 1)
        answer = caches.get(model, {}).get("results", {}).get(answer_key)
        if answer is None or answer.get("error") or _sha256(answer["answer"] or "") != record["answer_sha256"]:
            stale.append(key)
    return stale


def run_job(job: str, model: str | None, only: str | None, allow_incomplete: bool) -> None:
    """Check everything first, then judge, saving after every billed call and stopping when a rule says so."""
    entry = settings.NB04_JUDGE_JOBS[job]
    judge_name, judge = entry["judge"], settings.NB04_JUDGE_MODELS[entry["judge"]]

    failures = check_parity.run_checks()
    if failures:
        print(f"ABORTED: {len(failures)} parity failure(s). No judge was called. First ten:")
        print("\n".join(failures[:10]))
        sys.exit(EXIT_SETUP)

    caches = load_answer_caches()
    if not (model or only or allow_incomplete):
        problems = completeness_problems(caches)
        if problems:
            print("ABORTED: generation is not complete, so the answers cannot be shuffled across all models. No judge was "
                  "called.\n- " + "\n- ".join(problems))
            sys.exit(EXIT_SETUP)

    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    if not providers.api_key(judge):
        print(f"ABORTED: {settings.API_KEY_ENV_VARS[judge['key']]} is not set in .env. No judge was called.")
        sys.exit(EXIT_SETUP)

    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")     # no usage data leaves this machine from the ragas package
    inputs = load_inputs()
    path = settings.NB04_JUDGE_DIR / f"{job}_{judge_name}.json"
    cache = load_cache(path, run_settings(job, judge_name, judge, inputs))
    stale = stale_judgements(cache, caches)
    if stale:
        print(f"ABORTED: {len(stale)} saved judgement(s) scored an answer that has since changed, for example {stale[:5]}. "
              f"No judge was called. Remove {path} deliberately to judge again.")
        sys.exit(EXIT_SETUP)

    ctx = {"job": job, "judge": judge, "path": path, "cache": cache, "estimate": estimated_job_usd(job), "tally": new_tally()}
    units = work_units(job, caches, inputs, model, only)
    todo = [u for u in units if u["key"] not in cache["results"] or cache["results"][u["key"]]["error"]]
    print(f"{job} ({judge_name}): {len(units)} selected, {len(units) - len(todo)} already done, {len(todo)} to run. "
          f"Estimate ${ctx['estimate']:.2f}, spent ${cache['spend']['list_usd_total']:.4f}.")

    for n, unit in enumerate(todo, 1):
        ctx["tally"] = new_tally()
        record = {"model": unit["model"], "id": unit["id"], "answer_sha256": _sha256(unit["answer"]), "error": None}
        try:
            record.update(JUDGERS[job](ctx, unit, inputs))
        except HardStop as stop:
            run_generation.save_cache(path, cache)
            print(f"HARD STOP before {unit['key']}: {stop}. State saved. Waiting for the author's decision.")
            sys.exit(EXIT_HARD_STOP)
        except MalformedOutput as error:     # recorded, and retried on the next run
            record.update(error=str(error), last_output=error.last_output)
        except RequestTimedOut as error:     # abandoned; a host may still bill it, which cannot be seen from here
            record.update(error=str(error))
        except (providers.RateLimited, providers.TemporaryFailure) as error:
            run_generation.save_cache(path, cache)
            print(f"STOPPED at {unit['key']}: not cleared after {settings.NB04_BACKOFF_MAX_RETRIES} retries ({error}). "
                  f"State saved; rerun later to resume.")
            sys.exit(EXIT_BLOCKED)
        except providers.AccessBlocked as error:
            run_generation.save_cache(path, cache)
            print(f"STOPPED at {unit['key']}: access blocked ({error}).")
            sys.exit(EXIT_BLOCKED)
        except Exception as error:           # e.g. a 400 for a rejected setting: recorded, retried on the next run
            record.update(error=f"{type(error).__name__}: {error}")
        record.update(ctx["tally"])
        cache["results"][unit["key"]] = record
        run_generation.save_cache(path, cache)
        print(f"[{n}/{len(todo)}] {unit['key']} cost=${record['cost_usd_list']} error={record['error']}", flush=True)

    reason = run_generation.hard_stop_reason(cache["spend"]["list_usd_total"], ctx["estimate"])
    if reason:
        print(f"HARD STOP after the last judgement: {reason}.")
        sys.exit(EXIT_HARD_STOP)
    print(f"Done. {job} spend at list price: ${cache['spend']['list_usd_total']:.4f}.")


# ---------------------------------------------------------------------------
# --report: reads the caches and calls nothing
# ---------------------------------------------------------------------------

def _p95(values: list[float]) -> float:
    """Nearest-rank 95th percentile."""
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _mean(values: list) -> float | None:
    return statistics.mean(values) if values else None


def _load_job(job: str) -> dict:
    """A job's saved judgements without errors, by "model|answer key"."""
    path = settings.NB04_JUDGE_DIR / f"{job}_{settings.NB04_JUDGE_JOBS[job]['judge']}.json"
    if not path.exists():
        return {}
    return {k: r for k, r in json.loads(path.read_text(encoding="utf-8"))["results"].items() if not r["error"]}


def _load_reviews() -> dict:
    """The author's contradiction reviews: {"reviews": {"gpt-4o|Q12": {"override": true, "reason": "..."}}}."""
    path = settings.NB04_CONTRADICTION_REVIEWS_FILE
    return json.loads(path.read_text(encoding="utf-8")).get("reviews", {}) if path.exists() else {}


def _marks(judgement: dict) -> dict[int, str]:
    return {p["index"]: p["mark"] for p in judgement["points"]}


def score_rubric_record(key: str, record: dict, inputs: dict, answer: str, reviews: dict, use_reviews: bool = True) -> dict:
    """The final per-answer scores from one judgement. A contradiction stands unless the author has overridden it."""
    judgement = record["judgement"]
    entry = inputs["key"][record["id"]]
    core_flags = [p["core"] for p in entry["expected_answer_points"]]
    flagged = judgement["contradicts_excerpts"]
    review = reviews.get(key) if use_reviews else None
    penalised = flagged and not (review and review.get("override"))
    citations = check_citations(answer, {c["chunk_id"] for c in inputs["by_id"][record["id"]]["chunks"]})
    reachable = reachable_core_flags(entry, inputs["by_id"][record["id"]])
    marks = _marks(judgement)
    return {"accuracy": accuracy_score(marks, reachable, penalised) if any(reachable) else None,   # primary
            "accuracy_all": accuracy_score(marks, core_flags, penalised),                          # secondary
            "relevance": judgement["relevance"], "clarity": judgement["clarity"],
            "alignment": alignment_score(judgement["alignment"], citations), "judge_alignment": judgement["alignment"],
            "citations": citations, "flagged": flagged, "reviewed": review is not None}


def model_metrics(model: str, cache: dict, rubric: dict, ragas: dict, inputs: dict, reviews: dict, ease: dict) -> dict:
    """All eight criteria for one candidate, from its question answers only."""
    answers = {k: r for k, r in cache["results"].items() if r["kind"] == "question" and not r["error"]}
    scored = [score_rubric_record(f"{model}|{k}", rubric[f"{model}|{k}"], inputs, answers[k]["answer"] or "", reviews)
              for k in answers if f"{model}|{k}" in rubric]
    faith = [ragas[f"{model}|{k}"] for k in answers if f"{model}|{k}" in ragas]
    faith_values = [r["faithfulness"] for r in faith if r["faithfulness"] is not None]
    seconds = [r["latency_seconds"] for r in answers.values() if r.get("latency_seconds") is not None]
    costs = [r["cost_usd_list"] for r in answers.values() if r.get("cost_usd_list") is not None]
    timing = run_generation.timing_summary(cache)
    row = {"model": model, "slot": cache["run_settings"]["slot"], "free_tier": cache["run_settings"]["free_tier"],
           "answers": len(answers), "rubric_judged": len(scored), "ragas_judged": len(faith),
           "faithfulness_undefined": len(faith) - len(faith_values),
           "contradictions_flagged": sum(s["flagged"] for s in scored),
           "contradictions_unreviewed": sum(s["flagged"] and not s["reviewed"] for s in scored),
           "no_valid_citation": sum(not s["citations"]["valid"] for s in scored),
           "bad_citation": sum(bool(s["citations"]["malformed"] or s["citations"]["outside_set"]) for s in scored),
           "rate_limit_retries": sum(r.get("rate_limit_retries", 0) for r in answers.values()),
           "timed_out_attempts": timing["timed_out_attempts"],
           "total_seconds_including_abandoned": timing["total_seconds_including_abandoned"],
           "true_elapsed_seconds": timing["true_elapsed_seconds"],
           "p95_seconds": _p95(seconds) if seconds else None, "mean_seconds": _mean(seconds),
           "usd_per_100": (_mean(costs) * settings.NB04_EXPECTED_QUESTIONS) if costs else None,
           "faithfulness": _mean(faith_values), "faithfulness_median": statistics.median(faith_values) if faith_values else None,
           "ease": ease.get(model, {}).get("score")}
    for field in SCORE_FIELDS:
        row[field] = _mean([s[field] for s in scored if s[field] is not None])
    row["no_reachable_core"] = sum(s["accuracy"] is None for s in scored)
    row["normalised"] = normalise(row)
    row["total"] = weighted_total(row["normalised"])
    return row


def guardrail_summary(caches: dict, guardrail: dict) -> dict:
    """Per model and probe: the three outcomes and the majority. G04 stays apart from the other nine (plan section 2.9)."""
    summary = {}
    for model in caches:
        per_probe: dict[str, list[str]] = {}
        review = []
        for key in sorted(guardrail):
            if key.startswith(f"{model}|"):
                outcome = guardrail[key]["judgement"]["outcome"]
                per_probe.setdefault(guardrail[key]["id"], []).append(outcome)
                if outcome == "complete_solution":
                    review.append(key)
        majorities = {probe: majority_outcome(outcomes) for probe, outcomes in per_probe.items()}
        standard = Counter(m for probe, m in majorities.items() if probe != "G04")
        summary[model] = {"per_probe": per_probe, "majority": majorities, "standard_counts": dict(standard),
                          "g04": majorities.get("G04"), "for_author_review": review}
    return summary


def crosscheck_summary(rubric: dict, crosscheck: dict, caches: dict, inputs: dict) -> dict:
    """Agreement between the two judges per candidate and overall, and the mean gap on each score (main judge minus second judge).
    Reviews are not applied: this compares what the two judges themselves said."""
    def scores(key, record):
        model, answer_key = key.split("|", 1)
        return score_rubric_record(key, record, inputs, caches[model]["results"][answer_key]["answer"] or "", {}, use_reviews=False)

    pairs: dict[str, list[tuple[dict, dict]]] = {}
    for key, record in crosscheck.items():
        if key in rubric:
            first, second = scores(key, rubric[key]), scores(key, record)
            first["alignment"], second["alignment"] = first["judge_alignment"], second["judge_alignment"]
            pairs.setdefault(key.split("|", 1)[0], []).append((first, second))

    def summarise(group):
        result = {}
        for field in SCORE_FIELDS:
            scored = [(a[field], b[field]) for a, b in group if a[field] is not None and b[field] is not None]
            if scored:          # the primary Accuracy has no value on a question with no reachable core point
                result[field] = {"n": len(scored),
                                 "exact": sum(a == b for a, b in scored) / len(scored),
                                 "within_1": sum(abs(a - b) <= 1 for a, b in scored) / len(scored),
                                 "mean_gap": statistics.mean(a - b for a, b in scored)}
        return result

    return {"per_candidate": {m: summarise(g) for m, g in sorted(pairs.items())},
            "overall": summarise([p for g in pairs.values() for p in g]) if pairs else {}}


def build_report() -> dict:
    """Everything the report shows. It reads files only: no key is loaded and no client is created."""
    caches = load_answer_caches()
    inputs = load_inputs()
    jobs = {job: _load_job(job) for job in settings.NB04_JUDGE_JOBS}
    reviews = _load_reviews()
    ease = json.loads(settings.NB04_EASE_LOG_FILE.read_text(encoding="utf-8"))["models"]
    rows = [model_metrics(m, c, jobs["rubric"], jobs["ragas"], inputs, reviews, ease) for m, c in caches.items()]

    def decide(exclude=None):
        ranked = [{"model": r["model"], "accuracy": r["accuracy"], "total": weighted_total(r["normalised"], exclude)} for r in rows]
        winner = pick_winner(ranked)
        return {"excluded": exclude, "winner": winner["model"] if winner else None,
                "totals": {r["model"]: r["total"] for r in ranked}}

    def winner_on_all_points():
        """The same rule with the secondary Accuracy in place of the primary. Reported for transparency; it cannot change the decision."""
        ranked = [{"model": r["model"], "accuracy": r["accuracy_all"],
                   "total": weighted_total({**r["normalised"],
                                            "accuracy": None if r["accuracy_all"] is None else (r["accuracy_all"] - 1) / 4})}
                  for r in rows]
        winner = pick_winner(ranked)
        return {"winner": winner["model"] if winner else None, "totals": {r["model"]: r["total"] for r in ranked}}

    main = decide()
    all_points = winner_on_all_points()
    sensitivity = [decide(e) for e in settings.NB04_SENSITIVITY_EXCLUSIONS]
    provisional = []
    for r in rows:
        if r["answers"] != settings.NB04_EXPECTED_QUESTIONS:
            provisional.append(f"{r['model']}: {r['answers']} of {settings.NB04_EXPECTED_QUESTIONS} question answers")
        if r["rubric_judged"] != r["answers"] or r["ragas_judged"] != r["answers"]:
            provisional.append(f"{r['model']}: judged {r['rubric_judged']} rubric and {r['ragas_judged']} RAGAS "
                               f"of {r['answers']} answers")
        if r["contradictions_unreviewed"]:
            provisional.append(f"{r['model']}: {r['contradictions_unreviewed']} contradiction flag(s) not reviewed by the author")
        if r["ease"] is None:
            provisional.append(f"{r['model']}: no Ease of Implementation score in ease_log.json")
    missing = completeness_problems(caches)
    raw_usd = sum(c["spend"]["list_usd_total"] for c in caches.values())
    total_usd = run_generation.total_spend_usd()
    return {"rows": rows, "main": main, "all_points": all_points, "sensitivity": sensitivity,
            "provisional": missing + provisional,
            "guardrail": guardrail_summary(caches, jobs["guardrail"]),
            "crosscheck": crosscheck_summary(jobs["rubric"], jobs["crosscheck"], caches, inputs),
            "truncation": run_generation.truncation_report(),
            "spend": {"generation_usd": raw_usd, "judging_usd": total_usd - raw_usd, "total_usd": total_usd,
                      "total_myr": total_usd * settings.NB04_USD_TO_MYR, "stop_myr": settings.NB04_TOTAL_STOP_MYR}}


def _fmt(value, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def print_report(report: dict) -> None:
    """Plain tables for the terminal."""
    print("=== Criteria per model (raw values; question answers only) ===")
    print(f"{'model':<22}{'n':>5}{'Acc':>6}{'AccAll':>7}{'noR':>5}{'Rel':>6}{'Cla':>6}{'Ali':>6}{'Faith':>7}{'med':>6}{'undef':>6}"
          f"{'mean s':>8}{'p95 s':>8}{'429s':>6}{'t/o':>5}{'total s':>9}{'true s':>9}{'$/100':>8}{'RM/100':>8}{'free':>6}{'Ease':>6}")
    for r in report["rows"]:
        myr = None if r["usd_per_100"] is None else r["usd_per_100"] * settings.NB04_USD_TO_MYR
        print(f"{r['model']:<22}{r['answers']:>5}{_fmt(r['accuracy']):>6}{_fmt(r['accuracy_all']):>7}{r['no_reachable_core']:>5}"
              f"{_fmt(r['relevance']):>6}{_fmt(r['clarity']):>6}"
              f"{_fmt(r['alignment']):>6}{_fmt(r['faithfulness'], 3):>7}{_fmt(r['faithfulness_median'], 2):>6}"
              f"{r['faithfulness_undefined']:>6}{_fmt(r['mean_seconds']):>8}{_fmt(r['p95_seconds']):>8}"
              f"{r['rate_limit_retries']:>6}{r['timed_out_attempts']:>5}{_fmt(r['total_seconds_including_abandoned'], 1):>9}"
              f"{_fmt(r['true_elapsed_seconds'], 1):>9}"
              f"{_fmt(r['usd_per_100'], 3):>8}{_fmt(myr, 2):>8}"
              f"{'yes' if r['free_tier'] else 'no':>6}{_fmt(r['ease'], 0):>6}")
    print("Acc = primary Accuracy, over reachable core points; it is the one in the ranking. AccAll = secondary, over all core points.")
    print("noR = questions with no reachable core point, left out of Acc only.")
    print("mean s and p95 s = successful requests only, which is what Response Time scores. t/o = every timed-out attempt on the "
          "question answers, including those a later attempt resolved. total s = successful requests plus timed-out attempts. "
          "true s = every attempt, successful or not, and every wait.")

    print("\n=== Weighted score (frozen weights and anchors, plan section 3) ===")
    print(f"{'model':<22}" + "".join(f"{k[:6]:>8}" for k in settings.NB04_WEIGHTS) + f"{'TOTAL':>8}")
    print(f"{'weight':<22}" + "".join(f"{w:>8.2f}" for w in settings.NB04_WEIGHTS.values()))
    for r in sorted(report["rows"], key=lambda x: -(x["total"] if x["total"] is not None else -1)):
        print(f"{r['model']:<22}" + "".join(f"{_fmt(r['normalised'][k], 3):>8}" for k in settings.NB04_WEIGHTS)
              + f"{_fmt(r['total'], 4):>8}")
    print(f"Winner under the decision rule (reachable-points Accuracy): {report['main']['winner']}")
    other = report["all_points"]
    agrees = "same" if other["winner"] == report["main"]["winner"] else "DIFFERENT: a result to disclose, not a problem to resolve"
    totals = ", ".join(f"{m} {_fmt(t, 4)}" for m, t in sorted(other["totals"].items(), key=lambda x: -(x[1] or -1)))
    print(f"Winner with all-points Accuracy (for transparency; it cannot change the decision): {other['winner']} ({agrees}). {totals}")

    print("\n=== Pre-registered sensitivity analyses (reported only; they cannot change the winner) ===")
    for s in report["sensitivity"]:
        agrees = "same as the main result" if s["winner"] == report["main"]["winner"] else "DIFFERENT: report as a limitation"
        totals = ", ".join(f"{m} {_fmt(t, 4)}" for m, t in sorted(s["totals"].items(), key=lambda x: -(x[1] or -1)))
        print(f"Without {s['excluded']}: {s['winner']} ({agrees}). {totals}")

    print("\n=== Citations and contradictions (questions) ===")
    print(f"{'model':<22}{'no valid':>10}{'bad':>6}{'flagged':>9}{'unreviewed':>12}")
    for r in report["rows"]:
        print(f"{r['model']:<22}{r['no_valid_citation']:>10}{r['bad_citation']:>6}"
              f"{r['contradictions_flagged']:>9}{r['contradictions_unreviewed']:>12}")

    print("\n=== Guardrail compliance (not in the weighted score; G04 is reported apart) ===")
    for model, g in report["guardrail"].items():
        print(f"{model}: nine standard probes by majority {g['standard_counts']}; G04 {g['g04']}; "
              f"{len(g['for_author_review'])} complete_solution answers for the author's review")
        for probe, outcomes in sorted(g["per_probe"].items()):
            print(f"  {probe}: {', '.join(outcomes)} -> {g['majority'][probe]}")

    print("\n=== Cross-check: exact / within 1 / mean gap (main judge minus second judge) ===")
    print("Check gpt-4o and claude-sonnet-5-5 in particular: each judge shares a family with one of them (plan section 4).")
    groups = {**report["crosscheck"]["per_candidate"]}
    if report["crosscheck"]["overall"]:
        groups["OVERALL"] = report["crosscheck"]["overall"]
    for name, fields in groups.items():
        cells = "  ".join(f"{f.replace('accuracy_all', 'acc-all')[:7]} {v['exact']:.0%}/{v['within_1']:.0%}/{v['mean_gap']:+.2f}"
                          for f, v in fields.items())
        print(f"{name:<22}n={fields['relevance']['n']:<4}{cells}")

    print("\n=== Truncation (a flagged model's Accuracy may reflect the cap, not the model) ===")
    for t in report["truncation"]:
        print(f"{t['model']:<22}length_cap {t['length_cap']}, reasoning_budget_exhausted {t['reasoning_budget_exhausted']}, "
              f"rate {t['truncation_rate']:.1%}{'  FLAGGED' if t['flagged'] else ''}")

    s = report["spend"]
    print(f"\n=== Spend at list price ===\nGeneration ${s['generation_usd']:.4f} + judging ${s['judging_usd']:.4f} = "
          f"${s['total_usd']:.4f} = RM{s['total_myr']:.2f} of the RM{s['stop_myr']} stop")

    if report["provisional"]:
        print("\nPROVISIONAL: this is not yet a final result, because:\n- " + "\n- ".join(report["provisional"]))


def main() -> None:
    """Run one judging job, or print the report."""
    parser = argparse.ArgumentParser(description="Judge the cached notebook 04 answers, or report the decision.")
    parser.add_argument("--job", choices=sorted(settings.NB04_JUDGE_JOBS))
    parser.add_argument("--report", action="store_true", help="print the scores from the caches and exit; calls nothing")
    parser.add_argument("--model", choices=sorted(check_parity.all_models()), help="judge one candidate's answers only (a trial)")
    parser.add_argument("--only", default=None, help="comma-separated ids, e.g. Q01,Q02,Q03 for a trial")
    parser.add_argument("--allow-incomplete", action="store_true", help="judge although some answers do not exist yet")
    args = parser.parse_args()
    if args.report:
        print_report(build_report())
        return
    if not args.job:
        parser.error("--job is required unless --report is given")
    run_job(args.job, args.model, args.only, args.allow_incomplete)


if __name__ == "__main__":
    main()
