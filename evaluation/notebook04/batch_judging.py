"""
Half-price judging for notebook 04 (author's STEP 3, 2026-11-27): PREPARED, NOT RUN. The four judge jobs of run_judging.py through the
providers' batch routes, the OpenAI Batch API (rubric, ragas, guardrail) and Anthropic Message Batches (crosscheck), at their official
batch list prices (half the standard rate, settings.NB04_JUDGE_BATCH_PRICES).

ONLY THE PRICING ROUTE CHANGES. Everything that decides a score is run_judging.py's own code, called unchanged: the work units and their
fixed shuffled order, the judge models, the request built by run_judging.build_request (prompts, temperature 0, reasoning effort, forced
JSON / schema, output limits), the validators, the one malformed-output retry, the RAGAS prompts, and the cache files and record format
that --report reads. A batch cache records pricing_route "batch" and the batch prices, so run_judging.py's resume lock (which includes the
prices) refuses to mix the two routes in one cache.

How it works. A batch cannot answer a call while the code waits, so each pass runs the normal judging code with run_judging.call_judge
replaced: a call is answered from the batch replies already received, and a call with no reply yet raises Deferred and is collected. After
a pass, the collected requests are submitted as one batch; when it has ended its replies are stored and billed, and the next pass reuses
them. Rubric, cross-check and guardrail judgements need one round; RAGAS needs two, because its second prompt (NLI on the statements)
depends on the first reply (the statements). A malformed reply is resubmitted once, as run_judging.py does; a request the batch itself
returns as errored or expired is resubmitted up to NB04_BATCH_TRANSPORT_RETRIES times.

Spend: every reply is billed at the batch price into the same judge cache, so the shared total and the RM96 stop see it. Before each
submission the batch's projected cost (the job's per-request estimate at batch price) is checked against the job's 120% stop (at batch
price), the shared total and each candidate's combined limit; a projected overrun stops the run before anything is sent. Submitted batch
IDs are saved before waiting, so an interrupted run resumes the same batch instead of submitting (and paying for) it again.

Credit (author, 2026-11-28). Before each submission the batch must also fit the provider's prepaid credit as kept by credit_ledger.py
(stated credit minus every recorded cost, minus a reservation for batches still running on that provider). Each request reserves the
larger of the job's per-request estimate and the mean reply cost seen so far, times NB04_BATCH_CREDIT_MARGIN; a RAGAS statements
request also reserves its NLI request. If only part of the batch fits, that part is sent (in the fixed unit order) and the rest waits;
when nothing more fits, or a provider replies that the credit is used up, the run stops with exit 5 and prints how much to top up.
--credit prints the ledger and the top-up needed for the remaining judging, and calls nothing.

Usage (from the project root; needs the judge's key in .env):
    venv\\Scripts\\python evaluation\\notebook04\\batch_judging.py --job rubric [--wait] [--model gpt-4o] [--only Q01,Q02]
    venv\\Scripts\\python evaluation\\notebook04\\batch_judging.py --dry-run     # 2 cached answers, fake batch clients, temp folders
    venv\\Scripts\\python evaluation\\notebook04\\batch_judging.py --credit      # ledger and top-up needed; calls nothing
Exit codes: 0 finished, 1 setup failure, 2 hard stop, 4 a batch is still running (rerun later, or use --wait),
5 the prepaid credit does not cover the next request (top up, then rerun the same command).
"""

import argparse
import hashlib
import json
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import check_parity  # noqa: E402
import credit_ledger  # noqa: E402
import providers  # noqa: E402
import run_generation  # noqa: E402
import run_judging as rj  # noqa: E402
from config import settings  # noqa: E402

EXIT_SETUP, EXIT_HARD_STOP, EXIT_PENDING, EXIT_CREDIT = 1, 2, 4, 5
RAGAS_STEPS = 2              # statements, then NLI: the requests one RAGAS judgement makes
OPENAI_ENDPOINT = "/v1/chat/completions"
BATCH_ERROR = "__batch_error__"   # key in a poll's replies when the provider rejected the whole batch


class Deferred(Exception):
    """A judge call with no batch reply yet. request is None when the call is already in a submitted batch."""
    def __init__(self, custom_id: str, request: dict | None):
        super().__init__(custom_id)
        self.custom_id, self.request = custom_id, request


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def request_hash(unit_key: str, request: dict) -> str:
    text = unit_key + "|" + json.dumps(request, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def custom_id(unit_key: str, request: dict, attempt: int) -> str:
    """Within both providers' limits (Anthropic: 1-64 characters of [a-zA-Z0-9_-]). The unit key is part of it, so every judgement
    gets its own call, as on the standard route, even when two answers (e.g. three repeats of a short refusal) are identical."""
    return f"{request_hash(unit_key, request)[:40]}-a{attempt}"


def batch_judge(judge_name: str) -> dict:
    """The judge's settings with the batch list prices: what gets billed and what the run settings record."""
    batch_in, batch_out = settings.NB04_JUDGE_BATCH_PRICES[judge_name]
    return {**settings.NB04_JUDGE_MODELS[judge_name], "input_usd_per_m": batch_in, "output_usd_per_m": batch_out,
            "price_status": "official (batch)"}


def batch_estimate_usd(job: str) -> float:
    """The plan section 4 estimate for the job at batch price: the base of its 120% stop on this route."""
    entry = settings.NB04_JUDGE_JOBS[job]
    batch_in, batch_out = settings.NB04_JUDGE_BATCH_PRICES[entry["judge"]]
    return entry["calls"] * (entry["est_input_tokens"] * batch_in + entry["est_output_tokens"] * batch_out) / 1_000_000


def per_request_estimate_usd(job: str) -> float:
    """Projected batch-price cost of one request: a RAGAS judgement is two requests."""
    return batch_estimate_usd(job) / settings.NB04_JUDGE_JOBS[job]["calls"] / (RAGAS_STEPS if job == "ragas" else 1)


# ---------------------------------------------------------------------------
# Batch state (submitted batches and their replies), saved atomically
# ---------------------------------------------------------------------------

def state_path(job: str, judge_name: str) -> Path:
    return settings.NB04_BATCH_DIR / f"{job}_{judge_name}.json"


def load_state(job: str, judge_name: str) -> dict:
    path = state_path(job, judge_name)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"job": job, "judge": judge_name, "batches": [], "replies": {}, "transport_failures": {}, "submitted": {}}


def save_state(job: str, judge_name: str, state: dict) -> None:
    run_generation.save_cache(state_path(job, judge_name), state)


def pending_batches(state: dict) -> list[dict]:
    return [b for b in state["batches"] if not b.get("ingested")]


# ---------------------------------------------------------------------------
# Provider batch adapters (the only network code here)
# ---------------------------------------------------------------------------

def _finish(word: str | None) -> str:
    return {"stop": "stop", "end_turn": "stop", "length": "length", "max_tokens": "length", "content_filter": "refusal",
            "refusal": "refusal"}.get(word, word or "unknown")


def openai_reply(line: dict) -> tuple[str, dict]:
    """One line of an OpenAI batch output or error file -> (custom_id, reply)."""
    response = line.get("response") or {}
    body = response.get("body") or {}
    if line.get("error") or response.get("status_code") != 200 or not body.get("choices"):
        return line["custom_id"], {"error": json.dumps(line.get("error") or body.get("error") or {"status": response.get("status_code")})[:500]}
    choice, usage = body["choices"][0], body.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return line["custom_id"], {"text": choice["message"].get("content") or "", "finish": _finish(choice.get("finish_reason")),
                               "input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0),
                               "reasoning_tokens": details.get("reasoning_tokens", 0) or 0, "served_model": body.get("model"),
                               "error": None}


def anthropic_reply(entry: dict) -> tuple[str, dict]:
    """One Anthropic batch result (as a dict) -> (custom_id, reply)."""
    result = entry.get("result") or {}
    if result.get("type") != "succeeded":
        return entry["custom_id"], {"error": json.dumps({"type": result.get("type"), "error": result.get("error")}, default=str)[:500]}
    message = result["message"]
    text = "".join(block.get("text", "") for block in message.get("content", []) if block.get("type") == "text")
    usage = message.get("usage") or {}
    return entry["custom_id"], {"text": text, "finish": _finish(message.get("stop_reason")),
                                "input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0),
                                "reasoning_tokens": 0, "served_model": message.get("model"), "error": None}


class OpenAIBatches:
    """OpenAI Batch API: a JSONL file of chat-completion requests, a 24-hour window, results in an output and an error file."""
    provider = "openai"

    def __init__(self, client):
        self.client = client

    def submit(self, requests: dict[str, dict]) -> str:
        lines = "\n".join(json.dumps({"custom_id": cid, "method": "POST", "url": OPENAI_ENDPOINT, "body": body}, ensure_ascii=False)
                          for cid, body in requests.items())
        uploaded = self.client.files.create(file=("judging_batch.jsonl", lines.encode("utf-8")), purpose="batch")
        batch = self.client.batches.create(input_file_id=uploaded.id, endpoint=OPENAI_ENDPOINT,
                                           completion_window=settings.NB04_BATCH_COMPLETION_WINDOW)
        return batch.id

    def poll(self, batch_id: str) -> tuple[str, dict | None]:
        batch = self.client.batches.retrieve(batch_id)
        if batch.status not in ("completed", "failed", "expired", "cancelled"):
            return batch.status, None
        replies = {}
        errors = getattr(getattr(batch, "errors", None), "data", None) or []
        if batch.status == "failed" and errors:      # rejected as a whole (e.g. the enqueued-token limit): no request ran
            replies[BATCH_ERROR] = "; ".join(f"{getattr(e, 'code', '')}: {getattr(e, 'message', '')}" for e in errors)[:500]
        for file_id in (batch.output_file_id, batch.error_file_id):
            if file_id:
                for raw in self.client.files.content(file_id).text.splitlines():
                    if raw.strip():
                        cid, reply = openai_reply(json.loads(raw))
                        replies[cid] = reply
        return batch.status, replies


class AnthropicBatches:
    """Anthropic Message Batches: up to 24 hours; each result is succeeded, errored, canceled or expired."""
    provider = "anthropic"

    def __init__(self, client):
        self.client = client

    def submit(self, requests: dict[str, dict]) -> str:
        return self.client.messages.batches.create(
            requests=[{"custom_id": cid, "params": body} for cid, body in requests.items()]).id

    def poll(self, batch_id: str) -> tuple[str, dict | None]:
        batch = self.client.messages.batches.retrieve(batch_id)
        if batch.processing_status != "ended":
            return batch.processing_status, None
        replies = {}
        for entry in self.client.messages.batches.results(batch_id):
            data = entry.model_dump() if hasattr(entry, "model_dump") else entry
            cid, reply = anthropic_reply(data)
            replies[cid] = reply
        return "ended", replies


def real_batches(judge: dict):
    """The provider's batch client for this judge (SDK retries on, for these control calls only)."""
    key = providers.api_key(judge)
    if judge["api"] == "anthropic":
        from anthropic import Anthropic
        return AnthropicBatches(Anthropic(api_key=key, max_retries=3, timeout=120))
    from openai import OpenAI
    return OpenAIBatches(OpenAI(api_key=key, base_url=judge["base_url"], max_retries=3, timeout=120))


# ---------------------------------------------------------------------------
# The batch version of run_judging.call_judge
# ---------------------------------------------------------------------------

def make_call_judge(state: dict):
    """run_judging.call_judge, answered from batch replies. Same request, same retry rule for malformed output, same errors."""
    def call_judge(ctx: dict, model: str, messages: list[dict], parse) -> object:
        request = rj.build_request(ctx["job"], ctx["judge"], messages)
        problem, last_text = None, None
        for attempt in range(1, settings.NB04_JUDGE_MALFORMED_RETRIES + 2):
            cid = custom_id(ctx["unit_key"], request, attempt)
            reply = state["replies"].get(cid)
            if reply is None:
                if state["transport_failures"].get(cid, 0) > settings.NB04_BATCH_TRANSPORT_RETRIES:
                    raise RuntimeError(f"batch returned no reply for {cid} after {settings.NB04_BATCH_TRANSPORT_RETRIES + 1} submissions")
                pending = any(cid in b["custom_ids"] for b in pending_batches(state))
                raise Deferred(cid, None if pending else request)
            tally = ctx["tally"]          # tokens and cost of every reply this judgement used (billed once, at ingestion)
            for field in ("input_tokens", "output_tokens", "reasoning_tokens"):
                tally[field] += reply[field]
            tally["cost_usd_list"] = round(tally["cost_usd_list"] + reply["cost_usd"], 6)
            tally["calls"] += 1
            if attempt > 1:
                tally["malformed_retries"] += 1
            last_text = reply["text"]
            if reply["finish"] != "stop":
                problem = f"finish reason {reply['finish']}"
                continue
            try:
                return parse(reply["text"])
            except ValueError as error:
                problem = f"{type(error).__name__}: {str(error)[:300]}"
        raise rj.MalformedOutput(f"malformed after {settings.NB04_JUDGE_MALFORMED_RETRIES + 1} attempts: {problem}",
                                 (last_text or "")[:500])
    return call_judge


def ingest(ctx: dict, state: dict, batches, job: str, judge_name: str) -> int:
    """Poll every submitted batch; store and bill the replies of each one that has ended. Returns how many batches are still running."""
    running = 0
    for batch in pending_batches(state):
        status, replies = batches.poll(batch["id"])
        batch["status"] = status
        if replies is None:
            running += 1
            continue
        if BATCH_ERROR in replies:
            # Rejected as a whole: nothing ran or was billed, so these are not transport failures. Too many queued tokens -> the
            # next batches are half the size (down to NB04_BATCH_MIN_REQUESTS); any other rejection stops the run.
            batch.update(ingested=True, ended_at=utc_now(), rejected=replies[BATCH_ERROR])
            text = replies[BATCH_ERROR].lower()
            sent = len(batch["custom_ids"])
            if ("token_limit" in text or "enqueued token" in text) and sent > settings.NB04_BATCH_MIN_REQUESTS:
                size = max(settings.NB04_BATCH_MIN_REQUESTS, sent // 2)
                state["max_requests"] = size
                state.setdefault("size_reductions", []).append({"at": utc_now(), "batch": batch["id"], "new_max_requests": size,
                                                                "reason": replies[BATCH_ERROR][:300]})
            else:
                state["batch_rejected"] = {"at": utc_now(), "batch": batch["id"], "error": replies[BATCH_ERROR][:300]}
            save_state(job, judge_name, state)
            continue
        for cid in batch["custom_ids"]:
            reply = replies.get(cid)
            if reply is not None and reply.get("error") and credit_exhausted(reply["error"]):
                # Credit used up: not a transport failure, so it is not resubmitted; the run stops and is resumed after a top-up.
                state["credit_exhausted"] = {"at": utc_now(), "batch": batch["id"], "error": reply["error"][:300]}
                continue
            if reply is None or reply.get("error"):          # errored or expired inside the batch: resubmitted next pass
                state["transport_failures"][cid] = state["transport_failures"].get(cid, 0) + 1
                continue
            cost = (reply["input_tokens"] * ctx["judge"]["input_usd_per_m"]
                    + reply["output_tokens"] * ctx["judge"]["output_usd_per_m"]) / 1_000_000
            reply["cost_usd"] = round(cost, 8)
            state["replies"][cid] = reply
            model = state["submitted"][cid]["model"]
            spend = ctx["cache"]["spend"]
            spend["list_usd_total"] = round(spend["list_usd_total"] + cost, 6)
            spend["billed_calls"] += 1
            spend.setdefault("by_candidate", {})[model] = round(spend["by_candidate"].get(model, 0.0) + cost, 6)
        batch.update(ingested=True, ended_at=utc_now())
        run_generation.save_cache(ctx["path"], ctx["cache"])
        save_state(job, judge_name, state)
    return running


def projected_stop(ctx: dict, requests: dict[str, dict], state: dict) -> str | None:
    """Why this batch must not be submitted: its projected cost would pass the job's 120% (at batch price), the shared total, or a
    candidate's combined limit."""
    per_request = per_request_estimate_usd(ctx["job"])
    projected = per_request * len(requests)
    limit = ctx["estimate"] * settings.NB04_MODEL_STOP_FACTOR
    if ctx["cache"]["spend"]["list_usd_total"] + projected > limit:
        return (f"{ctx['job']} job: spend ${ctx['cache']['spend']['list_usd_total']:.4f} + projected ${projected:.4f} would pass "
                f"{settings.NB04_MODEL_STOP_FACTOR:.0%} of its batch estimate (${limit:.4f})")
    total_myr = (run_generation.total_spend_usd() + projected) * settings.NB04_USD_TO_MYR
    if total_myr >= settings.NB04_TOTAL_STOP_MYR:
        return f"total spend with this batch would be RM{total_myr:.2f}, at or above the RM{settings.NB04_TOTAL_STOP_MYR} stop line"
    by_model = {}
    for cid in requests:
        model = state["submitted"].get(cid, {}).get("model")
        by_model[model] = by_model.get(model, 0.0) + per_request
    for model, extra in by_model.items():
        if model and rj.candidate_spend_usd(model) + extra > rj.candidate_limit_usd(model):
            return f"{model}: generation and judging spend with this batch would pass its combined limit"
    return None


# ---------------------------------------------------------------------------
# Prepaid credit: fit each batch to what the ledger says is left (credit_ledger.py)
# ---------------------------------------------------------------------------

def credit_exhausted(text: str) -> bool:
    """True when a provider's error says the prepaid credit is used up (resubmitting cannot help)."""
    lowered = (text or "").lower()
    return any(marker in lowered for marker in settings.NB04_CREDIT_EXHAUSTED_MARKERS)


def request_reserve_usd(job: str, state: dict) -> float:
    """What one request of this job reserves: the larger of the per-request estimate and the mean reply cost seen, times the margin."""
    per_request = per_request_estimate_usd(job)
    seen = [r["cost_usd"] for r in state["replies"].values() if r.get("cost_usd") is not None]
    if seen:
        per_request = max(per_request, sum(seen) / len(seen))
    return per_request * settings.NB04_BATCH_CREDIT_MARGIN


def requests_committed(job: str, cid: str, state: dict) -> int:
    """How many requests sending this one commits to: a RAGAS statements request (its unit has no reply yet) also brings its NLI
    request; every other request is one."""
    if job != "ragas":
        return 1
    unit = state["submitted"].get(cid, {}).get("unit")
    answered = {state["submitted"][c]["unit"] for c in state["replies"] if c in state["submitted"]}
    return 1 if unit in answered else RAGAS_STEPS


def running_reserve_usd(provider: str, skip: Path | None = None) -> float:
    """Credit held for batches still running on this provider, in every job's saved state except the one in skip."""
    total = 0.0
    folder = settings.NB04_BATCH_DIR
    for path in sorted(folder.glob("*.json")) if folder.exists() else []:
        if skip is not None and path.resolve() == skip.resolve():
            continue
        other = json.loads(path.read_text(encoding="utf-8"))
        if credit_ledger.provider_of(other["judge"]) != provider:
            continue
        waiting = sum(len(b["custom_ids"]) for b in pending_batches(other))
        total += waiting * request_reserve_usd(other["job"], other)
    return total


def fit_to_credit(job: str, judge_name: str, collected: dict[str, dict], state: dict) -> tuple[dict[str, dict], float | None]:
    """The leading part of collected (fixed unit order) whose reservation fits the provider's credit left, and that credit.
    Everything fits for a provider with no prepaid credit recorded."""
    provider = credit_ledger.provider_of(judge_name)
    left = credit_ledger.left_usd(provider, running_reserve_usd(provider, skip=state_path(job, judge_name)))
    if left is None:
        return collected, None
    per_request, chosen, used = request_reserve_usd(job, state), {}, 0.0
    for cid, request in collected.items():
        cost = per_request * requests_committed(job, cid, state)
        if used + cost > left:
            break
        chosen[cid], used = request, used + cost
    return chosen, left


def remaining_work(job: str) -> dict:
    """Judgements still to do for one job, and their expected cost at batch price: the mean cost per judgement recorded so far
    (once at least 10 exist), otherwise the plan estimate per judgement."""
    entry = settings.NB04_JUDGE_JOBS[job]
    path = settings.NB04_JUDGE_DIR / f"{job}_{entry['judge']}.json"
    results = json.loads(path.read_text(encoding="utf-8"))["results"] if path.exists() else {}
    units = rj.work_units(job, rj.load_answer_caches(), rj.load_inputs(), None, None)
    done = [results[u["key"]] for u in units if u["key"] in results and not results[u["key"]]["error"]]
    per_unit = batch_estimate_usd(job) / entry["calls"]
    costs = [r["cost_usd_list"] for r in done if r.get("cost_usd_list") is not None]
    if len(costs) >= 10:
        per_unit = sum(costs) / len(costs)
    remaining = max(0, len(units) - len(done))
    return {"job": job, "judge": entry["judge"], "provider": credit_ledger.provider_of(entry["judge"]), "units": len(units),
            "judged": len(done), "remaining": remaining, "per_judgement_usd": per_unit, "expected_usd": remaining * per_unit}


def top_up_report() -> str:
    """Ledger per provider and the top-up needed for every judgement not yet done: expected, and with the credit margin."""
    lines = []
    work = [remaining_work(job) for job in settings.NB04_JUDGE_JOBS]
    for provider in sorted({w["provider"] for w in work}):
        reserve = running_reserve_usd(provider)
        lines.append(credit_ledger.summary(provider, reserve))
        mine = [w for w in work if w["provider"] == provider]
        for w in mine:
            lines.append(f"  {w['job']} ({w['judge']}): {w['judged']} of {w['units']} judged, {w['remaining']} to do at about "
                         f"${w['per_judgement_usd']:.5f} each = ${w['expected_usd']:.4f}")
        expected = sum(w["expected_usd"] for w in mine)
        left = credit_ledger.left_usd(provider, reserve)
        if left is None:
            continue
        need, safe = max(0.0, expected - left), max(0.0, expected * settings.NB04_BATCH_CREDIT_MARGIN - left)
        lines.append(f"  remaining {provider} judging: expected ${expected:.4f}, with the {settings.NB04_BATCH_CREDIT_MARGIN:.0%} "
                     f"margin ${expected * settings.NB04_BATCH_CREDIT_MARGIN:.4f}; left ${left:.4f} -> top up "
                     f"${need:.2f} (expected) or ${safe:.2f} (with the margin)")
    return "\n".join(lines)


def credit_stop(job: str, judge_name: str, state: dict, reason: str) -> None:
    """Save the state, print why and how much to top up, and exit 5. Nothing more is sent."""
    save_state(job, judge_name, state)
    print(f"CREDIT STOP ({job}, {judge_name}): {reason}. Nothing more was sent.\n{top_up_report()}")
    sys.exit(EXIT_CREDIT)


def run_batch_job(job: str, model: str | None = None, only: str | None = None, allow_incomplete: bool = False,
                  wait: bool = False, batches=None) -> None:
    """run_judging.run_job on the batch route. batches: a provider adapter (tests and the dry run pass a fake one)."""
    entry = settings.NB04_JUDGE_JOBS[job]
    judge_name = entry["judge"]
    judge = batch_judge(judge_name)
    failures = check_parity.run_checks()
    if failures:
        print(f"ABORTED: {len(failures)} parity failure(s). No batch was submitted.\n" + "\n".join(failures[:10]))
        sys.exit(EXIT_SETUP)
    caches = rj.load_answer_caches()
    if not (model or only or allow_incomplete):
        problems = rj.completeness_problems(caches)
        if problems:
            print("ABORTED: generation is not complete. No batch was submitted.\n- " + "\n- ".join(problems))
            sys.exit(EXIT_SETUP)
    if batches is None:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
        if not providers.api_key(judge):
            print(f"ABORTED: {settings.API_KEY_ENV_VARS[judge['key']]} is not set in .env. No batch was submitted.")
            sys.exit(EXIT_SETUP)
        batches = real_batches(judge)
    inputs = rj.load_inputs()
    path = settings.NB04_JUDGE_DIR / f"{job}_{judge_name}.json"
    run_settings = {**rj.run_settings(job, judge_name, judge, inputs), "pricing_route": "batch",
                    "batch_provider": batches.provider,
                    "standard_prices": [settings.NB04_JUDGE_MODELS[judge_name]["input_usd_per_m"],
                                        settings.NB04_JUDGE_MODELS[judge_name]["output_usd_per_m"]]}
    cache = rj.load_cache(path, run_settings)
    stale = rj.stale_judgements(cache, caches)
    if stale:
        print(f"ABORTED: {len(stale)} saved judgement(s) scored an answer that has since changed, e.g. {stale[:5]}.")
        sys.exit(EXIT_SETUP)
    state = load_state(job, judge_name)
    if state.get("credit_exhausted"):        # from an earlier run; a new run (after a top-up) tries again, keeping the record
        state.setdefault("credit_exhausted_history", []).append(state.pop("credit_exhausted"))
    ctx = {"job": job, "judge": judge, "path": path, "cache": cache, "estimate": batch_estimate_usd(job), "tally": rj.new_tally()}
    units = rj.work_units(job, caches, inputs, model, only)
    with mock.patch.object(rj, "call_judge", make_call_judge(state)):
        while True:
            running = ingest(ctx, state, batches, job, judge_name)
            if state.get("batch_rejected"):
                rejected = state.pop("batch_rejected")
                state.setdefault("rejections", []).append(rejected)
                save_state(job, judge_name, state)
                print(f"STOPPED: the provider rejected batch {rejected['batch']}: {rejected['error']}. Nothing in it ran or was billed.")
                sys.exit(EXIT_SETUP)
            if state.get("credit_exhausted"):
                run_generation.save_cache(path, cache)        # the replies already billed are kept
                credit_stop(job, judge_name, state, f"the provider replied that the credit is used up "
                                                    f"({state['credit_exhausted']['error'][:120]})")
            todo =[u for u in units if u["key"] not in cache["results"] or cache["results"][u["key"]]["error"]]
            collected, done = {}, 0
            for unit in todo:
                ctx["tally"], ctx["unit_key"] = rj.new_tally(), unit["key"]
                record = {"model": unit["model"], "id": unit["id"], "answer_sha256": rj._sha256(unit["answer"]), "error": None,
                          "pricing_route": "batch"}
                try:
                    record.update(rj.JUDGERS[job](ctx, unit, inputs))
                except Deferred as deferred:
                    if deferred.request is not None:
                        collected[deferred.custom_id] = deferred.request
                        state["submitted"].setdefault(deferred.custom_id, {"model": unit["model"], "unit": unit["key"]})
                    continue
                except rj.MalformedOutput as error:
                    record.update(error=str(error), last_output=error.last_output)
                except Exception as error:                   # same as run_judging: recorded, retried on the next run
                    record.update(error=f"{type(error).__name__}: {error}")
                record.update(ctx["tally"])
                cache["results"][unit["key"]] = record
                done += 1
            run_generation.save_cache(path, cache)
            remaining = [u for u in units if u["key"] not in cache["results"] or cache["results"][u["key"]]["error"]]
            print(f"{job} (batch, {judge_name}): {len(units)} selected, {len(units) - len(remaining)} judged, "
                  f"{done} recorded this pass, {len(collected)} new requests, {running} batch(es) running. "
                  f"Spent ${cache['spend']['list_usd_total']:.4f} at batch price.", flush=True)
            if not collected and not running:
                errors = sum(1 for u in units if cache["results"].get(u["key"], {}).get("error"))
                print(f"Done. {job} spend at batch price: ${cache['spend']['list_usd_total']:.4f}; {errors} judgement(s) with an error.")
                return
            if collected and not running:
                # The spend limits first, on the whole batch as before; the credit can then only narrow what is sent.
                reason = projected_stop(ctx, collected, state)
                if reason:
                    save_state(job, judge_name, state)
                    print(f"HARD STOP before submitting {len(collected)} requests: {reason}. Nothing was sent.")
                    sys.exit(EXIT_HARD_STOP)
                size = state.get("max_requests", settings.NB04_BATCH_MAX_REQUESTS)
                if len(collected) > size:        # sent in parts, one after another, in the fixed unit order
                    collected = dict(list(collected.items())[:size])
                chosen, left = fit_to_credit(job, judge_name, collected, state)
                if not chosen:
                    credit_stop(job, judge_name, state, f"{len(collected)} request(s) are ready but the next one does not fit the "
                                                        f"${left:.4f} of credit left")
                if len(chosen) < len(collected):
                    print(f"CREDIT: {len(chosen)} of {len(collected)} requests fit the ${left:.4f} left; sending those, the rest wait.",
                          flush=True)
                collected = chosen
                try:
                    batch_id = batches.submit(collected)
                except Exception as error:
                    if credit_exhausted(str(error)):
                        credit_stop(job, judge_name, state, f"the provider refused the batch: {str(error)[:200]}")
                    raise
                state["batches"].append({"id": batch_id, "provider": batches.provider, "custom_ids": list(collected),
                                         "submitted_at": utc_now(), "status": "submitted", "ingested": False})
                save_state(job, judge_name, state)        # saved before waiting: an interrupted run resumes this batch
                print(f"Submitted batch {batch_id} with {len(collected)} requests.", flush=True)
            if not wait:
                print("A batch is running. Rerun this command later (or with --wait) to collect it.")
                sys.exit(EXIT_PENDING)
            time.sleep(settings.NB04_BATCH_POLL_SECONDS)


# ---------------------------------------------------------------------------
# Dry run: two cached answers, fake batch clients, temporary folders; nothing leaves the machine
# ---------------------------------------------------------------------------

class FakeBatches:
    """Stands in for a provider: records each submitted batch and answers it at once with replies made by respond(body)."""
    def __init__(self, provider: str, respond):
        self.provider, self.respond, self.submitted = provider, respond, []

    def submit(self, requests: dict[str, dict]) -> str:
        self.submitted.append(dict(requests))
        return f"fake_{self.provider}_batch_{len(self.submitted)}"

    def poll(self, batch_id: str) -> tuple[str, dict]:
        requests = self.submitted[int(batch_id.rsplit("_", 1)[1]) - 1]
        return "ended", {cid: {"text": self.respond(body), "finish": "stop", "input_tokens": 2000, "output_tokens": 200,
                               "reasoning_tokens": 0, "served_model": body["model"], "error": None} for cid, body in requests.items()}


def fake_judge_reply(body: dict) -> str:
    """A valid reply for any of the four jobs, shaped by the request it answers (the dry run never scores real quality)."""
    text = "\n".join(m["content"] if isinstance(m["content"], str) else "" for m in body["messages"])
    if "EXPECTED ANSWER POINTS:" in text:
        block = text.split("EXPECTED ANSWER POINTS:", 1)[1].split("REVIEW NOTE:", 1)[0]
        count = len(re.findall(r"^\[\d+\]", block, re.M))
        return json.dumps({"points": [{"index": n, "mark": "covered"} for n in range(1, count + 1)], "contradicts_excerpts": False,
                           "contradiction_quote": None, "relevance": 4, "clarity": 4, "alignment": 4,
                           "reasons": {"relevance": "dry run", "clarity": "dry run", "alignment": "dry run"}})
    if "REQUEST:" in text:
        return json.dumps({"outcome": "scaffolded", "solution_quote": None, "reason": "dry run"})
    if '"statements"' in text and "verdict" in text:
        return json.dumps({"statements": [{"statement": "dry run statement", "reason": "dry run", "verdict": 1}]})
    return json.dumps({"statements": ["dry run statement"]})


def dry_run() -> None:
    """Judge two cached answers (rubric through the OpenAI path, cross-check through the Anthropic path) with fake batch clients,
    in temporary judge folders, and show that the requests equal run_judging's and that billing uses the batch price."""
    caches = rj.load_answer_caches()
    picks = []
    for model_key in ("gpt-4o", "deepseek-flash"):
        cache = caches.get(model_key)
        if cache and not cache["results"].get("Q01", {}).get("error", True):
            picks.append((model_key, "Q01"))
    if len(picks) < 2:
        sys.exit("dry run needs cached gpt-4o and deepseek-flash answers to Q01")
    inputs = rj.load_inputs()
    with tempfile.TemporaryDirectory() as tmp:
        judge_dir = Path(tmp) / "judge"
        with mock.patch.object(settings, "NB04_JUDGE_DIR", judge_dir), mock.patch.object(settings, "NB04_BATCH_DIR", judge_dir / "batches"),                 mock.patch.object(settings, "NB04_BATCH_POLL_SECONDS", 0):
            report = {}
            for job, provider in (("rubric", "openai"), ("crosscheck", "anthropic")):
                fake = FakeBatches(provider, fake_judge_reply)
                sample = rj.crosscheck_questions(inputs["key"]) if job == "crosscheck" else ["Q01"]
                qid = sample[0]
                for model_key, _ in picks:
                    run_batch_job(job, model=model_key, only=qid, wait=True, batches=fake)
                judge_name = settings.NB04_JUDGE_JOBS[job]["judge"]
                cache = json.loads((judge_dir / f"{job}_{judge_name}.json").read_text(encoding="utf-8"))
                sent = [body for batch in fake.submitted for body in batch.values()]
                same = all(body == rj.build_request(job, settings.NB04_JUDGE_MODELS[judge_name],
                                                    rj.rubric_messages(inputs["key"][qid], inputs["by_id"][qid],
                                                                       caches[m]["results"][qid]["answer"] or ""))
                           for body, (m, _) in zip(sent, picks))
                standard = settings.NB04_JUDGE_MODELS[judge_name]
                report[job] = {"question": qid, "provider": provider, "batches": len(fake.submitted), "requests": len(sent),
                               "requests_equal_run_judging": same,
                               "judgements": {k: (r["error"] or "ok", r["cost_usd_list"]) for k, r in cache["results"].items()},
                               "spend_batch_usd": cache["spend"]["list_usd_total"],
                               "same_tokens_at_standard_usd": round(sum(2000 * standard["input_usd_per_m"] + 200 * standard["output_usd_per_m"]
                                                                        for _ in sent) / 1e6, 6),
                               "pricing_route": cache["run_settings"]["pricing_route"],
                               "prices_recorded": [cache["run_settings"]["input_usd_per_m"], cache["run_settings"]["output_usd_per_m"]]}
            print(json.dumps(report, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Notebook 04 judging through the providers' batch routes (half price).")
    parser.add_argument("--job", choices=sorted(settings.NB04_JUDGE_JOBS))
    parser.add_argument("--model", choices=sorted(check_parity.all_models()))
    parser.add_argument("--only", default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--wait", action="store_true", help="poll until every batch has ended, then finish the job")
    parser.add_argument("--dry-run", action="store_true", help="two cached answers, fake batch clients; calls nothing")
    parser.add_argument("--credit", action="store_true", help="print the credit ledger and the top-up needed; calls nothing")
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
        return
    if args.credit:
        print(top_up_report())
        return
    if not args.job:
        parser.error("--job is required unless --dry-run is given")
    run_batch_job(args.job, args.model, args.only, args.allow_incomplete, args.wait)


if __name__ == "__main__":
    main()
