"""
Writes evaluation/notebook04/cost_summary.md from the saved caches (author's COST DOCUMENTATION step, 2026-11-27). Calls nothing.

Per model and per judge: the official list price per 1M input/output tokens (date checked, source URL), total input and output tokens,
number of calls, mean tokens per call, mean cost per question, total in USD and RM (rate and date), the list-price cost against what was
actually charged, and a grand total against the RM96 hard stop.

"List price" is what the runner counts (every input token at the uncached rate, every output token, also on a free tier) and what the Cost
criterion uses. "Actually charged" is an ESTIMATE of the bill from the same token counts: a free tier is charged nothing; prompt-cache hits
reported by the host are priced at its cached rate (gpt-4o, DeepSeek); DeepSeek calls outside its peak hours are priced at half. The
providers' billing dashboards are the authority for what was charged. The file holds numbers only: no answer text, no chunk text, no keys.

Usage (from the project root):  venv\\Scripts\\python evaluation\\notebook04\\cost_summary.py
"""

import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8")
from config import settings  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import credit_ledger  # noqa: E402

OUT_FILE = settings.NB04_DIR / "cost_summary.md"
KEY_CHECK_FILE = settings.NB04_DIR / "key_check.json"
RM = settings.NB04_USD_TO_MYR


def is_deepseek_peak(started_at: str | None) -> bool:
    """DeepSeek's peak hours (UTC, Monday to Friday). A call with no timestamp counts as peak, the dearer rate."""
    if not started_at:
        return True
    when = datetime.fromisoformat(started_at).astimezone(timezone.utc)
    return when.weekday() < 5 and any(start <= when.hour < end for start, end in settings.NB04_DEEPSEEK_PEAK_HOURS_UTC)


def success_start(record: dict) -> str | None:
    """When the call that produced this record's tokens started."""
    calls = [c for c in record.get("call_log") or [] if c.get("outcome") == "success"]
    return calls[-1]["started_at"] if calls else None


def record_tier(record: dict, cache: dict) -> str:
    """The access tier an answer was made on; answers saved before the field existed have their run's original tier."""
    return record.get("access_tier") or ("free" if cache["run_settings"].get("free_tier") else "paid")


def charged_usd(model: str, candidate: dict, record: dict, cache: dict) -> float:
    """Estimated bill for one billed call (see the module docstring). A free-tier call is charged nothing."""
    if record_tier(record, cache) == "free":
        return 0.0
    tokens_in, tokens_out = record.get("input_tokens") or 0, record.get("output_tokens") or 0
    cached = min(record.get("cached_input_tokens") or 0, tokens_in)
    cached_rate = settings.NB04_CACHED_INPUT_USD_PER_M.get(model, candidate["input_usd_per_m"])
    usd = ((tokens_in - cached) * candidate["input_usd_per_m"] + cached * cached_rate
           + tokens_out * candidate["output_usd_per_m"]) / 1_000_000
    if model == "deepseek-flash" and not is_deepseek_peak(success_start(record)):
        usd *= settings.NB04_DEEPSEEK_OFF_PEAK_FACTOR
    return usd


def cache_row(model: str, candidate: dict, cache: dict) -> dict:
    """Token, call and cost figures for one answer cache (a model's run or its latency re-measure)."""
    results = list(cache["results"].values())
    billed = [r for r in results + cache.get("superseded", []) if r.get("input_tokens") is not None]
    done = [r for r in results if not r["error"]]
    questions = [r for r in done if r["kind"] == "question"]
    attempts = (sum(len(r.get("call_log") or []) for r in results + cache.get("superseded", []))
                + sum(len(e.get("call_log") or []) for e in cache.get("retries_exhausted", []) + cache.get("access_blocked", [])))
    tiers = {}
    for r in done:
        tiers[record_tier(r, cache)] = tiers.get(record_tier(r, cache), 0) + 1
    seconds = sorted(r["latency_seconds"] for r in questions if r.get("latency_seconds") is not None)
    return {
        "model": model, "candidate": candidate, "missing": False, "answers": len(done), "tiers": tiers,
        "errors": sum(bool(r["error"]) for r in results), "billed_calls": cache["spend"]["billed_calls"], "attempts": attempts,
        "input_tokens": sum(r["input_tokens"] for r in billed), "output_tokens": sum(r["output_tokens"] for r in billed),
        "reasoning_tokens": sum(r.get("reasoning_tokens") or 0 for r in billed),
        "cached_input_tokens": sum(r.get("cached_input_tokens") or 0 for r in billed),
        "list_usd": cache["spend"]["list_usd_total"],
        "charged_usd": sum(charged_usd(model, candidate, r, cache) for r in billed),
        "question_cost_mean": (sum(r["cost_usd_list"] for r in questions) / len(questions)) if questions else None,
        "superseded": len(cache.get("superseded", [])),
        "deepseek_off_peak_calls": (sum(not is_deepseek_peak(success_start(r)) for r in billed) if model == "deepseek-flash" else None),
        "latency": ({"median": statistics.median(seconds), "mean": statistics.fmean(seconds),
                     "p95": seconds[max(0, math.ceil(0.95 * len(seconds)) - 1)], "n": len(seconds)} if seconds else None),
    }


def model_rows(folder: Path | None = None) -> list[dict]:
    """One row per candidate with a cache in the folder (default: the answer caches; missing ones are marked)."""
    rows = []
    for model, candidate in settings.NB04_CANDIDATES.items():
        path = (folder or settings.NB04_RAW_DIR) / f"{model}.json"
        if path.exists():
            rows.append(cache_row(model, candidate, json.loads(path.read_text(encoding="utf-8"))))
        elif folder is None:
            rows.append({"model": model, "candidate": candidate, "missing": True})
    return rows


def access_label(row: dict) -> str:
    """e.g. "paid", "free tier" or "free tier 5 / paid 125" for a model that moved tier mid-run."""
    tiers = row["tiers"] or {"free" if row["candidate"]["free_tier"] else "paid": 0}
    if len(tiers) == 1:
        return "free tier" if "free" in tiers else "paid"
    return " / ".join(f"{'free tier' if t == 'free' else 'paid'} {n}" for t, n in sorted(tiers.items()))


def judge_rows() -> list[dict]:
    """One row per judge job: actual spend if its cache exists, otherwise the plan estimate at standard and at batch price."""
    rows = []
    for job, entry in settings.NB04_JUDGE_JOBS.items():
        judge = settings.NB04_JUDGE_MODELS[entry["judge"]]
        estimate = entry["calls"] * (entry["est_input_tokens"] * judge["input_usd_per_m"]
                                     + entry["est_output_tokens"] * judge["output_usd_per_m"]) / 1_000_000
        batch_in, batch_out = settings.NB04_JUDGE_BATCH_PRICES[entry["judge"]]
        batch_estimate = entry["calls"] * (entry["est_input_tokens"] * batch_in + entry["est_output_tokens"] * batch_out) / 1_000_000
        path = settings.NB04_JUDGE_DIR / f"{job}_{entry['judge']}.json"
        cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        rows.append({"job": job, "judge": entry["judge"], "judge_cfg": judge, "calls_planned": entry["calls"], "estimate_usd": estimate,
                     "batch_prices": (batch_in, batch_out), "batch_estimate_usd": batch_estimate,
                     "spent": cache["spend"] if cache else None,
                     "route": cache["run_settings"].get("pricing_route", "standard") if cache else None})
    return rows


def key_check_usd() -> tuple[float, float, int]:
    """Pre-flight key checks: (list USD, charged USD, calls). Free-tier models were charged nothing."""
    if not KEY_CHECK_FILE.exists():
        return 0.0, 0.0, 0
    calls = json.loads(KEY_CHECK_FILE.read_text(encoding="utf-8"))["calls"]
    # They ran on 2026-11-27, when every Google model was on the free tier (billing was activated on 2026-11-28).
    free = {name for name, c in settings.NB04_CANDIDATES.items() if c["key"] == "google"}
    listed = sum(c.get("cost_usd_list", 0.0) for c in calls)
    return listed, sum(c.get("cost_usd_list", 0.0) for c in calls if c["name"] not in free), len(calls)


def money(usd: float) -> str:
    return f"${usd:.4f} (RM{usd * RM:.2f})"


def render() -> str:
    rows, judges = model_rows(), judge_rows()
    key_list, key_charged, key_calls = key_check_usd()
    now = datetime.now(timezone.utc).isoformat(timespec="minutes")
    complete = all(not r["missing"] and r["answers"] >= settings.NB04_ANSWERS_PER_MODEL and not r["errors"] for r in rows)
    out = [f"# Notebook 04 cost summary",
           "",
           f"Generated {now} by `evaluation/notebook04/cost_summary.py` from the saved caches (numbers only: no answer text, chunk text or keys).",
           f"Status: generation {'COMPLETE' if complete else 'IN PROGRESS'} for the {len(rows)} candidates "
           f"({settings.NB04_ANSWERS_PER_MODEL} answers each: 100 questions + 10 probes x 3); judging "
           f"{'not run yet' if all(j['spent'] is None for j in judges) else 'partly or fully run'}.",
           "",
           f"- Exchange rate: USD 1 = RM {RM} ({settings.NB04_USD_TO_MYR_SOURCE}).",
           f"- List prices read on {settings.NB04_PRICES_CHECKED} from each provider's own pricing page (sources below).",
           "- **List price** = what the runner counts and what the Cost criterion uses: every input token at the uncached rate and every output "
           "token (reasoning included), also on a free tier. **Actually charged** = an estimate of the bill from the same tokens: free tier "
           "RM0; prompt-cache hits at the host's cached rate (gpt-4o, DeepSeek); DeepSeek off-peak calls at half. The providers' dashboards "
           "are the authority.",
           "",
           "## Generation",
           "",
           "| Model | Access | List price in / out per 1M | Answers (errors) | Billed calls / all attempts | Input tokens (cached) | Output tokens (reasoning) "
           "| Mean in / out per call | Mean cost per question | List-price total | Actually charged (est.) |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        c = r["candidate"]
        if r["missing"]:
            out.append(f"| `{r['model']}` | | ${c['input_usd_per_m']:.2f} / ${c['output_usd_per_m']:.2f} | not started | | | | | | | |")
            continue
        calls = max(r["billed_calls"], 1)
        access = access_label(r)
        note = f"; {r['deepseek_off_peak_calls']} of {r['billed_calls']} calls off-peak" if r["deepseek_off_peak_calls"] is not None else ""
        out.append(f"| `{r['model']}` | {access}{note} | ${c['input_usd_per_m']:.2f} / ${c['output_usd_per_m']:.2f} | {r['answers']} ({r['errors']}) "
                   f"| {r['billed_calls']} / {r['attempts']} | {r['input_tokens']:,} ({r['cached_input_tokens']:,}) "
                   f"| {r['output_tokens']:,} ({r['reasoning_tokens']:,}) | {r['input_tokens'] / calls:,.0f} / {r['output_tokens'] / calls:,.0f} "
                   f"| {'$%.5f' % r['question_cost_mean'] if r['question_cost_mean'] is not None else '-'} | {money(r['list_usd'])} "
                   f"| {money(r['charged_usd'])} |")
    gen_list = sum(r.get("list_usd", 0.0) for r in rows)
    gen_charged = sum(r.get("charged_usd", 0.0) for r in rows)
    out += [f"| **Generation total** | | | | | | | | | **{money(gen_list)}** | **{money(gen_charged)}** |",
            "",
            "Access: the tier each answer was made on (recorded per answer). Gemma 4 has no paid tier on the Gemini API, so its answers are "
            "free-tier answers even from a billed project, and its list price is $0.",
            "",
            "Superseded answers (made from an earlier prompt of their item, billed and kept for the record): "
            + (", ".join(f"`{r['model']}` {r['superseded']}" for r in rows if not r["missing"] and r["superseded"]) or "none") + ".",
            "",
            "## Latency re-measure (pre-registration a)",
            "",
            f"The first {settings.NB04_REMEASURE_QUESTIONS} frozen questions sent again on the paid tier for Response Time "
            f"(`raw/{settings.NB04_REMEASURE_SUBDIR}/`); these answers are never scored. Gemma's is descriptive only (no paid tier).",
            ""]
    remeasure = model_rows(settings.NB04_RAW_DIR / settings.NB04_REMEASURE_SUBDIR)
    if remeasure:
        out += ["| Model | Access | Answers (errors) | Latency median / mean / p95 (n) | List-price total | Actually charged (est.) |",
                "|---|---|---|---|---|---|"]
        for r in remeasure:
            lat = r["latency"]
            timing = f"{lat['median']:.2f} / {lat['mean']:.2f} / {lat['p95']:.2f} s ({lat['n']})" if lat else "-"
            out.append(f"| `{r['model']}` | {access_label(r)} | {r['answers']} ({r['errors']}) | {timing} | {money(r['list_usd'])} "
                       f"| {money(r['charged_usd'])} |")
    else:
        out.append("Not run yet.")
    remeasure_list = sum(r["list_usd"] for r in remeasure)
    remeasure_charged = sum(r["charged_usd"] for r in remeasure)
    out += ["",
            "## Judging",
            "",
            "| Job | Judge | Standard price in / out per 1M | Batch price in / out per 1M | Calls planned | Estimate at standard price "
            "| Estimate at batch price | Spent (route) |",
            "|---|---|---|---|---|---|---|---|"]
    judge_spent = 0.0
    for j in judges:
        cfg = j["judge_cfg"]
        spent = j["spent"]
        judge_spent += spent["list_usd_total"] if spent else 0.0
        out.append(f"| {j['job']} | `{j['judge']}` | ${cfg['input_usd_per_m']:.2f} / ${cfg['output_usd_per_m']:.2f} "
                   f"| ${j['batch_prices'][0]:.3f} / ${j['batch_prices'][1]:.2f} | {j['calls_planned']} "
                   f"| {money(j['estimate_usd'])} | {money(j['batch_estimate_usd'])} "
                   f"| {money(spent['list_usd_total']) + ', ' + str(spent['billed_calls']) + ' calls (' + j['route'] + ')' if spent else 'not run'} |")
    total_list = gen_list + remeasure_list + judge_spent + key_list
    out += ["",
            "## Pre-flight key checks (not part of the run)",
            "",
            f"{key_calls} tiny calls (`key_check.json`): list price {money(key_list)}, charged {money(key_charged)} (free-tier calls charged nothing).",
            "",
            "## Grand total",
            "",
            "| | List price | Actually charged (est.) |",
            "|---|---|---|",
            f"| Generation | {money(gen_list)} | {money(gen_charged)} |",
            f"| Latency re-measure | {money(remeasure_list)} | {money(remeasure_charged)} |",
            f"| Judging | {money(judge_spent)} | {money(judge_spent)} (paid judges: no free tier; batch discounts appear here once used) |",
            f"| Pre-flight checks | {money(key_list)} | {money(key_charged)} |",
            f"| **All** | **{money(total_list)}** | **{money(gen_charged + remeasure_charged + judge_spent + key_charged)}** |",
            "",
            f"Hard stop: RM{settings.NB04_TOTAL_STOP_MYR} (USD {settings.NB04_TOTAL_STOP_MYR / RM:.2f}), counted at list price on generation, "
            f"re-measures and judging. Used so far: RM{(gen_list + remeasure_list + judge_spent) * RM:.2f} "
            f"({(gen_list + remeasure_list + judge_spent) * RM / settings.NB04_TOTAL_STOP_MYR:.1%}).",
            "",
            "## Prepaid credit (ledger)",
            "",
            "No provider reports a prepaid key's balance through its API, so this is a ledger: the credit stated on 2026-11-27 minus every "
            "cost recorded since, at list (or batch list) price, an upper bound on the bill. The billing pages are the authority.",
            ""]
    out += [f"- {credit_ledger.summary(provider)}" for provider in settings.NB04_PREPAID_CREDIT_USD]
    out += ["",
            "## Sources",
            ""]
    out += [f"- {provider}: {url} (read {settings.NB04_PRICES_CHECKED})" for provider, url in settings.NB04_PRICE_SOURCES.items()]
    out += [f"- Exchange rate: https://www.bnm.gov.my/kuala-lumpur-usd/myr-reference-rate ({settings.NB04_USD_TO_MYR_SOURCE})", ""]
    return "\n".join(out)


def main() -> None:
    text = render()
    OUT_FILE.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nsaved {OUT_FILE}")


if __name__ == "__main__":
    main()
