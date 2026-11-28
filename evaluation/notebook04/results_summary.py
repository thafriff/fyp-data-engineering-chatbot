"""
STEP 4 of notebook 04 (author, 2026-11-28): the Table 3.9 scores of the five candidates, with the pre-registered 95% bootstrap intervals,
written to evaluation/notebook04/results_summary.json. Reads saved files only: no model or judge is called and no key is loaded.

Every number that decides the result comes from run_judging.py's own code: build_report() for the point estimates, the decision rule and
the two sensitivity rankings; normalise() and weighted_total() for every bootstrap resample. The frozen weights, anchors and rules
(plan section 3) are read from settings and never changed here.

Bootstrap (plan section 6a(c), fixed by the author on 2026-11-27 before any result). Implementation details, all fixed in code:
- numpy.random.default_rng(NB04_BOOTSTRAP_SEED = 42); NB04_BOOTSTRAP_RESAMPLES = 10,000; percentile intervals (2.5th and 97.5th).
- One matrix of resampled question indices over the 100 frozen questions is shared by all five models, so every comparison is PAIRED.
  Each resample's weighted score is recomputed from its own means: the 1-5 criteria, faithfulness, Response Time (mean latency) and
  Cost (mean list cost per question x 100), each normalised with the frozen anchors; Ease is one score per model and enters unchanged.
  Inside the weighted score, primary Accuracy is the mean over the resampled questions that have a reachable core point.
- Primary Accuracy's own interval (and its pairwise differences) resamples the 94 questions with a reachable core point, from a second
  matrix drawn from the same generator; the all-points measures resample the 100.
- Pre-registration (a): Gemini's Response Time is its paid-tier re-measure of Q01-Q20, so in each resample its mean latency is taken over
  the resampled questions among Q01-Q20, with their re-measured latencies. Every other model uses its generation latency.
- A difference between two models is NOT significant when the 95% interval of the paired difference includes 0. Every pair is reported,
  with no correction for multiple comparisons: descriptive only. The intervals never change the winner.

Usage (from the project root):  venv\\Scripts\\python evaluation\\notebook04\\results_summary.py
"""

import hashlib
import itertools
import json
import math
import statistics
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_generation  # noqa: E402
import run_judging as rj  # noqa: E402
from config import settings  # noqa: E402

CRITERIA = list(settings.NB04_WEIGHTS)        # the eight Table 3.9 criteria, in the frozen order
RAW_FIELD = {"accuracy": "accuracy", "alignment": "alignment", "faithfulness": "faithfulness", "relevance": "relevance",
             "clarity": "clarity", "response_time": "mean_seconds", "cost": "usd_per_100", "ease": "ease"}
PER_QUESTION = ["accuracy", "accuracy_all", "relevance", "clarity", "alignment", "faithfulness", "latency", "cost"]
TOLERANCE = 1e-9


# ---------------------------------------------------------------------------
# Per-question values, built with run_judging's own scoring
# ---------------------------------------------------------------------------

def remeasure_by_id(model: str) -> dict[str, float] | None:
    """The re-measured latency of each question, when pre-registration (a) applies to this model (same rule as
    run_judging.remeasured_latencies, which must agree)."""
    seconds = rj.remeasured_latencies(model)
    if seconds is None:
        return None
    path = run_generation.remeasure_dir() / f"{model}.json"
    results = json.loads(path.read_text(encoding="utf-8"))["results"]
    by_id = {r["id"]: r["latency_seconds"] for r in results.values() if r["kind"] == "question" and not r["error"]
             and r.get("access_tier") == "paid" and r.get("latency_seconds") is not None}
    if sorted(by_id.values()) != sorted(seconds):
        raise AssertionError(f"{model}: the re-measure read here differs from run_judging.remeasured_latencies")
    return by_id


def question_tables(report: dict, caches: dict, inputs: dict, jobs: dict, reviews: dict) -> tuple[list[str], dict]:
    """Per model and question, the values whose means are the criteria; NaN where a value does not exist (primary Accuracy on a
    question with no reachable core point; Response Time outside the re-measured questions). The means over all questions must
    equal build_report's, or this stops."""
    qids = [i["id"] for i in inputs["items"] if i["kind"] == "question"]
    rows = {r["model"]: r for r in report["rows"]}
    tables = {}
    for model, cache in caches.items():
        answers = cache["results"]
        remeasured = remeasure_by_id(model)
        values = {field: [] for field in PER_QUESTION}
        for qid in qids:
            record, key = answers[qid], f"{model}|{qid}"
            scored = rj.score_rubric_record(key, jobs["rubric"][key], inputs, record["answer"] or "", reviews)
            faith = jobs["ragas"][key]["faithfulness"]
            values["accuracy"].append(math.nan if scored["accuracy"] is None else scored["accuracy"])
            for field in ("accuracy_all", "relevance", "clarity", "alignment"):
                values[field].append(scored[field])
            values["faithfulness"].append(math.nan if faith is None else faith)
            if remeasured is None:
                values["latency"].append(record["latency_seconds"])
            else:
                values["latency"].append(remeasured.get(qid, math.nan))
            values["cost"].append(record["cost_usd_list"])
        tables[model] = {field: np.array(v, dtype=float) for field, v in values.items()}
        check = {"accuracy": rows[model]["accuracy"], "accuracy_all": rows[model]["accuracy_all"],
                 "relevance": rows[model]["relevance"], "clarity": rows[model]["clarity"], "alignment": rows[model]["alignment"],
                 "faithfulness": rows[model]["faithfulness"], "latency": rows[model]["mean_seconds"],
                 "cost": rows[model]["usd_per_100"] / settings.NB04_EXPECTED_QUESTIONS}
        for field, expected in check.items():
            got = float(np.nanmean(tables[model][field]))
            if abs(got - expected) > TOLERANCE:
                raise AssertionError(f"{model} {field}: per-question mean {got} differs from build_report's {expected}")
    return qids, tables


# ---------------------------------------------------------------------------
# The bootstrap
# ---------------------------------------------------------------------------

def _nanmean_rows(values: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)      # an all-NaN resample gives NaN, handled as None below
        return np.nanmean(values, axis=1)


def _none_if_nan(value: float) -> float | None:
    return None if value is None or math.isnan(value) else float(value)


def resample_metrics(tables: dict, ease: dict, index: np.ndarray) -> dict[str, list[dict]]:
    """For each model, one metrics dict per resample (row of index), in the shape run_judging.normalise expects."""
    out = {}
    for model, t in tables.items():
        means = {field: _nanmean_rows(t[field][index]) for field in PER_QUESTION}
        out[model] = [{"accuracy": _none_if_nan(means["accuracy"][b]), "accuracy_all": _none_if_nan(means["accuracy_all"][b]),
                       "relevance": float(means["relevance"][b]), "clarity": float(means["clarity"][b]),
                       "alignment": float(means["alignment"][b]), "faithfulness": _none_if_nan(means["faithfulness"][b]),
                       "mean_seconds": _none_if_nan(means["latency"][b]),
                       "usd_per_100": float(means["cost"][b]) * settings.NB04_EXPECTED_QUESTIONS,
                       "ease": ease[model]} for b in range(index.shape[0])]
    return out


def score_replicates(metrics: dict[str, list[dict]]) -> dict[str, dict[str, np.ndarray]]:
    """Per model and resample: each normalised criterion, the weighted total, the two sensitivity totals and the all-points total,
    all through run_judging.normalise and run_judging.weighted_total."""
    out = {}
    raw_keys = [f"raw_{c}" for c in CRITERIA] + ["raw_accuracy_all"]
    for model, rows in metrics.items():
        series = {key: [] for key in CRITERIA + raw_keys + ["total", "without_response_time", "without_cost", "all_points_total"]}
        for m in rows:
            norm = rj.normalise(m)
            for criterion in CRITERIA:
                series[criterion].append(norm[criterion])
                series[f"raw_{criterion}"].append(m[RAW_FIELD[criterion]])
            series["raw_accuracy_all"].append(m["accuracy_all"])
            series["total"].append(rj.weighted_total(norm))
            series["without_response_time"].append(rj.weighted_total(norm, "response_time"))
            series["without_cost"].append(rj.weighted_total(norm, "cost"))
            all_points = None if m["accuracy_all"] is None else (m["accuracy_all"] - 1) / 4
            series["all_points_total"].append(rj.weighted_total({**norm, "accuracy": all_points}))
        out[model] = {k: np.array([math.nan if v is None else v for v in vals], dtype=float) for k, vals in series.items()}
    return out


def interval(values: np.ndarray) -> list[float] | None:
    """The percentile interval at NB04_BOOTSTRAP_INTERVAL, over the resamples that have a value."""
    kept = values[~np.isnan(values)]
    if kept.size == 0:
        return None
    tail = (1 - settings.NB04_BOOTSTRAP_INTERVAL) / 2 * 100
    low, high = np.percentile(kept, [tail, 100 - tail])
    return [float(low), float(high)]


def run_bootstrap(qids: list[str], tables: dict, ease: dict, reachable: list[int]) -> dict:
    """All resampled quantities. index100 is shared by every model (paired); index94 resamples the reachable questions only."""
    rng = np.random.default_rng(settings.NB04_BOOTSTRAP_SEED)
    resamples, n = settings.NB04_BOOTSTRAP_RESAMPLES, len(qids)
    index100 = rng.integers(0, n, size=(resamples, n))
    reach = np.array(reachable)
    index94 = reach[rng.integers(0, len(reach), size=(resamples, len(reach)))]
    replicates = score_replicates(resample_metrics(tables, ease, index100))
    accuracy94 = {}
    for model, t in tables.items():
        raw = _nanmean_rows(t["accuracy"][index94])
        accuracy94[model] = {"raw": raw, "normalised": (raw - 1) / 4}
    return {"replicates": replicates, "accuracy94": accuracy94, "resamples": resamples,
            "questions": n, "reachable_questions": len(reach)}


def pairwise(models: list[str], series: dict[str, np.ndarray], points: dict[str, float]) -> list[dict]:
    """Every pair: the point difference (first minus second), its paired 95% interval, and whether 0 lies outside it."""
    out = []
    for a, b in itertools.combinations(models, 2):
        ci = interval(series[a] - series[b])
        out.append({"a": a, "b": b, "difference": points[a] - points[b], "ci95": ci,
                    "significant": ci is not None and not (ci[0] <= 0 <= ci[1])})
    return out


# ---------------------------------------------------------------------------
# The summary
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _latency(values: list[float]) -> dict | None:
    if not values:
        return None
    return {"n": len(values), "median": statistics.median(values), "mean": statistics.fmean(values), "p95": rj._p95(values)}


def guardrail_counts(g: dict) -> dict:
    """Outcome counts over the 27 standard runs (nine probes x three), the majorities, and G04 apart. Labels only, no text."""
    runs = {"scaffolded": 0, "complete_solution": 0, "refused": 0}
    for probe, outcomes in g["per_probe"].items():
        if probe != "G04":
            for outcome in outcomes:
                runs[outcome] = runs.get(outcome, 0) + 1
    return {"standard_runs": runs, "standard_majorities": g["standard_counts"], "g04_runs": g["per_probe"].get("G04", []),
            "g04_majority": g["g04"], "complete_solution_answers_for_author_review": len(g["for_author_review"])}


def build_summary() -> dict:
    """Score everything from the saved files and return the summary (nothing is written here)."""
    report = rj.build_report()
    caches, inputs = rj.load_answer_caches(), rj.load_inputs()
    jobs = {job: rj._load_job(job) for job in settings.NB04_JUDGE_JOBS}
    reviews = rj._load_reviews()
    models = [m for m in settings.NB04_CANDIDATES if m in caches]
    rows = {r["model"]: r for r in report["rows"]}
    ease = {m: rows[m]["ease"] for m in models}
    qids, tables = question_tables(report, caches, inputs, jobs, reviews)
    reachable = [n for n, qid in enumerate(qids) if any(rj.reachable_core_flags(inputs["key"][qid], inputs["by_id"][qid]))]
    boot = run_bootstrap(qids, tables, ease, reachable)
    rep = boot["replicates"]

    points = {m: {"total": rows[m]["total"], "all_points_total": report["all_points"]["totals"][m],
                  **{f"without_{s['excluded']}": s["totals"][m] for s in report["sensitivity"]}} for m in models}

    def ranking(key):
        ordered = sorted(models, key=lambda m: -points[m][key])
        return [{"rank": n, "model": m, "total": points[m][key], "ci95": interval(rep[m][key])} for n, m in enumerate(ordered, 1)]

    remeasure_dir = run_generation.remeasure_dir()
    out_models = []
    for m in models:
        r, cache = rows[m], caches[m]
        questions = [x for x in cache["results"].values() if x["kind"] == "question" and not x["error"]]
        generation_seconds = [x["latency_seconds"] for x in questions]
        rt_seconds = [float(v) for v in tables[m]["latency"] if not math.isnan(v)]
        criteria = {}
        for c in CRITERIA:
            raw_value = r[RAW_FIELD[c]]
            if c == "accuracy":
                raw_ci, norm_ci = interval(boot["accuracy94"][m]["raw"]), interval(boot["accuracy94"][m]["normalised"])
            elif c == "ease":
                raw_ci = norm_ci = None                      # one score per model: no interval (plan section 6a)
            else:
                raw_ci, norm_ci = interval(rep[m][f"raw_{c}"]), interval(rep[m][c])
            criteria[c] = {"weight": settings.NB04_WEIGHTS[c], "raw": raw_value, "raw_ci95": raw_ci,
                           "normalised": r["normalised"][c], "normalised_ci95": norm_ci,
                           "weighted_points": settings.NB04_WEIGHTS[c] * r["normalised"][c]}
        out_models.append({
            "model": m, "slot": r["slot"], "access_tiers": r["access_tiers"], "response_time_source": r["response_time_source"],
            "criteria": criteria, "total": r["total"], "total_ci95": interval(rep[m]["total"]),
            "without_response_time": points[m]["without_response_time"],
            "without_response_time_ci95": interval(rep[m]["without_response_time"]),
            "without_cost": points[m]["without_cost"], "without_cost_ci95": interval(rep[m]["without_cost"]),
            "accuracy_primary": {"mean_1_to_5": r["accuracy"], "ci95": interval(boot["accuracy94"][m]["raw"]),
                                 "questions": settings.NB04_EXPECTED_QUESTIONS - r["no_reachable_core"]},
            "accuracy_secondary": {"mean_1_to_5": r["accuracy_all"], "ci95": interval(rep[m]["raw_accuracy_all"]),
                                   "questions": settings.NB04_EXPECTED_QUESTIONS},
            "all_points_total": points[m]["all_points_total"], "all_points_total_ci95": interval(rep[m]["all_points_total"]),
            "latency_seconds": {"generation": _latency(generation_seconds), "response_time_basis": _latency(rt_seconds)},
            "cost": {"usd_per_100_questions": r["usd_per_100"], "rm_per_100_questions": r["usd_per_100"] * settings.NB04_USD_TO_MYR,
                     "price_basis": "official paid list price" + (" (free of charge: no paid price)" if r["usd_per_100"] == 0 else "")},
            "citations": {"no_valid_citation": r["no_valid_citation"], "bad_citation": r["bad_citation"]},
            "contradictions": {"flagged": r["contradictions_flagged"], "unreviewed": r["contradictions_unreviewed"]},
            "guardrail": guardrail_counts(report["guardrail"][m]),
            "crosscheck": report["crosscheck"]["per_candidate"].get(m, {}),
        })

    gemma_path = remeasure_dir / "gemma-4-31b-it.json"
    gemma_note = None
    if gemma_path.exists():
        g = json.loads(gemma_path.read_text(encoding="utf-8"))
        seconds = [x["latency_seconds"] for x in g["results"].values() if not x["error"]]
        same = [caches["gemma-4-31b-it"]["results"][k]["latency_seconds"] for k in g["results"]] if "gemma-4-31b-it" in caches else []
        gemma_note = {"note": ("DESCRIPTIVE ONLY, not in Gemma's score: the first 20 questions sent again on 2026-11-28 from the billed "
                               "project. Gemma has no paid tier, so its Response Time stays its free-tier generation latency "
                               "(pre-registration a). Time of day and billing changed together, so the speed-up cannot be put on "
                               "billing alone."),
                      "billing_period": _latency(seconds), "same_questions_in_free_tier_generation": _latency(same)}

    decision = report["main"]
    ordered = sorted(models, key=lambda m: -points[m]["total"])
    gap = points[ordered[0]]["total"] - points[ordered[1]]["total"] if len(ordered) > 1 else None
    files = {**{f"raw/{m}.json": settings.NB04_RAW_DIR / f"{m}.json" for m in models},
             **{f"raw/remeasure/{p.name}": p for p in sorted(remeasure_dir.glob("*.json"))},
             **{f"judge/{job}_{e['judge']}.json": settings.NB04_JUDGE_DIR / f"{job}_{e['judge']}.json"
                for job, e in settings.NB04_JUDGE_JOBS.items()},
             "ease_log.json": settings.NB04_EASE_LOG_FILE, "contradiction_reviews.json": settings.NB04_CONTRADICTION_REVIEWS_FILE,
             "answer_key.json": settings.NB04_ANSWER_KEY_FILE, "contexts_manifest.json": settings.NB04_MANIFEST_FILE}
    return {
        "meta": {
            "title": "Notebook 04 model comparison: Table 3.9 weighted scores",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "generated_by": "evaluation/notebook04/results_summary.py (reads saved files only; no model or judge called)",
            "contents": "aggregate numbers only: no answer text, no chunk text, no judge quotes, no keys",
            "system_prompt_version": settings.SYSTEM_PROMPT_VERSION.split(";")[0][:120],
            "weights": settings.NB04_WEIGHTS,
            "anchors": {"one_to_five": "(mean - 1) / 4", "faithfulness": "as is",
                        "response_time_seconds_best_worst": [settings.NB04_RESPONSE_TIME_BEST_SECONDS,
                                                             settings.NB04_RESPONSE_TIME_WORST_SECONDS],
                        "cost_usd_per_100_questions_worst": settings.NB04_COST_WORST_USD},
            "decision_rule": (f"highest weighted total wins; if the top two are within {settings.NB04_TIE_MARGIN}, the higher primary "
                              "Accuracy wins (plan section 3)"),
            "bootstrap": {"resamples": boot["resamples"], "seed": settings.NB04_BOOTSTRAP_SEED, "interval": settings.NB04_BOOTSTRAP_INTERVAL,
                          "method": "percentile", "rng": "numpy.random.default_rng", "questions": boot["questions"],
                          "reachable_questions_for_primary_accuracy": boot["reachable_questions"],
                          "paired": True, "multiplicity_correction": None,
                          "details": "see the docstring of results_summary.py and plan section 6a(c)"},
            "usd_to_myr": settings.NB04_USD_TO_MYR, "usd_to_myr_source": settings.NB04_USD_TO_MYR_SOURCE,
            "inputs_sha256": {name: _sha256(path) for name, path in files.items()},
            "provisional": report["provisional"],
        },
        "models": out_models,
        "decision": {
            "winner": decision["winner"], "top_by_total": ordered[0], "gap_between_top_two": gap,
            "tie_rule_applied": decision["winner"] != ordered[0],
            "winner_with_all_points_accuracy": report["all_points"]["winner"],
            "ranking": ranking("total"),
            "sensitivity": [{"excluded": s["excluded"], "winner": s["winner"], "same_as_main": s["winner"] == decision["winner"],
                             "ranking": ranking(f"without_{s['excluded']}")} for s in report["sensitivity"]],
        },
        "pairwise": {
            "total": pairwise(models, {m: rep[m]["total"] for m in models}, {m: points[m]["total"] for m in models}),
            "without_response_time": pairwise(models, {m: rep[m]["without_response_time"] for m in models},
                                              {m: points[m]["without_response_time"] for m in models}),
            "without_cost": pairwise(models, {m: rep[m]["without_cost"] for m in models},
                                     {m: points[m]["without_cost"] for m in models}),
            "criteria": {c: pairwise(models, {m: (boot["accuracy94"][m]["normalised"] if c == "accuracy" else rep[m][c]) for m in models},
                                     {m: rows[m]["normalised"][c] for m in models})
                         for c in CRITERIA if c != "ease"},
        },
        "crosscheck_overall": report["crosscheck"]["overall"],
        "descriptive_notes": {
            "gemma_billing_period_latency": gemma_note,
            "free_tier": ("gemini-3.8-flash answered Q01-Q05 on Google's free tier and Q06-G10 on the paid tier (billing active from "
                          "2026-11-28); its Response Time is its paid-tier re-measure of Q01-Q20 (pre-registration a). gemma-4-31b-it "
                          "has no paid tier: all its answers and its Response Time are free tier, which partly measures provider load "
                          "(Report TODO 41). Cost uses the official paid list price; Gemma is free of charge."),
            "ease": "Ease of Implementation scored by the author on 2026-11-28 by the plan section 2.8 rule, before any scoring was run.",
        },
        "spend": report["spend"],
    }


def write_summary(path: Path | None = None) -> dict:
    """Score, write results_summary.json (atomically) and return the summary."""
    summary = build_summary()
    run_generation.save_cache(path or settings.NB04_RESULTS_SUMMARY_FILE, summary)
    return summary


def _fmt(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def _ci(ci, digits=3):
    return "-" if ci is None else f"[{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def print_summary(summary: dict) -> None:
    """The final ranking with every criterion, the sensitivity rankings and the differences that are not significant."""
    by_model = {m["model"]: m for m in summary["models"]}
    print("=== Final ranking (Table 3.9 weighted score, frozen weights; 95% bootstrap interval) ===")
    head = "".join(f"{c[:9]:>10}" for c in CRITERIA)
    print(f"{'rank':<5}{'model':<20}{head}{'TOTAL':>9}  95% CI")
    print(f"{'':<5}{'weight':<20}" + "".join(f"{settings.NB04_WEIGHTS[c]:>10.2f}" for c in CRITERIA))
    for row in summary["decision"]["ranking"]:
        m = by_model[row["model"]]
        print(f"{row['rank']:<5}{row['model']:<20}" + "".join(f"{m['criteria'][c]['normalised']:>10.3f}" for c in CRITERIA)
              + f"{row['total']:>9.4f}  {_ci(row['ci95'], 4)}")
    d = summary["decision"]
    print(f"Winner (decision rule): {d['winner']}; gap to second {_fmt(d['gap_between_top_two'], 4)}; tie rule applied: "
          f"{d['tie_rule_applied']}; winner with all-points Accuracy: {d['winner_with_all_points_accuracy']}")
    for s in d["sensitivity"]:
        print(f"Without {s['excluded']}: winner {s['winner']} ({'same' if s['same_as_main'] else 'DIFFERENT'}): "
              + ", ".join(f"{r['model']} {r['total']:.4f} {_ci(r['ci95'], 4)}" for r in s["ranking"]))
    print("Weighted-score differences NOT significant (95% paired interval includes 0):")
    for p in summary["pairwise"]["total"]:
        if not p["significant"]:
            print(f"  {p['a']} - {p['b']}: {p['difference']:+.4f} {_ci(p['ci95'], 4)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    result = write_summary()
    print_summary(result)
    print(f"\nsaved {settings.NB04_RESULTS_SUMMARY_FILE}")
