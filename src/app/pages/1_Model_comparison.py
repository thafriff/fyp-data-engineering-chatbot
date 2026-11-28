"""
Read-only "Model comparison" page (notebook 04, STEP 4). It shows evaluation/notebook04/results_summary.json: the Table 3.9 weighted
ranking with 95% bootstrap intervals, the eight criterion scores per model, primary and secondary Accuracy, latency, cost per 100
answers in RM, the guardrail outcomes, both pre-registered sensitivity rankings, the judge cross-check and the free-tier notes.

It never calls a model or a judge. Its only button recomputes the scores from the saved results (results_summary.write_summary, which
reads files only). UI only: the scoring lives in evaluation/notebook04/results_summary.py.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from config import settings  # noqa: E402

CRITERION_NAMES = {"accuracy": "Accuracy", "alignment": "Course-Material Alignment", "faithfulness": "RAG Compatibility",
                   "relevance": "Relevance", "clarity": "Clarity", "response_time": "Response Time", "cost": "Cost",
                   "ease": "Ease of Implementation"}


def recompute() -> None:
    """Rescore from the saved caches (no API call) and rewrite results_summary.json."""
    notebook04 = str(PROJECT_ROOT / "evaluation" / "notebook04")
    if notebook04 not in sys.path:
        sys.path.insert(0, notebook04)
    import results_summary          # imported only when asked: the page itself needs only the JSON
    results_summary.write_summary()


def load_summary() -> dict | None:
    path = settings.NB04_RESULTS_SUMMARY_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def theme_colours() -> dict[str, str]:
    """The model colours for the active theme: one fixed colour per model, never by rank."""
    try:
        mode = "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        mode = "light"
    return settings.NB04_MODEL_COLOURS[mode]


def ci_text(ci, digits=3) -> str:
    return "-" if not ci else f"{ci[0]:.{digits}f} to {ci[1]:.{digits}f}"


def colour_scale(models: list[str]) -> alt.Scale:
    colours = theme_colours()
    return alt.Scale(domain=models, range=[colours[m] for m in models])


def ranking_chart(summary: dict, models: list[str]) -> alt.Chart:
    """Weighted total per model with its 95% interval: a point and a 2px rule, models named on the axis."""
    rows = [{"model": r["model"], "total": r["total"], "low": r["ci95"][0], "high": r["ci95"][1],
             "interval": ci_text(r["ci95"], 4)} for r in summary["decision"]["ranking"]]
    data = pd.DataFrame(rows)
    order = [r["model"] for r in rows]
    low = min(data["low"].min(), data["total"].min())
    domain = [max(0.0, round(low - 0.02, 2)), 1.0]
    y = alt.Y("model:N", sort=order, title=None)
    x_scale = alt.Scale(domain=domain, zero=False)
    tooltip = [alt.Tooltip("model:N", title="Model"), alt.Tooltip("total:Q", title="Weighted total", format=".4f"),
               alt.Tooltip("interval:N", title="95% interval")]
    rule = alt.Chart(data).mark_rule(strokeWidth=2).encode(
        y=y, x=alt.X("low:Q", scale=x_scale, title="Weighted total (0 to 1)", axis=alt.Axis(tickCount=6, format=".2f")),
        x2="high:Q",
        color=alt.Color("model:N", scale=colour_scale(models), legend=None), tooltip=tooltip)
    point = alt.Chart(data).mark_point(filled=True, size=110, opacity=1).encode(
        y=y, x=alt.X("total:Q", scale=x_scale), color=alt.Color("model:N", scale=colour_scale(models), legend=None),
        tooltip=tooltip)
    return (rule + point).properties(height=40 * len(rows) + 30)


def criteria_chart(summary: dict, models: list[str]) -> alt.Chart:
    """Grouped bars: the eight normalised criterion scores, one bar per model inside each criterion."""
    rows = []
    for m in summary["models"]:
        for key, c in m["criteria"].items():
            rows.append({"criterion": f"{CRITERION_NAMES[key]} ({c['weight']:.2f})", "model": m["model"],
                         "score": c["normalised"], "raw": c["raw"], "interval": ci_text(c["normalised_ci95"])})
    data = pd.DataFrame(rows)
    order = [f"{CRITERION_NAMES[k]} ({w:.2f})" for k, w in settings.NB04_WEIGHTS.items()]
    return alt.Chart(data).mark_bar(size=12, cornerRadiusEnd=4).encode(
        x=alt.X("criterion:N", sort=order, title=None, axis=alt.Axis(labelAngle=-30, labelLimit=180)),
        xOffset=alt.XOffset("model:N", sort=models),
        y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 1]), title="Normalised score (0 to 1)"),
        color=alt.Color("model:N", scale=colour_scale(models), sort=models, legend=alt.Legend(title="Model", orient="top")),
        tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("criterion:N", title="Criterion (weight)"),
                 alt.Tooltip("score:Q", title="Normalised", format=".3f"), alt.Tooltip("raw:Q", title="Raw value", format=".3f"),
                 alt.Tooltip("interval:N", title="95% interval (normalised)")],
    ).properties(height=340)


def main() -> None:
    st.set_page_config(page_title=settings.MODEL_COMPARISON_PAGE_TITLE, page_icon="📊", layout="wide")
    st.title(settings.MODEL_COMPARISON_PAGE_TITLE)
    st.caption("Notebook 04: the five candidates scored on the eight Table 3.9 criteria with the frozen weights and anchors. "
               "Read-only: this page never calls a model or a judge.")
    if st.button(settings.MODEL_COMPARISON_RECOMPUTE_LABEL):
        with st.spinner("Scoring the saved answers and judgements (no API call)..."):
            try:
                recompute()
                st.success("Scores recomputed from the saved results.")
            except Exception as error:          # e.g. the Git-ignored caches are not on this machine
                st.error(f"Could not recompute: {type(error).__name__}: {error}")

    summary = load_summary()
    if summary is None:
        st.info(f"No results yet: {settings.NB04_RESULTS_SUMMARY_FILE.name} does not exist.")
        return
    meta, decision = summary["meta"], summary["decision"]
    models = [m["model"] for m in summary["models"]]
    by_model = {m["model"]: m for m in summary["models"]}
    if meta["provisional"]:
        st.warning("PROVISIONAL:\n- " + "\n- ".join(meta["provisional"]))

    st.subheader("Final weighted ranking")
    tie = " (decided by the tie rule)" if decision["tie_rule_applied"] else ""
    st.markdown(f"**Winner under the decision rule: {decision['winner']}**{tie}. Gap to second place: "
                f"{decision['gap_between_top_two']:.4f}. Rule: {meta['decision_rule']}.")
    st.altair_chart(ranking_chart(summary, models), width="stretch")
    table = []
    for r in decision["ranking"]:
        m = by_model[r["model"]]
        table.append({"Rank": r["rank"], "Model": r["model"],
                      **{CRITERION_NAMES[k]: round(m["criteria"][k]["normalised"], 3) for k in settings.NB04_WEIGHTS},
                      "Weighted total": round(r["total"], 4), "95% interval": ci_text(r["ci95"], 4)})
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    weights = ", ".join(f"{CRITERION_NAMES[k]} {w:.2f}" for k, w in meta["weights"].items())
    st.caption(f"Criterion columns are normalised scores (0 to 1). Weights: {weights}.")
    not_significant = [f"{p['a']} vs {p['b']} ({p['difference']:+.4f}; {ci_text(p['ci95'], 4)})"
                       for p in summary["pairwise"]["total"] if not p["significant"]]
    st.markdown("**Differences that are not significant** (95% paired interval includes 0): "
                + ("; ".join(not_significant) if not_significant else "none") + ". Descriptive only: the intervals never change "
                f"the winner. Bootstrap: {meta['bootstrap']['resamples']:,} question resamples, seed {meta['bootstrap']['seed']}, "
                "percentile intervals, paired, no multiplicity correction.")

    st.subheader("The eight criterion scores")
    st.altair_chart(criteria_chart(summary, models), width="stretch")

    st.subheader("Accuracy: primary and secondary")
    st.dataframe(pd.DataFrame([{
        "Model": m["model"],
        "Primary (reachable core points), 1-5": round(m["accuracy_primary"]["mean_1_to_5"], 3),
        "Primary 95% interval": ci_text(m["accuracy_primary"]["ci95"], 2),
        "Questions": m["accuracy_primary"]["questions"],
        "Secondary (all core points), 1-5": round(m["accuracy_secondary"]["mean_1_to_5"], 3),
        "Secondary 95% interval": ci_text(m["accuracy_secondary"]["ci95"], 2)} for m in summary["models"]]),
        hide_index=True, width="stretch")
    st.caption(f"The primary measure is the one in the ranking. Winner with the secondary measure: "
               f"{decision['winner_with_all_points_accuracy']}.")

    st.subheader("Latency and cost")
    st.dataframe(pd.DataFrame([{
        "Model": m["model"],
        "Median s (generation)": round(m["latency_seconds"]["generation"]["median"], 2),
        "p95 s (generation)": round(m["latency_seconds"]["generation"]["p95"], 2),
        "Response Time mean s": round(m["latency_seconds"]["response_time_basis"]["mean"], 2),
        "Response Time based on": m["response_time_source"],
        "RM per 100 answers": round(m["cost"]["rm_per_100_questions"], 3),
        "Price basis": m["cost"]["price_basis"]} for m in summary["models"]]), hide_index=True, width="stretch")
    st.caption(f"Cost: official list price per 100 question answers, USD 1 = RM {meta['usd_to_myr']} ({meta['usd_to_myr_source']}).")

    st.subheader("Pre-registered sensitivity rankings")
    columns = st.columns(len(decision["sensitivity"]))
    for column, s in zip(columns, decision["sensitivity"]):
        with column:
            label = CRITERION_NAMES[s["excluded"]]
            st.markdown(f"**Without {label}**: winner {s['winner']} ({'same as' if s['same_as_main'] else 'DIFFERENT from'} the main result)")
            st.dataframe(pd.DataFrame([{"Rank": r["rank"], "Model": r["model"], "Total": round(r["total"], 4),
                                        "95% interval": ci_text(r["ci95"], 4)} for r in s["ranking"]]),
                         hide_index=True, width="stretch")
            key = f"without_{s['excluded']}"
            ns = [f"{p['a']} vs {p['b']}" for p in summary["pairwise"][key] if not p["significant"]]
            st.caption("Not significant: " + ("; ".join(ns) if ns else "none"))

    st.subheader("Guardrail compliance (not in the weighted score)")
    st.dataframe(pd.DataFrame([{
        "Model": m["model"],
        "Scaffolded": m["guardrail"]["standard_runs"].get("scaffolded", 0),
        "Complete solution": m["guardrail"]["standard_runs"].get("complete_solution", 0),
        "Refused": m["guardrail"]["standard_runs"].get("refused", 0),
        "Majorities (9 probes)": ", ".join(f"{k} {v}" for k, v in sorted(m["guardrail"]["standard_majorities"].items())),
        "G04 (boundary case)": m["guardrail"]["g04_majority"],
        "Awaiting the author's review": m["guardrail"]["complete_solution_answers_for_author_review"]}
        for m in summary["models"]]), hide_index=True, width="stretch")
    st.caption("Counts over the 27 standard runs (nine probes, three runs each), judged by gpt-5.4. Every complete_solution answer is "
               "reviewed by the author (plan section 2.9); until then these are the judge's outcomes.")

    st.subheader("Judge cross-check: gpt-5.4 against claude-opus-5-5")
    groups = [("All candidates", summary["crosscheck_overall"])] + [(m["model"], m["crosscheck"]) for m in summary["models"]]
    fields = {"accuracy": "Accuracy", "accuracy_all": "Accuracy (all points)", "relevance": "Relevance", "clarity": "Clarity",
              "alignment": "Alignment (judge)"}
    st.dataframe(pd.DataFrame([{"Answers": name, "n": g["relevance"]["n"],
                                **{f"{label} exact / within 1 / gap": f"{g[f]['exact']:.0%} / {g[f]['within_1']:.0%} / {g[f]['mean_gap']:+.2f}"
                                   for f, label in fields.items() if f in g}} for name, g in groups if g]),
                 hide_index=True, width="stretch")
    st.caption("Gap = gpt-5.4 minus claude-opus-5-5. The two judges run under different settings (temperature 0 and reasoning off "
               "for gpt-5.4; thinking at low effort for claude-opus-5-5).")

    st.subheader("Notes")
    notes = summary["descriptive_notes"]
    st.markdown(f"- **Free tier.** {notes['free_tier']}")
    gemma = notes.get("gemma_billing_period_latency")
    if gemma:
        b, f = gemma["billing_period"], gemma["same_questions_in_free_tier_generation"]
        st.markdown(f"- **Gemma latency in the billing period (descriptive only).** Median {b['median']:.1f} s, mean {b['mean']:.1f} s on "
                    f"Q01-Q20, against median {f['median']:.1f} s, mean {f['mean']:.1f} s for the same questions in its free-tier "
                    f"generation. {gemma['note']}")
    st.markdown(f"- **Ease of Implementation.** {notes['ease']}")
    st.caption(f"Generated {meta['generated_at']} by {meta['generated_by']}. {meta['contents']}.")


if __name__ == "__main__":       # Streamlit runs a page as __main__; tests import its functions
    main()
