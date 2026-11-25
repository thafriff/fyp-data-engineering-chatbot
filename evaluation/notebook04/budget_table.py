"""
Notebook 04 budget table in RM (plan section 8), computed from config/settings.py only: candidates, judge jobs, prices, token
estimates, the 120% stop factor, the overall hard stop and the USD to RM rate. Calls nothing and reads no answer.

"Expected" is the estimate each hard stop is built on. "Max (stop)" is 120% of it: the runner stops a model or a judge job once
its spend passes that, saving state, and waits for the author. "Billed" is what a provider would actually charge: a free-tier
model is counted at list price by the runner (so it can only over-count) but billed nothing.

Usage (from the project root):  venv\\Scripts\\python evaluation\\notebook04\\budget_table.py
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_DIR.parents[1]), str(SCRIPT_DIR)]
sys.stdout.reconfigure(encoding="utf-8")

import run_generation  # noqa: E402
import run_judging  # noqa: E402
from config import settings  # noqa: E402

RM = settings.NB04_USD_TO_MYR
FACTOR = settings.NB04_MODEL_STOP_FACTOR


def rows() -> list[tuple[str, str, str, float, bool]]:
    """(group, name, provider, expected USD, billed?) for every model run and every judge job."""
    out = []
    for name, cand in settings.NB04_CANDIDATES.items():
        out.append(("Generation (130 answers each)", name, cand["key"], run_generation.estimated_spend_usd(cand),
                    not cand["free_tier"]))
    labels = {"rubric": "Rubric judge (Accuracy, Relevance, Clarity, Alignment)", "ragas": "RAGAS faithfulness judge",
              "guardrail": "Guardrail judge", "crosscheck": "Cross-check judge"}
    for job, entry in settings.NB04_JUDGE_JOBS.items():
        judge = settings.NB04_JUDGE_MODELS[entry["judge"]]
        out.append((labels[job], f"{entry['judge']} x {entry['calls']} calls", judge["key"], run_judging.estimated_job_usd(job), True))
    return out


def main() -> None:
    table = rows()
    print(f"USD 1 = RM {RM:.2f} (settings.NB04_USD_TO_MYR, an assumption); stop factor {FACTOR:.0%}; "
          f"NB04_EST_INPUT_TOKENS {settings.NB04_EST_INPUT_TOKENS}\n")
    print(f"| Item | Model | Provider | Expected RM | Max (stop) RM | Billed? |\n|---|---|---|---|---|---|")
    for group, name, provider, usd, billed in table:
        print(f"| {group} | {name} | {provider} | {usd * RM:.2f} | {usd * RM * FACTOR:.2f} | {'yes' if billed else 'no (free tier)'} |")
    total = sum(r[3] for r in table)
    billed = sum(r[3] for r in table if r[4])
    gen = sum(r[3] for r in table if r[0].startswith("Generation"))
    print(f"\nGeneration total: expected RM {gen * RM:.2f}, max RM {gen * RM * FACTOR:.2f}")
    print(f"Judging total:    expected RM {(total - gen) * RM:.2f}, max RM {(total - gen) * RM * FACTOR:.2f}")
    print(f"ALL, counted at list price: expected RM {total * RM:.2f}, max (every stop at 120%) RM {total * RM * FACTOR:.2f}")
    print(f"ALL, actually billed:       expected RM {billed * RM:.2f}, max RM {billed * RM * FACTOR:.2f}")
    print(f"Overall hard stop (settings.NB04_TOTAL_STOP_MYR): RM {settings.NB04_TOTAL_STOP_MYR}")
    by_provider = {}
    for _, _, provider, usd, is_billed in table:
        if is_billed:
            by_provider[provider] = by_provider.get(provider, 0.0) + usd
    print("\nBilled by provider (top-up needed): " + "; ".join(
        f"{p} expected RM {u * RM:.2f}, max RM {u * RM * FACTOR:.2f}" for p, u in by_provider.items()))


if __name__ == "__main__":
    main()
