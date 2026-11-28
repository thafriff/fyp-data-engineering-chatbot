"""
Prepaid credit ledger for notebook 04 (author, 2026-11-28: check that the OpenAI and Anthropic balances fit before each batch).

Neither OpenAI nor Anthropic reports a prepaid API key's balance through its API, so the balance is kept as a ledger: the credit the
author stated on 2026-11-27, before the pre-flight (settings.NB04_PREPAID_CREDIT_USD), minus every cost recorded on disk since:
generation caches (raw/*.json), latency re-measures (raw/remeasure/*.json), judge caches (judge/*.json, at the price their route
billed) and the pre-flight key checks (key_check.json). Everything is at list price (batch list price for batch judging), which is an
upper bound on the real bill: cached input and DeepSeek's off-peak discount were billed less. The provider's billing page stays the
authority. Reads files only; calls no API.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings  # noqa: E402

KEY_CHECK_FILE = settings.NB04_DIR / "key_check.json"


def provider_of(name: str) -> str | None:
    """The key (provider) a candidate or judge model is billed to, e.g. "openai"."""
    entry = settings.NB04_CANDIDATES.get(name) or settings.NB04_JUDGE_MODELS.get(name)
    return entry["key"] if entry else None


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def spent_usd(provider: str) -> dict:
    """What the ledger has recorded against one provider, by part, in USD."""
    parts = {"generation": 0.0, "remeasure": 0.0, "judging": 0.0, "key_checks": 0.0}
    remeasure = settings.NB04_RAW_DIR / settings.NB04_REMEASURE_SUBDIR
    for part, folder in (("generation", settings.NB04_RAW_DIR), ("remeasure", remeasure)):
        for path in sorted(folder.glob("*.json")) if folder.exists() else []:
            if provider_of(path.stem) == provider:
                parts[part] += _read(path).get("spend", {}).get("list_usd_total", 0.0)
    folder = settings.NB04_JUDGE_DIR
    for path in sorted(folder.glob("*.json")) if folder.exists() else []:
        cache = _read(path)
        if provider_of(cache.get("run_settings", {}).get("judge", "")) == provider:
            parts["judging"] += cache.get("spend", {}).get("list_usd_total", 0.0)
    if KEY_CHECK_FILE.exists():
        for call in _read(KEY_CHECK_FILE).get("calls", []):
            if provider_of(call.get("name", "")) == provider:
                parts["key_checks"] += call.get("cost_usd_list") or 0.0
    return {k: round(v, 6) for k, v in parts.items()}


def credit_usd(provider: str) -> float | None:
    """The stated prepaid credit, or None for a provider without one (Google: free tier or billed afterwards)."""
    return settings.NB04_PREPAID_CREDIT_USD.get(provider)


def left_usd(provider: str, reserved: float = 0.0) -> float | None:
    """Credit minus recorded spend minus a reservation (e.g. batches still running). None without a prepaid credit."""
    credit = credit_usd(provider)
    if credit is None:
        return None
    return credit - sum(spent_usd(provider).values()) - reserved


def summary(provider: str, reserved: float = 0.0) -> str:
    """One line for the terminal and for cost_summary.md."""
    credit = credit_usd(provider)
    if credit is None:
        return f"{provider}: no prepaid credit recorded"
    parts = spent_usd(provider)
    shown = ", ".join(f"{k} ${v:.4f}" for k, v in parts.items() if v)
    reserve = f", reserved for running batches ${reserved:.4f}" if reserved else ""
    return (f"{provider}: credit ${credit:.2f} - recorded ${sum(parts.values()):.4f} ({shown or 'nothing'}){reserve} "
            f"= ${left_usd(provider, reserved):.4f} left (ledger at list price, an upper bound on the bill)")


if __name__ == "__main__":
    for name in settings.NB04_PREPAID_CREDIT_USD:
        print(summary(name))
