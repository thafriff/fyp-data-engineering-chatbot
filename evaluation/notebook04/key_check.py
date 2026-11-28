"""
Pre-flight key check for notebook 04 (STEP 1 of the author's staged plan, 2026-11-27). NOT part of the run.

Sends ONE tiny request per model, the six candidates with their real notebook 04 settings (thinking switch, temperature, output
budget) and the three judges with their job settings, so that each key in .env and each setting is proven before any spend.
The prompt is a fixed connectivity line: no course question, no chunk text, no personal data. Where a host has a free
model-list endpoint (OpenAI, DeepSeek, Mistral), the listed IDs are read too, to confirm the exact model strings.

Nothing is written to the answer or judge caches, so this spend is not in the run's totals: it is saved, with its list-price
cost, in evaluation/notebook04/key_check.json (no keys: errors pass through providers.describe_error, which redacts them).

Usage (from the project root):  venv\\Scripts\\python evaluation\\notebook04\\key_check.py [--only deepseek-flash,gemini-3.8-flash]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_ROOT), str(SCRIPT_DIR)]
sys.stdout.reconfigure(encoding="utf-8")

import providers  # noqa: E402
import run_judging  # noqa: E402
from config import settings  # noqa: E402
from src.core.timeouts import call_with_timeout  # noqa: E402

OUT_FILE = settings.NB04_DIR / "key_check.json"
# JSON is named so that the judges' json_object / schema settings are satisfied; the candidates simply answer it.
MESSAGES = [{"role": "system", "content": "This is a connectivity test. Reply with a short JSON object only."},
            {"role": "user", "content": 'Return {"ok": true} as JSON.'}]
JUDGE_JOB = {"gpt-5.4": "rubric", "gpt-5.4-mini": "ragas", "claude-opus-5-5": "crosscheck"}
# (Mistral was listed here until it was dropped on 2026-11-27; its two checks stay in key_check.json as the record.)
LISTED_IDS = {"openai": ["gpt-4o", "gpt-5.4", "gpt-5.4-mini"], "deepseek": ["deepseek-flash", "deepseek-chat"]}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def spy_on_openai_client(config: dict, captured: dict) -> None:
    """Keep the raw OpenAI-style response, so the record shows the reply's shape (string or Mistral's chunk list) and every
    usage field the host sent. The adapter still parses the same response object."""
    completions = providers._client(config).chat.completions
    original = completions.create

    def create(**kwargs):
        response = original(**kwargs)
        captured["raw"] = response
        return response
    completions.create = create


def shape(response) -> dict:
    """What the raw OpenAI-style reply looked like: content type, chunk types, usage fields (numbers only)."""
    content = response.choices[0].message.content
    usage = response.usage.model_dump() if hasattr(response.usage, "model_dump") else dict(vars(response.usage))
    return {"content_type": type(content).__name__,
            "chunk_types": [providers._field(p, "type") for p in content] if isinstance(content, list) else None,
            "usage_fields": usage}


def check(name: str, config: dict, request: dict) -> dict:
    """One tiny call through the real adapter; returns what was learned (never a key)."""
    captured, started = {}, time.perf_counter()
    entry = {"name": name, "model_id": config["model_id"], "api": config["api"], "key_variable": settings.API_KEY_ENV_VARS[config["key"]],
             "key_set": bool(providers.api_key(config)), "started_at": utc_now(),
             "request_settings": {k: v for k, v in request.items() if k not in ("messages", "contents", "system")}}
    if not entry["key_set"]:
        return {**entry, "ok": False, "error": "key not set in .env"}
    try:
        if config["api"] == "openai_compatible":
            spy_on_openai_client(config, captured)
        result = call_with_timeout(lambda: providers.send(config, request), settings.NB04_REQUEST_TIMEOUT_SECONDS)
    except Exception as error:      # translated errors keep the SDK's original as their cause
        return {**entry, "ok": False, "seconds": round(time.perf_counter() - started, 2), "translated_as": type(error).__name__,
                **providers.describe_error(error)}
    cost = (result.input_tokens * config["input_usd_per_m"] + result.output_tokens * config["output_usd_per_m"]) / 1_000_000
    entry.update(ok=bool(result.text.strip()), seconds=round(time.perf_counter() - started, 2), served_model=result.served_model,
                 finish=result.finish, input_tokens=result.input_tokens, output_tokens=result.output_tokens,
                 reasoning_tokens=result.reasoning_tokens, reasoning_tokens_estimated=result.reasoning_tokens_estimated,
                 cached_input_tokens=result.cached_input_tokens, text=result.text.strip()[:120],
                 cost_usd_list=round(cost, 6))
    if "raw" in captured:
        entry["reply_shape"] = shape(captured["raw"])
    return entry


def listed_models(provider: str, config: dict) -> dict:
    """The model-list endpoint (no tokens, no charge): which of the IDs of interest exist, with any alias or deprecation info."""
    try:
        models = list(providers._client(config).models.list())
    except Exception as error:
        return {"error": providers.describe_error(error)}
    ids = {m.id: m for m in models}
    out = {"models_listed": len(models), "ids_of_interest": {}}
    for wanted in LISTED_IDS[provider]:
        m = ids.get(wanted)
        if m is None:
            out["ids_of_interest"][wanted] = None
            continue
        extra = m.model_dump() if hasattr(m, "model_dump") else {}
        out["ids_of_interest"][wanted] = {k: extra.get(k) for k in ("id", "name", "aliases", "deprecation", "owned_by",
                                                                    "max_context_length", "context_window") if k in extra}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="One tiny request per notebook 04 model.")
    parser.add_argument("--only", default=None, help="comma-separated names")
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    wanted = set(args.only.split(",")) if args.only else None
    record = {"meta": {"purpose": __doc__.strip().splitlines()[0], "date": utc_now(), "prices_checked": settings.NB04_PRICES_CHECKED,
                       "prompt": MESSAGES, "note": "not part of the run; spend not in the run caches"},
              "calls": [], "model_lists": {}}
    if OUT_FILE.exists():
        record = json.loads(OUT_FILE.read_text(encoding="utf-8")) | {"meta": record["meta"]}
        record["calls"] = [c for c in record.get("calls", []) if wanted and c["name"] not in wanted]
    for name, candidate in settings.NB04_CANDIDATES.items():
        if wanted and name not in wanted:
            continue
        record["calls"].append(entry := check(name, candidate, providers.build_request(candidate, MESSAGES)))
        print(json.dumps(entry, ensure_ascii=False, default=str), flush=True)
    for name, judge in settings.NB04_JUDGE_MODELS.items():
        if wanted and name not in wanted:
            continue
        request = run_judging.build_request(JUDGE_JOB[name], judge, MESSAGES)
        record["calls"].append(entry := check(name, judge, request))
        print(json.dumps(entry, ensure_ascii=False, default=str), flush=True)
    for provider, name in (("openai", "gpt-4o"), ("deepseek", "deepseek-flash")):
        if not wanted or name in wanted:      # the free model-list endpoint, for the providers checked in this call
            record["model_lists"][provider] = listed_models(provider, settings.NB04_CANDIDATES[name])
            print(provider, json.dumps(record["model_lists"][provider], ensure_ascii=False, default=str), flush=True)
    record["meta"]["total_cost_usd_list"] = round(sum(c.get("cost_usd_list", 0.0) for c in record["calls"]), 6)
    keys = [v for v in settings.API_KEY_ENV_VARS.values() if os.environ.get(v)]
    text = json.dumps(record, indent=1, ensure_ascii=False, default=str)
    assert not any(os.environ[v] in text for v in keys), "a key would have been written"
    OUT_FILE.write_text(text + "\n", encoding="utf-8")
    print(f"saved {OUT_FILE}; list-price cost of these calls ${record['meta']['total_cost_usd_list']:.6f}")


if __name__ == "__main__":
    main()
