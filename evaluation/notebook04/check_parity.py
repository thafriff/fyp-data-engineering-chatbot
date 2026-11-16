"""
Parity checks for notebook 04 (plan section 6). run_generation.py calls run_checks() first and aborts on any failure,
so no model is called unless all of these hold:
1. contexts.json holds all 100 questions and 10 probes, each with 5 chunks, and each reaches generation;
2. each stored prompt hash matches its stored messages;
3. the messages equal a fresh render from the current settings.py and the current chunks.json, so the frozen prompts
   are not stale (this also blocks a prompt wording that has not been applied to settings.py);
4. the prompt SHA-256 is identical across all models, read back from each model's built API request;
5. "section:" appears exactly five times in every prompt;
6. Q43's prompt contains "fct_monthly_zone_revenue: select";
7. the 100 questions' chunk IDs equal the baseline run's retrieved IDs;
8. every cached answer was made from its own item's current prompt (per item; run_generation.py, which replaces stale answers, tolerates them);
9. contexts_manifest.json records the same IDs, order, scores, search text and hashes as contexts.json, and holds no
   chunk text.

Usage (from the project root; exit code 1 means failed):
    venv\\Scripts\\python evaluation\\notebook04\\check_parity.py [--system-prompt-file PATH]
--system-prompt-file previews check 3 against text that is not in settings.py yet. The runner never uses it.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import providers  # noqa: E402
from freeze_contexts import prompt_sha256  # noqa: E402
from config import settings  # noqa: E402
from src.core import llm  # noqa: E402
from src.core.retriever import RetrievedChunk  # noqa: E402
from freeze_contexts import corrections_for  # noqa: E402

Q43_REQUIRED_TEXT = "fct_monthly_zone_revenue: select"


def all_models() -> dict:
    """The six candidates plus the fallback hosts."""
    return {**settings.NB04_CANDIDATES, **settings.NB04_FALLBACK_HOSTS}


def _stored_chunks(item: dict) -> list[RetrievedChunk]:
    """Rebuild retriever objects from the frozen file so the real prompt builder can render them again."""
    return [RetrievedChunk(chunk_id=c["chunk_id"], score=c["score"], text=c["text"], source_id=c["source_id"],
                           source_name=c["source_name"], source_type=c["source_type"], section=c["section"],
                           page=c["page"], file=c["file"], topic=c["topic"]) for c in item["chunks"]]


def run_checks(allow_stale_answers: bool = False) -> list[str]:
    """Run every check and return one message per failure (an empty list means all passed).
    allow_stale_answers: used only by run_generation.py, which replaces any cached answer made from an earlier prompt of its
    item (the per-item resume rule). Everything else, judging included, keeps check 8 strict."""
    if not settings.NB04_CONTEXTS_FILE.exists():
        return [f"{settings.NB04_CONTEXTS_FILE} does not exist; run freeze_contexts.py first."]
    items = json.loads(settings.NB04_CONTEXTS_FILE.read_text(encoding="utf-8"))["items"]
    current_chunks = {c["chunk_id"]: c["text"]
                      for c in json.loads(settings.CHUNKS_FILE.read_text(encoding="utf-8"))}
    baseline = {r["id"]: r for r in json.loads(settings.NB04_BASELINE_FILE.read_text(encoding="utf-8"))["results"]}
    failures = []

    questions = [i for i in items if i["kind"] == "question"]
    probes = [i for i in items if i["kind"] == "probe"]
    if len(questions) != settings.NB04_EXPECTED_QUESTIONS or len(probes) != settings.NB04_EXPECTED_PROBES:
        failures.append(f"Check 1: expected {settings.NB04_EXPECTED_QUESTIONS} questions and "
                        f"{settings.NB04_EXPECTED_PROBES} probes, found {len(questions)} and {len(probes)}.")

    for item in items:
        tag, messages = item["id"], item["messages"]
        if len(item["chunks"]) != settings.TOP_K or not item["reaches_generation"]:
            failures.append(f"Check 1: {tag} has {len(item['chunks'])} chunks or is below the similarity threshold.")
        if prompt_sha256(messages) != item["prompt_sha256"]:
            failures.append(f"Check 2: {tag} stored hash does not match its stored messages.")
        if any(current_chunks.get(c["chunk_id"]) != c["text"] for c in item["chunks"]):
            failures.append(f"Check 3: {tag} holds chunk text that differs from chunks.json.")
        corrections = corrections_for(item, item["chunks"])     # the spelling hint, rebuilt from stored values
        if llm.build_messages(item["text"], _stored_chunks(item), corrections=corrections) != messages:
            failures.append(f"Check 3: {tag} prompt differs from a fresh render with the current settings.py.")
        hashes = {name: prompt_sha256(providers.request_messages(model, providers.build_request(model, messages)))
                  for name, model in all_models().items()}
        if set(hashes.values()) != {item["prompt_sha256"]}:
            failures.append(f"Check 4: {tag} prompt hash differs across models: "
                            f"{[n for n, h in hashes.items() if h != item['prompt_sha256']]}.")
        found = sum(m["content"].count("section:") for m in messages)
        if found != settings.TOP_K:
            failures.append(f"Check 5: {tag} has 'section:' {found} times, expected {settings.TOP_K}.")
        if tag == "Q43" and Q43_REQUIRED_TEXT not in messages[1]["content"]:
            failures.append(f"Check 6: Q43's prompt does not contain '{Q43_REQUIRED_TEXT}'.")
        if item["kind"] == "question":
            frozen = [c["chunk_id"] for c in item["chunks"]]
            if frozen != [c["chunk_id"] for c in baseline.get(tag, {}).get("retrieved", [])]:
                failures.append(f"Check 7: {tag} chunk IDs differ from the baseline.")

    by_id = {i["id"]: i["prompt_sha256"] for i in items}
    for cache in sorted(settings.NB04_RAW_DIR.glob("*.json")) if settings.NB04_RAW_DIR.exists() else []:
        results = json.loads(cache.read_text(encoding="utf-8")).get("results", {})
        # A repeated probe's result key is "<id>#r<repeat>".
        stale = [k for k, r in results.items() if r.get("prompt_sha256") != by_id.get(k.split("#")[0])]
        unknown = [k for k in stale if k.split("#")[0] not in by_id]
        if allow_stale_answers:
            stale = unknown         # answers for items that no longer exist are still a failure
        if stale:
            failures.append(f"Check 8: {cache.name} holds answers from other prompts: {stale[:5]}.")
    failures += _manifest_failures(items)
    return failures


def _manifest_failures(items: list[dict]) -> list[str]:
    """Check 9: the committable manifest describes exactly this freeze and carries no licensed text."""
    if not settings.NB04_MANIFEST_FILE.exists():
        return [f"Check 9: {settings.NB04_MANIFEST_FILE} does not exist; run freeze_contexts.py."]
    text = settings.NB04_MANIFEST_FILE.read_text(encoding="utf-8")
    manifest = json.loads(text)["items"]
    expected = [{"id": i["id"], "kind": i["kind"], "search_text": i["search_query"],
                 "chunk_ids": [c["chunk_id"] for c in i["chunks"]], "scores": [c["score"] for c in i["chunks"]],
                 "prompt_sha256": i["prompt_sha256"]} for i in items]
    failures = [] if manifest == expected else ["Check 9: the manifest does not match contexts.json."]
    if any(c["text"] in text for i in items for c in i["chunks"]):
        failures.append("Check 9: the manifest contains chunk text.")
    return failures


def main() -> None:
    """Print the outcome and exit with code 1 if any check failed."""
    parser = argparse.ArgumentParser(description="Parity checks for the notebook 04 generation run.")
    parser.add_argument("--system-prompt-file", type=Path, default=None, help="preview unapplied prompt text")
    args = parser.parse_args()
    if args.system_prompt_file:
        settings.SYSTEM_PROMPT = args.system_prompt_file.read_text(encoding="utf-8").strip()
    failures = run_checks()
    if failures:
        print(f"PARITY FAILED: {len(failures)} problem(s). First ten:")
        print("\n".join(failures[:10]))
        sys.exit(1)
    print(f"PARITY PASSED: all checks hold across {len(all_models())} model configurations.")


if __name__ == "__main__":
    main()
