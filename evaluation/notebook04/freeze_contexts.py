"""
Freeze retrieval for notebook 04 (plan section 6): run the 100 evaluation questions and the 10 guardrail probes through
the real retriever once, and save the chunks and rendered prompts every candidate will receive. No LLM is called.

Retrieval follows the pipeline's order for a real question: the search text from RagPipeline.search_text (the one
function the chatbot also uses: leading greetings and politeness removed, then spelling corrected), then the locked
bge-small index, top 5. Intent routing is skipped, as in the baseline: all 110 items are real requests and each
scores at or above the threshold, so the router's similarity override could not stop them anyway.

The 100 questions are compared with the baseline run stored at settings.NB04_BASELINE_COMMIT; differences are reported.

contexts.json holds licensed chunk text and is Git ignored. contexts_manifest.json records the same freeze without any
chunk text: with the frozen index and chunks.json, the contexts can be rebuilt and verified against its hashes.

--system-prompt-file renders the prompts with text that is not in settings.py yet, for review before it is applied.
check_parity.py fails until settings.SYSTEM_PROMPT holds the same text, so an unapproved prompt cannot be sent.

Usage (from the project root):
    venv\\Scripts\\python evaluation\\notebook04\\freeze_contexts.py [--system-prompt-file PATH] [--force]
"""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings  # noqa: E402
from src.core import llm  # noqa: E402
from src.core.rag import RagPipeline  # noqa: E402
from src.core.search_text import spelling_corrections  # noqa: E402

# Recorded in the freeze metadata, because the prompt builder changed while SYSTEM_PROMPT did not (plan section 6).
PROMPT_BUILDER = ("src.core.llm.build_messages: system prompt, 5 excerpts, the question as typed, then one spelling-hint line "
                  "only when every word changed by spelling correction, other than British spellings, has 3 or more letters "
                  "and appears in the retrieved excerpts (search_text.spelling_corrections; added 2026-11-11 and narrowed the "
                  "same day, British spellings excluded 2026-11-16, all before any candidate result), then "
                  "settings.ANSWER_REMINDER")


def corrections_for(item: dict, chunks: list) -> list:
    """The spelling-hint word pairs for one item, from its question, search text and retrieved excerpts."""
    return spelling_corrections(item["text"], item["search_query"], [c["text"] if isinstance(c, dict) else c.text for c in chunks])

CHARS_PER_TOKEN = 4       # same estimate as the plan's first measurement, so the two are comparable


def prompt_sha256(messages: list[dict]) -> str:
    """Hash of the exact messages, used to prove every model receives the same prompt."""
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def estimated_tokens(messages: list[dict]) -> int:
    """Rough token count of a prompt: characters / 4."""
    return round(sum(len(m["content"]) for m in messages) / CHARS_PER_TOKEN)


def load_items() -> list[dict]:
    """The 100 frozen questions followed by the 10 frozen probes, in one shape."""
    questions = json.loads(settings.TEST_QUESTIONS_FILE.read_text(encoding="utf-8"))["questions"]
    probes = json.loads(settings.NB04_PROBES_FILE.read_text(encoding="utf-8"))["probes"]
    return ([{"id": q["id"], "kind": "question", "text": q["question"]} for q in questions]
            + [{"id": p["id"], "kind": "probe", "text": p["request"]} for p in probes])


def freeze_item(pipeline: RagPipeline, item: dict) -> dict:
    """Retrieve the chunks for one item and render its prompt."""
    search_query = pipeline.search_text(item["text"])      # the same function the chatbot uses, so a re-freeze cannot drift
    chunks = pipeline.retriever.retrieve(search_query)
    # The spelling hint, exactly as the chatbot adds it: only when spelling correction changed a word of the search text.
    messages = llm.build_messages(item["text"], chunks,
                                  corrections=corrections_for({**item, "search_query": search_query}, chunks))
    return {
        **item,
        "search_query": search_query,
        "chunks": [{"chunk_id": c.chunk_id, "score": round(c.score, 4), "text": c.text,
                    "source_id": c.source_id, "source_name": c.source_name, "source_type": c.source_type,
                    "section": c.section, "page": c.page, "file": c.file, "topic": c.topic} for c in chunks],
        "reaches_generation": bool(chunks) and chunks[0].score >= settings.SIMILARITY_THRESHOLD,
        "messages": messages,
        "prompt_sha256": prompt_sha256(messages),
        "estimated_prompt_tokens": estimated_tokens(messages),
    }


def baseline_at_commit() -> dict:
    """The baseline results as stored in Git at the recorded commit, not whatever is in the working folder."""
    path = settings.NB04_BASELINE_FILE.relative_to(PROJECT_ROOT).as_posix()
    shown = subprocess.run(["git", "show", f"{settings.NB04_BASELINE_COMMIT}:{path}"], cwd=PROJECT_ROOT,
                           capture_output=True, check=True)
    return {r["id"]: r for r in json.loads(shown.stdout.decode("utf-8"))["results"]}


def compare_with_baseline(items: list[dict]) -> dict:
    """Compare the questions' retrieved chunk IDs (and their order and scores) with the baseline run."""
    baseline = baseline_at_commit()
    different_ids, different_scores, different_query = [], [], []
    for item in (i for i in items if i["kind"] == "question"):
        old = baseline[item["id"]]
        if [c["chunk_id"] for c in item["chunks"]] != [c["chunk_id"] for c in old["retrieved"]]:
            different_ids.append(item["id"])
        elif [c["score"] for c in item["chunks"]] != [c["score"] for c in old["retrieved"]]:
            different_scores.append(item["id"])
        if item["search_query"] != old["search_query"]:
            different_query.append(item["id"])
    return {"baseline_commit": settings.NB04_BASELINE_COMMIT, "questions_compared": len(baseline),
            "chunk_id_differences": different_ids, "score_differences": different_scores,
            "search_query_differences": different_query, "identical": not (different_ids or different_scores)}


def token_stats(items: list[dict]) -> dict:
    """Mean and max estimated prompt tokens for a set of items."""
    tokens = [i["estimated_prompt_tokens"] for i in items]
    return {"count": len(tokens), "mean": round(statistics.mean(tokens), 1), "max": max(tokens)}


def retrieval_of(item: dict) -> list:
    """The part of an item that must never change once frozen."""
    return [(c["chunk_id"], c["score"], c["text"]) for c in item["chunks"]]


def build_manifest(data: dict) -> dict:
    """The committable record of the freeze: IDs, scores, search text and hashes, and no chunk text."""
    meta = data["meta"]
    return {
        "meta": {
            "purpose": "Committable record of evaluation/notebook04/contexts.json, without its licensed chunk text. "
                       "Rerun freeze_contexts.py with the frozen index and chunks.json to rebuild the contexts, then "
                       "compare with these hashes.",
            **{k: meta[k] for k in ("created", "embedding_model", "top_k", "similarity_threshold", "spell_correction",
                                    "system_prompt_version", "system_prompt_sha256", "prompt_builder",
                                    "spelling_hint_template", "spelling_hint_items", "baseline_check")},
            "resume_rule": "an answer cached by run_generation.py is reused only while its prompt_sha256 equals its item's",
        },
        "items": [{"id": i["id"], "kind": i["kind"], "search_text": i["search_query"],
                   "chunk_ids": [c["chunk_id"] for c in i["chunks"]], "scores": [c["score"] for c in i["chunks"]],
                   "prompt_sha256": i["prompt_sha256"]} for i in data["items"]],
    }


def back_up_previous_freeze() -> Path | None:
    """Copy the current contexts.json and manifest into settings.NB04_BACKUP_DIR/<UTC time>/ before they are replaced. The folder
    is inside the project and Git ignored; a temp folder is not safe, because Windows' cleanup deletes old files there."""
    existing = [p for p in (settings.NB04_CONTEXTS_FILE, settings.NB04_MANIFEST_FILE) if p.exists()]
    if not existing:
        return None
    folder = settings.NB04_BACKUP_DIR / datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    folder.mkdir(parents=True, exist_ok=True)
    for path in existing:
        (folder / path.name).write_bytes(path.read_bytes())
    return folder


def write_atomic(path: Path, data: dict) -> None:
    """Write JSON through a temp file, so an interruption cannot leave a broken file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    """Build contexts.json and its manifest. Refuses to change frozen retrieval unless --force is given."""
    parser = argparse.ArgumentParser(description="Freeze chunks and rendered prompts for notebook 04.")
    parser.add_argument("--system-prompt-file", type=Path, default=None,
                        help="render with this text instead of settings.SYSTEM_PROMPT (for review)")
    parser.add_argument("--force", action="store_true", help="allow frozen retrieval to change")
    args = parser.parse_args()

    if args.system_prompt_file:      # for this process only; settings.py is not edited
        settings.SYSTEM_PROMPT = args.system_prompt_file.read_text(encoding="utf-8").strip()

    pipeline = RagPipeline()
    items = [freeze_item(pipeline, item) for item in load_items()]

    if settings.NB04_CONTEXTS_FILE.exists() and not args.force:
        previous = json.loads(settings.NB04_CONTEXTS_FILE.read_text(encoding="utf-8"))["items"]
        if [retrieval_of(i) for i in previous] != [retrieval_of(i) for i in items]:
            sys.exit("Retrieval differs from the frozen contexts.json. Nothing was written; use --force only if the "
                     "knowledge base was deliberately changed.")

    questions = [i for i in items if i["kind"] == "question"]
    probes = [i for i in items if i["kind"] == "probe"]
    data = {
        "meta": {
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "embedding_model": settings.EMBEDDING_MODEL,
            "top_k": settings.TOP_K,
            "similarity_threshold": settings.SIMILARITY_THRESHOLD,
            "spell_correction": settings.SPELL_CORRECTION_ENABLED,
            "system_prompt_source": args.system_prompt_file.as_posix() if args.system_prompt_file
            else "config/settings.py",
            "system_prompt": settings.SYSTEM_PROMPT,
            "system_prompt_version": settings.SYSTEM_PROMPT_VERSION,
            "system_prompt_sha256": hashlib.sha256(settings.SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            "prompt_builder": PROMPT_BUILDER,
            "spelling_hint_template": {"line": settings.SPELLING_HINT_TEMPLATE, "typo": settings.SPELLING_HINT_TYPO,
                                       "min_letters": settings.SPELLING_HINT_MIN_LETTERS,
                                       "rule": "British spellings are not typos; every other corrected word has min_letters "
                                               "or more and appears in the excerpts",
                                       "british_variants": settings.SPELLING_BRITISH_VARIANTS},
            "spelling_hint_items": [i["id"] for i in items if corrections_for(i, i["chunks"])],
            "token_estimate": f"characters / {CHARS_PER_TOKEN}",
            "tokens": {"questions": token_stats(questions), "probes": token_stats(probes)},
            "baseline_check": compare_with_baseline(items),
        },
        "items": items,
    }
    backup = back_up_previous_freeze()
    if backup:
        print(f"Previous freeze copied to {backup}")
    write_atomic(settings.NB04_CONTEXTS_FILE, data)
    write_atomic(settings.NB04_MANIFEST_FILE, build_manifest(data))

    print(f"Wrote {settings.NB04_CONTEXTS_FILE} and its manifest with {len(items)} items.")
    print(f"Prompt source: {data['meta']['system_prompt_source']}")
    print(f"Tokens: {data['meta']['tokens']}")
    print(f"Items below the threshold: {[i['id'] for i in items if not i['reaches_generation']]}")
    print(f"Baseline check: {data['meta']['baseline_check']}")


if __name__ == "__main__":
    main()
