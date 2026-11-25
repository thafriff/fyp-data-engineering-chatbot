"""
Live check of the intent classifier on the fixed set evaluation/intent_classifier_cases.json (DEV PROTOTYPE; app layer only,
not part of notebook 04). Each message goes through the chat pipeline's own routing (intent.route with the pipeline's
search-text and similarity functions) and, when it is routed to retrieval, the pipeline's threshold decision (top-1 similarity
against settings.SIMILARITY_THRESHOLD): that decides "fallback" against "answered". No answer is generated, so the check is
fast; generation is not what this check is about.

--classifier-file runs with another classifier prompt and examples (a Python dict literal with "prompt" and "examples"), for
example the version before a change, without editing settings.py. The case file's hash is checked first.

Usage (from the project root, Ollama running):
    venv\\Scripts\\python evaluation\\intent_classifier_check.py OUT.json [--runs 3] [--classifier-file PATH]
"""

import argparse
import ast
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from config import settings  # noqa: E402
from src.core import intent  # noqa: E402
from src.core.rag import RagPipeline  # noqa: E402

CASES = ROOT / "evaluation" / "intent_classifier_cases.json"
HASH = ROOT / "evaluation" / "intent_classifier_cases.sha256"


def outcome(pipeline: RagPipeline, message: str) -> dict:
    """Route one message the way RagPipeline.answer does, and say what the student would get."""
    start = time.perf_counter()
    routed = intent.route(message, top_score=pipeline._top_score, clean=pipeline.search_text)
    rec = {"kind": routed.kind, "path": routed.path, "label": routed.label,
           "override_score": None if routed.top_score is None else round(routed.top_score, 3)}
    if routed.kind == "question":
        score = pipeline.retriever.retrieve(pipeline.search_text(message), top_k=1)[0].score
        rec["top1"] = round(score, 3)
        rec["result"] = "answered" if score >= settings.SIMILARITY_THRESHOLD else "fallback"
    else:
        rec["result"] = routed.kind          # "emotional", "small_talk" or "unclear"
        rec["reply"] = routed.reply
    rec["seconds"] = round(time.perf_counter() - start, 2)
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("out", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--classifier-file", type=Path, default=None)
    args = parser.parse_args()
    recorded = HASH.read_text(encoding="utf-8").split()[0]
    if hashlib.sha256(CASES.read_bytes()).hexdigest() != recorded:
        sys.exit("The case file does not match its recorded hash; it must not be edited.")
    if args.classifier_file:
        variant = ast.literal_eval(args.classifier_file.read_text(encoding="utf-8"))
        settings.INTENT_CLASSIFIER_PROMPT, settings.INTENT_CLASSIFIER_EXAMPLES = variant["prompt"], variant["examples"]
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    pipeline = RagPipeline()
    records, passed = [], 0
    for case in cases:
        runs = [outcome(pipeline, case["message"]) for _ in range(args.runs)]
        ok = sum(r["result"] == case["expect"] for r in runs)
        passed += ok == args.runs
        label_agree = sum(r["label"] == case.get("label") for r in runs) if case.get("label") else None
        records.append({**case, "runs": runs, "passed_runs": ok, "label_agreement": label_agree})
        results = Counter(r["result"] for r in runs)
        labels = Counter(r["label"] for r in runs if r["label"])
        print(f"{'PASS' if ok == args.runs else 'FAIL':4} {case['message'][:40]:40} expect {case['expect']:10} got "
              f"{dict(results)} labels {dict(labels)}", flush=True)
    meta = {"model": settings.DEV_LLM_MODEL, "runs": args.runs, "case_sha256": recorded,
            "classifier_source": args.classifier_file.as_posix() if args.classifier_file else "config/settings.py",
            "classifier_prompt_sha256": hashlib.sha256(settings.INTENT_CLASSIFIER_PROMPT.encode("utf-8")).hexdigest(),
            "classifier_examples": settings.INTENT_CLASSIFIER_EXAMPLES,
            "cases_passed_all_runs": passed, "cases": len(cases)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"meta": meta, "records": records}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{passed} of {len(cases)} cases passed in all {args.runs} runs. Saved {args.out}")


if __name__ == "__main__":
    main()
