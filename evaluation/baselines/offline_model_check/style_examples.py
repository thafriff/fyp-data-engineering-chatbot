"""Answer-style check, 2026-11-24: the author's example questions through the real chat pipeline (routing, search text,
retrieval, threshold, hint, generation) with the default model, before and after the answer-style prompt change.
Usage: python style_examples.py before|after [runs]. Writes answer_style/examples_<tag>.json (Git ignored: answers quote
licensed chunk text)."""
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from config import settings  # noqa: E402
from src.core import citations  # noqa: E402
from src.core.rag import RagPipeline  # noqa: E402

QUESTIONS = ["whats the difference between oltp and olap", "wht is data engineering",
             "is a data warehouse the same as a database?", "best nasi lemak in Shah Alam",
             "ugh sql is so hard i want to give up"]


def words(text: str) -> int:
    """Word count of the answer as the student sees it, without the cut/repetition notes."""
    for note in (settings.ANSWER_CUT_NOTE, settings.ANSWER_REPEAT_NOTE):
        text = text.replace(note, "")
    return len(re.findall(r"\S+", text))


def main() -> None:
    tag = sys.argv[1]
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    pipeline = RagPipeline()
    out = {"meta": {"tag": tag, "model": settings.DEV_LLM_MODEL, "runs": runs,
                    "system_prompt_sha256": hashlib.sha256(settings.SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
                    "system_prompt_version": settings.SYSTEM_PROMPT_VERSION}, "records": []}
    for q in QUESTIONS:
        for run in range(1, runs + 1):
            start = time.perf_counter()
            result = pipeline.answer(q)
            answer = "".join(result.answer_stream) if result.answer_stream is not None else (result.reply or "")
            ids = [c.chunk_id for c in result.chunks]
            check = citations.check(answer, ids)
            rec = {"question": q, "run": run, "kind": result.routing.kind, "route": result.routing.path,
                   "label": result.routing.label, "top_score": result.routing.top_score,
                   "search_query": result.search_query, "fallback": result.used_fallback,
                   "chunks": [[c.chunk_id, round(c.score, 3)] for c in result.chunks],
                   "seconds": round(time.perf_counter() - start, 1), "words": words(answer),
                   "hit_cap": settings.ANSWER_CUT_NOTE in answer, "valid_citations": check.counts["valid"],
                   "contains_fallback": settings.FALLBACK_MESSAGE.split(".")[0] in answer, "answer": answer}
            out["records"].append(rec)
            print(f"{q[:40]:40} r{run} {rec['route']:20} {rec['seconds']:5.1f}s words {rec['words']:4} "
                  f"cites {rec['valid_citations']} cap {rec['hit_cap']} fallback {rec['contains_fallback']}", flush=True)
    path = HERE / "answer_style" / f"examples_{tag}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("saved", path)


if __name__ == "__main__":
    main()
