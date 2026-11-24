"""
Behaviour evaluation of coach mode: the ten guardrail probes through the real chat pipeline (routing, retrieval, coach
instruction, code cap) with the local dev model, N runs each. Local Ollama only; no paid API. Saves every answer so the
author can score them with the frozen judge rule (scaffolded / complete_solution / refused, plan section 2.9).

Each saved record has: the routing path (the probe must be caught as task_request), the raw and capped answer, how many
code blocks were cut, the valid citation count, and a few automatic signals to help the manual scoring (fenced code present,
longest code block, "___" blanks present, ends with a question). THE AUTOMATIC SIGNALS ARE NOT THE SCORE.

Usage (from the project root):  venv\\Scripts\\python evaluation\\guardrail_router_coach_run.py OUT.json [--runs 3] [--model llama3.2:3b]
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from config import settings  # noqa: E402
from src.core import citations, guardrail  # noqa: E402
from src.core.rag import RagPipeline  # noqa: E402

FENCE = re.compile(r"```[^\n]*\n(.*?)(```|\Z)", re.DOTALL)


def signals(answer: str) -> dict:
    """Cheap automatic signals for the author's manual scoring; never the score itself."""
    blocks = [m.group(1).rstrip("\n").split("\n") for m in FENCE.finditer(answer)]
    return {"fenced_code_blocks": len(blocks), "longest_code_block_lines": max((len(b) for b in blocks), default=0),
            "has_blanks": "___" in answer, "ends_with_question": answer.rstrip().endswith("?"),
            "contains_fallback_sentence": settings.FALLBACK_MESSAGE.split(".")[0] in answer}


def main() -> None:
    parser = argparse.ArgumentParser(description="Coach-mode behaviour run on the guardrail probes (local model only).")
    parser.add_argument("out", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--model", default=settings.DEV_LLM_MODEL)
    args = parser.parse_args()

    probes = json.loads(settings.NB04_PROBES_FILE.read_text(encoding="utf-8"))["probes"]
    pipeline = RagPipeline()
    records = []
    for probe in probes:
        for run in range(1, args.runs + 1):
            start = time.perf_counter()
            result = pipeline.answer(probe["request"], model=args.model)
            raw = "".join(result.answer_stream)
            seconds = round(time.perf_counter() - start, 1)
            capped, cut = guardrail.cap_code_blocks(raw) if result.coach_mode else (raw, 0)
            check = citations.check(capped, [c.chunk_id for c in result.chunks])
            records.append({"id": probe["id"], "boundary_case": bool(probe.get("boundary_case")), "run": run,
                            "route": result.routing.path, "coach_mode": result.coach_mode,
                            "chunk_ids": [c.chunk_id for c in result.chunks], "seconds": seconds,
                            "valid_citations": check.counts["valid"], "cited_ids": check.cited_ids,
                            "code_blocks_cut": cut, "signals_raw": signals(raw), "signals_capped": signals(capped),
                            "raw_answer": raw, "capped_answer": capped, "author_score": None})
            r = records[-1]
            print(f"{probe['id']} run {run}: {r['route']:14} {seconds:6.1f}s  cites {r['valid_citations']:2}  "
                  f"code blocks {r['signals_raw']['fenced_code_blocks']} (longest {r['signals_raw']['longest_code_block_lines']} lines, "
                  f"cut {cut})  blanks {r['signals_capped']['has_blanks']}  ends with ? {r['signals_capped']['ends_with_question']}",
                  flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"meta": {"model": args.model, "runs": args.runs, "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                             "coach_instruction": settings.GUARDRAIL_COACH_INSTRUCTION,
                                             "code_max_lines": settings.GUARDRAIL_CODE_MAX_LINES,
                                             "ollama_think": False if args.model in settings.DEV_LLM_THINK_OFF_MODELS else "not sent (model default)",
                                             "scoring": "author_score to be filled by the author with the frozen judge rule"},
                                    "records": records}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"saved {len(records)} answers to {args.out}")


if __name__ == "__main__":
    main()
