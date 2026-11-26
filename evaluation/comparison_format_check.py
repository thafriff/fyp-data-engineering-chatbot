"""
Live check of the comparison answer format (DEV PROTOTYPE; app layer only, not part of notebook 04). Each question goes through
the real chat pipeline (RagPipeline.answer: routing, search text, retrieval, threshold, generation) with the default local model,
with the format ON and, as a control, OFF (settings.COMPARISON_FORMAT_ENABLED patched in-process; settings.py is not edited).

A run has a PROPER TABLE when the answer holds exactly one markdown table with a header of at least 3 cells (aspect + 2 items),
a separator row, 3 to 5 data rows, and the same number of cells in every row. Also recorded: data rows with a valid citation in
a cell, invalid or outside citations, sentences before and after the table.

Usage (from the project root, Ollama running):
    venv\\Scripts\\python evaluation\\comparison_format_check.py OUT.json [--runs 3]
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from config import settings  # noqa: E402
from src.core import citations  # noqa: E402
from src.core.rag import RagPipeline  # noqa: E402

QUESTIONS = ["whts the difference between olap and oltp", "data lake vs data warehouse",
             "is a data warehouse the same as a database?"]
CITE = re.compile(settings.NB04_CITATION_ID_PATTERN)


def cells(line: str) -> list[str]:
    """The cells of one markdown table line ("| a | b |" -> ["a", "b"])."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def sentences(text: str) -> int:
    """Rough sentence count of prose (citations removed), for the 'one sentence before, at most 2 after' rule."""
    prose = CITE.sub("", re.sub(r"\[[^\]]*\]", "", text)).strip()
    return len([s for s in re.split(r"(?<=[.!?])\s+", prose) if re.search(r"[A-Za-z]{3}", s)])


def analyse(answer: str, retrieved: list[str]) -> dict:
    """Table structure and citation checks for one answer."""
    lines = answer.splitlines()
    table_idx = [i for i, l in enumerate(lines) if l.strip().startswith("|")]
    blocks, current = [], []
    for i in table_idx:
        if current and i != current[-1] + 1:
            blocks.append(current)
            current = []
        current.append(i)
    if current:
        blocks.append(current)
    out = {"tables": len(blocks)}
    if not blocks:
        out.update(proper_table=False)
        return out
    rows = [lines[i] for i in blocks[0]]
    header, separator, data = rows[0], rows[1] if len(rows) > 1 else "", rows[2:]
    widths = {len(cells(r)) for r in [header] + data}
    sep_ok = bool(re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?", separator.strip()))
    cited_rows = 0
    valid_ids = set(retrieved)
    for r in data:
        ids = CITE.findall(r)
        cited_rows += any(i in valid_ids for i in ids)
    check = citations.check(answer, retrieved)
    out.update(header_cells=len(cells(header)), separator_ok=sep_ok, data_rows=len(data), consistent_width=len(widths) == 1,
               rows_with_valid_citation=cited_rows, valid_citations=check.counts["valid"], outside_set=check.outside_set,
               malformed=check.malformed,
               sentences_before=sentences("\n".join(lines[:blocks[0][0]])),
               sentences_after=sentences("\n".join(lines[blocks[0][-1] + 1:])))
    out["proper_table"] = (len(blocks) == 1 and out["header_cells"] >= 3 and sep_ok and 3 <= len(data) <= 5
                           and out["consistent_width"])
    out["citations_kept"] = cited_rows == len(data) and not check.outside_set and not check.malformed
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Live check of the comparison answer format.")
    parser.add_argument("out", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    pipeline = RagPipeline()
    records = []
    for arm in ("format_on", "format_off"):
        with mock.patch.object(settings, "COMPARISON_FORMAT_ENABLED", arm == "format_on"):
            for q in QUESTIONS:
                for run in range(1, args.runs + 1):
                    start = time.perf_counter()
                    result = pipeline.answer(q)
                    answer = "".join(result.answer_stream)
                    retrieved = [c.chunk_id for c in result.chunks]
                    rec = {"arm": arm, "question": q, "run": run, "route": result.routing.path,
                           "comparison_format": result.comparison_format, "fallback": result.used_fallback,
                           "chunks": retrieved, "seconds": round(time.perf_counter() - start, 1),
                           "words": len(answer.split()), "hit_cap": settings.ANSWER_CUT_NOTE in answer,
                           **analyse(answer, retrieved), "answer": answer}
                    records.append(rec)
                    print(f"{arm:10} {q[:42]:42} r{run} format={rec['comparison_format']!s:5} proper={rec['proper_table']!s:5} "
                          f"rows={rec.get('data_rows', '-')} cited_rows={rec.get('rows_with_valid_citation', '-')} "
                          f"before={rec.get('sentences_before', '-')} after={rec.get('sentences_after', '-')} "
                          f"words={rec['words']} {rec['seconds']}s", flush=True)
    meta = {"model": settings.DEV_LLM_MODEL, "runs": args.runs, "instruction": settings.COMPARISON_FORMAT_INSTRUCTION,
            "system_prompt_version": settings.SYSTEM_PROMPT_VERSION}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"meta": meta, "records": records}, indent=1, ensure_ascii=False), encoding="utf-8")
    for arm in ("format_on", "format_off"):
        sub = [r for r in records if r["arm"] == arm]
        print(f"{arm}: proper table {sum(r['proper_table'] for r in sub)}/{len(sub)}, "
              f"citations kept {sum(r.get('citations_kept', False) for r in sub)}/{len(sub)}")
    print("saved", args.out)


if __name__ == "__main__":
    main()
