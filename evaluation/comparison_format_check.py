"""
Live check of the app-layer answer formats (DEV PROTOTYPE; not part of notebook 04): the comparison format
(src/core/answer_format.py) and the opening rule for every answer. Each question goes through the real chat pipeline
(RagPipeline.answer: routing, search text, retrieval, threshold, generation) with the default local model.

--arms: "app" runs the current settings once; "format_on,format_off" also runs a control with the comparison format switched
off (settings.COMPARISON_FORMAT_ENABLED patched in-process; settings.py is not edited).

Measured per run:
- PROPER TABLE: exactly one markdown table, a header of at least 3 cells (aspect + 2 items), a separator row, 3 to 5 data rows,
  the same number of cells in every row; data rows with a valid citation in a cell; invalid or outside citations;
- SIMPLE TERMS: the answer opens with the line "In simple terms:", then one bullet per compared item (as many bullets as the
  table has item columns, or 2 without a table), each a single line of at most 25 words, with no other prose before the table;
- OPENING: whether the first line starts with "The course materials", "According to the course materials", "Based on the ..."
  or a similar phrase about the source;
- sentences before and after the table.

Results are saved after every run (meta "finished" is false until the last one), so a stopped run keeps its data.

Usage (from the project root, Ollama running):
    venv\\Scripts\\python evaluation\\comparison_format_check.py OUT.json [--runs 3] [--arms app] [--questions "q1" "q2" ...]
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
from src.core import citations, llm  # noqa: E402
from src.core.rag import RagPipeline  # noqa: E402

DEFAULT_QUESTIONS = ["whts the difference between olap and oltp", "data lake vs data warehouse",
                     "is a data warehouse the same as a database?"]
CITE = re.compile(settings.NB04_CITATION_ID_PATTERN)
BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
SOURCE_OPENING = re.compile(r"^\W*(?:based on|according to|as (?:stated|described|explained|shown) in|in|from|per)?\s*the\s+"
                            r"(?:provided\s+|given\s+)?(?:course\s+materials?|excerpts?|materials|text|context)\b", re.IGNORECASE)
SIMPLE_TERMS = re.compile(r"^\W*in simple terms\W*:?\W*$", re.IGNORECASE)


def cells(line: str) -> list[str]:
    """The cells of one markdown table line ("| a | b |" -> ["a", "b"])."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def sentences(text: str) -> int:
    """Rough sentence count of prose (citations removed)."""
    prose = CITE.sub("", re.sub(r"\[[^\]]*\]", "", text)).strip()
    return len([s for s in re.split(r"(?<=[.!?])\s+", prose) if re.search(r"[A-Za-z]{3}", s)])


def words(text: str) -> int:
    """Words of a line without citations or markdown emphasis."""
    return len(re.sub(r"\[[^\]]*\]|\*", " ", text).split())


def analyse(answer: str, retrieved: list[str]) -> dict:
    """Opening, simple-terms bullets, table structure and citation checks for one answer."""
    lines = answer.splitlines()
    nonempty = [l for l in lines if l.strip()]
    first = nonempty[0] if nonempty else ""
    out = {"first_line": first[:120], "opens_with_source_phrase": bool(SOURCE_OPENING.match(first))}
    check = citations.check(answer, retrieved)
    out.update(valid_citations=check.counts["valid"], outside_set=check.outside_set, malformed=check.malformed)
    table_idx = [i for i, l in enumerate(lines) if l.strip().startswith("|")]
    blocks, current = [], []
    for i in table_idx:
        if current and i != current[-1] + 1:
            blocks.append(current)
            current = []
        current.append(i)
    if current:
        blocks.append(current)
    out["tables"] = len(blocks)
    pre = lines[:blocks[0][0]] if blocks else lines
    # "In simple terms:" block: the first non-empty line, then bullets, then nothing else before the table.
    pre_nonempty = [l for l in pre if l.strip()]
    has_line = bool(pre_nonempty) and bool(SIMPLE_TERMS.match(pre_nonempty[0]))
    bullets, extra = [], []
    for l in pre_nonempty[1:] if has_line else []:
        (bullets if BULLET.match(l) and not extra else extra).append(l)
    out.update(simple_terms_line=has_line, bullets=len(bullets), max_bullet_words=max((words(b) for b in bullets), default=0),
               prose_between_bullets_and_table=len(extra))
    if not blocks:
        out.update(proper_table=False, simple_terms_ok=has_line and len(bullets) == 2 and out["max_bullet_words"] <= 25,
                   rule_pass=False)
        return out
    rows = [lines[i] for i in blocks[0]]
    header, separator, data = rows[0], rows[1] if len(rows) > 1 else "", rows[2:]
    widths = {len(cells(r)) for r in [header] + data}
    sep_ok = bool(re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?", separator.strip()))
    cited_rows = sum(any(i in set(retrieved) for i in CITE.findall(r)) for r in data)
    items = len(cells(header)) - 1
    out.update(header_cells=len(cells(header)), separator_ok=sep_ok, data_rows=len(data), consistent_width=len(widths) == 1,
               rows_with_valid_citation=cited_rows, sentences_before=sentences("\n".join(pre)),
               sentences_after=sentences("\n".join(lines[blocks[0][-1] + 1:])))
    out["proper_table"] = (len(blocks) == 1 and out["header_cells"] >= 3 and sep_ok and 3 <= len(data) <= 5
                           and out["consistent_width"])
    out["citations_kept"] = cited_rows == len(data) and not check.outside_set and not check.malformed
    out["simple_terms_ok"] = (has_line and len(bullets) == items and out["max_bullet_words"] <= 25 and not extra)
    # The author's decision rule (2026-11-26): "In simple terms:" + one bullet per item + a proper table.
    out["rule_pass"] = has_line and len(bullets) == items and out["proper_table"]
    return out


def save(path: Path, meta: dict, records: list[dict]) -> None:
    """Write the results so far (atomic: temp file, then replace). Called after every run, so a stopped run keeps its data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"meta": meta, "records": records}, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live check of the app-layer answer formats.")
    parser.add_argument("out", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--arms", default="format_on,format_off")
    parser.add_argument("--questions", nargs="+", default=DEFAULT_QUESTIONS)
    parser.add_argument("--model", default=None, help="answer model; default settings.DEV_LLM_MODEL; a cloud model goes "
                                                       "through the app's own cloud path (llm.cloud_generate)")
    args = parser.parse_args()
    pipeline = RagPipeline()
    records = []
    meta = {"model": args.model or settings.DEV_LLM_MODEL, "runs": args.runs, "arms": args.arms, "questions": args.questions,
            "comparison_instruction": settings.COMPARISON_FORMAT_INSTRUCTION,
            "opening_instruction": getattr(settings, "ANSWER_OPENING_INSTRUCTION", None),
            "system_prompt_version": settings.SYSTEM_PROMPT_VERSION, "finished": False}
    for arm in args.arms.split(","):
        enabled = settings.COMPARISON_FORMAT_ENABLED if arm == "app" else arm == "format_on"
        with mock.patch.object(settings, "COMPARISON_FORMAT_ENABLED", enabled):
            for q in args.questions:
                for run in range(1, args.runs + 1):
                    start = time.perf_counter()
                    result = pipeline.answer(q, model=args.model)
                    cloud, error = {}, None
                    if result.cloud_call:
                        try:
                            reply = result.cloud_call()
                            answer = reply.text
                            cloud = {"input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens,
                                     "latency_seconds": reply.latency_seconds, "retries": reply.retries,
                                     "cost_usd_list_not_billed_on_free_tier": reply.cost_usd_list}
                        except llm.CloudError as exc:
                            answer, error = "", str(exc)
                    else:
                        answer = "".join(result.answer_stream)
                    retrieved = [c.chunk_id for c in result.chunks]
                    rec = {"arm": arm, "model": result.model or args.model or settings.DEV_LLM_MODEL, "error": error, "cloud": cloud,
                           "question": q, "run": run, "route": result.routing.path,
                           "comparison_format": result.comparison_format, "fallback": result.used_fallback,
                           "chunks": retrieved, "seconds": round(time.perf_counter() - start, 1),
                           "words": len(answer.split()), "hit_cap": settings.ANSWER_CUT_NOTE in answer,
                           **analyse(answer, retrieved), "answer": answer}
                    records.append(rec)
                    save(args.out, meta, records)          # keep partial data if the run is stopped
                    print(f"{arm:10} {q[:44]:44} r{run} cmp={rec['comparison_format']!s:5} source_opening="
                          f"{rec['opens_with_source_phrase']!s:5} simple={rec['simple_terms_ok']!s:5} "
                          f"bullets={rec['bullets']} rule={rec['rule_pass']!s:5} proper={rec['proper_table']!s:5} "
                          f"rows={rec.get('data_rows', '-')} "
                          f"cited_rows={rec.get('rows_with_valid_citation', '-')} after={rec.get('sentences_after', '-')} "
                          f"cites={rec['valid_citations']} words={rec['words']} {rec['seconds']}s", flush=True)
    meta["finished"] = True
    save(args.out, meta, records)
    print("saved", args.out)


if __name__ == "__main__":
    main()
