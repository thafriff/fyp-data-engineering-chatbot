"""Tally of the app-layer format checks (2026-11-26): files written by evaluation/comparison_format_check.py.
Usage: python format_tally.py FILE [FILE ...]"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
SOURCE_SENTENCE = re.compile(r"^\W*(?:based on|according to|as (?:stated|described|explained|shown) in|in|from|per)?\s*the\s+"
                             r"(?:provided\s+|given\s+)?(?:course\s+materials?|excerpts?)\b", re.IGNORECASE)
SIMPLE_START = re.compile(r"^\W*in simple terms", re.IGNORECASE)


def source_sentences(answer: str) -> int:
    """Sentences or lines that start with a phrase about the source ("The course materials highlight ...")."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", answer)
    return sum(bool(SOURCE_SENTENCE.match(p.strip("|*- "))) for p in parts if p.strip())


for name in sys.argv[1:]:
    records = json.loads(Path(name).read_text(encoding="utf-8"))["records"]
    cmp_runs = [r for r in records if r["comparison_format"]]
    print(f"== {Path(name).name}: {len(records)} runs, {len(cmp_runs)} with the comparison format")
    print(f"  answer STARTS with a source phrase: {sum(r['opens_with_source_phrase'] for r in records)}/{len(records)}")
    print(f"  answers with ANY sentence starting with a source phrase: {sum(source_sentences(r['answer']) > 0 for r in records)}"
          f"/{len(records)} (sentences: {sum(source_sentences(r['answer']) for r in records)})")
    if cmp_runs:
        n = len(cmp_runs)
        opens_simple = sum(bool(SIMPLE_START.match(r["first_line"])) for r in cmp_runs)
        own_line = sum(r["simple_terms_line"] and r["bullets"] > 0 for r in cmp_runs)
        all_cited = sum(r.get("rows_with_valid_citation", -1) == r.get("data_rows") for r in cmp_runs)
        rows = [f"{r.get('rows_with_valid_citation')}/{r.get('data_rows')}" for r in cmp_runs]
        print(f"  opens with 'In simple terms': {opens_simple}/{n}")
        print(f"  'In simple terms:' on its own line then bullets: {own_line}/{n}")
        print(f"  one bullet per item, each <= 25 words, nothing else before the table: {sum(r['simple_terms_ok'] for r in cmp_runs)}/{n}")
        print(f"  proper table: {sum(r['proper_table'] for r in cmp_runs)}/{n}")
        print(f"  DECISION RULE ('In simple terms:' + one bullet per item + proper table): "
              f"{sum(r.get('rule_pass', False) for r in cmp_runs)}/{n}")
        print(f"  every table row cited: {all_cited}/{n}; rows cited per run: {rows}")
        print(f"  at most 1 sentence after the table: {sum(r.get('sentences_after', 99) <= 1 for r in cmp_runs)}/{n}")
    print(f"  answers with valid citations: {sum(r['valid_citations'] > 0 for r in records)}/{len(records)}; "
          f"malformed or outside: {[(r['question'][:12], r['run'], r['malformed'], r['outside_set']) for r in records if r['malformed'] or r['outside_set']]}")
