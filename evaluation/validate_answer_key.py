"""
Validate evaluation/answer_key.json against the frozen question set and the chunk corpus.

Checks:
- exactly 100 answers, ids Q01-Q100 in order, each matching evaluation/test_questions.json;
- question_type is definitional, conceptual or procedural;
- every expected answer point has a chunk_id that exists in data/chunks/chunks.json and a boolean core flag;
- core points per question are 2-4, except full-set list questions (up to 6) and Q76 (1 core point, only
  because its source is a one-sentence definition; must be needs_review);
- outside_relevant_set is true exactly when the cited chunk is NOT in the question's relevant_chunk_ids,
  and any question using such a point is needs_review;
- needs_review questions have a review_note.
Prints warnings, the question-type counts, and exits with code 1 if any check fails.

Run from the project root:  venv\\Scripts\\python evaluation\\validate_answer_key.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

# Make the project root importable (config/) when the script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings  # noqa: E402

ANSWER_KEY_FILE = settings.TEST_QUESTIONS_FILE.parent / "answer_key.json"
EXPECTED_COUNT = 100
QUESTION_TYPES = {"definitional", "conceptual", "procedural"}
CORE_MIN, CORE_MAX = 2, 4
# Questions that ask for a complete set (all the steps / dimensions / types / layers ...): one core point per
# item, up to FULL_SET_CORE_MAX.
FULL_SET_LIST_QUESTIONS = {"Q22", "Q26", "Q49", "Q51", "Q77", "Q81", "Q87", "Q90", "Q91", "Q98"}
FULL_SET_CORE_MAX = 6
# Questions allowed a single core point (one-sentence source); they must be needs_review.
SINGLE_CORE_QUESTIONS = {"Q76"}


def load_json(path: Path):
    """Read a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def check_core_count(qid: str, n_core: int, needs_review: bool, errors: list[str]) -> None:
    """Apply the core-point rules, including the two documented exceptions."""
    if qid in SINGLE_CORE_QUESTIONS:
        if n_core != 1 or not needs_review:
            errors.append(f"{qid}: must have exactly 1 core point and be needs_review (has {n_core})")
        return
    upper = FULL_SET_CORE_MAX if qid in FULL_SET_LIST_QUESTIONS else CORE_MAX
    if not CORE_MIN <= n_core <= upper:
        errors.append(f"{qid}: {n_core} core points (allowed {CORE_MIN}-{upper})")


def validate() -> int:
    """Run all checks; print the results and return the process exit code."""
    key = load_json(ANSWER_KEY_FILE)
    questions = load_json(settings.TEST_QUESTIONS_FILE)["questions"]
    chunk_ids = {c["chunk_id"] for c in load_json(settings.CHUNKS_FILE)}
    by_id = {q["id"]: q for q in questions}
    answers = key["answers"]
    errors, warnings = [], []

    expected_ids = [f"Q{i:02d}" for i in range(1, EXPECTED_COUNT + 1)]
    ids = [a["id"] for a in answers]
    if len(answers) != EXPECTED_COUNT:
        errors.append(f"expected {EXPECTED_COUNT} answers, found {len(answers)}")
    if ids != expected_ids:
        missing = sorted(set(expected_ids) - set(ids))
        extra = sorted(set(ids) - set(expected_ids))
        errors.append(f"ids are not Q01-Q100 in order (missing {missing}, unexpected {extra})")

    for a in answers:
        qid = a["id"]
        q = by_id.get(qid)
        if q is None:
            errors.append(f"{qid}: not in {settings.TEST_QUESTIONS_FILE.name}")
            continue
        if a.get("question") != q["question"]:
            errors.append(f"{qid}: question text differs from {settings.TEST_QUESTIONS_FILE.name}")
        if a.get("question_type") not in QUESTION_TYPES:
            errors.append(f"{qid}: bad question_type {a.get('question_type')!r}")

        points = a.get("expected_answer_points", [])
        if not points:
            errors.append(f"{qid}: no expected_answer_points")
        uses_outside = False
        for i, p in enumerate(points, 1):
            cid = p.get("chunk_id")
            if cid not in chunk_ids:
                errors.append(f"{qid} point {i}: chunk_id {cid!r} not in {settings.CHUNKS_FILE.name}")
            if not isinstance(p.get("core"), bool):
                errors.append(f"{qid} point {i}: missing boolean core flag")
            outside_flag = p.get("outside_relevant_set", False) is True
            outside_actual = cid not in q["relevant_chunk_ids"]
            if outside_flag != outside_actual:
                errors.append(f"{qid} point {i}: outside_relevant_set={outside_flag} but chunk {cid} is "
                              f"{'not ' if outside_actual else ''}in relevant_chunk_ids")
            uses_outside = uses_outside or outside_actual

        n_core = sum(1 for p in points if p.get("core") is True)
        check_core_count(qid, n_core, a.get("needs_review", False), errors)
        if n_core > CORE_MAX and qid in FULL_SET_LIST_QUESTIONS:
            warnings.append(f"{qid}: {n_core} core points (full-set list question)")
        if uses_outside and not a.get("needs_review"):
            errors.append(f"{qid}: cites a chunk outside its relevant set but is not needs_review")
        if a.get("needs_review") and not a.get("review_note"):
            errors.append(f"{qid}: needs_review without review_note")

    print(f"Answer key: {ANSWER_KEY_FILE.relative_to(PROJECT_ROOT)}")
    print(f"Status:     {key['meta'].get('status', '(none)')}")
    print(f"Answers:    {len(answers)}  |  points: {sum(len(a['expected_answer_points']) for a in answers)}  |  "
          f"needs_review: {sum(1 for a in answers if a.get('needs_review'))}")
    types = Counter(a.get("question_type") for a in answers)
    print("Question types: " + ", ".join(f"{t} {types.get(t, 0)}" for t in sorted(QUESTION_TYPES)))
    for w in warnings:
        print(f"  note: {w}")
    if errors:
        print(f"\nFAILED - {len(errors)} error(s):")
        for e in errors:
            print(f"  {e}")
        return 1
    print("\nOK - all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(validate())
