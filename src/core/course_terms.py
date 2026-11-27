"""
Course terms: words that belong to the course rather than to everyday English, used by the multi-turn short-follow-up rescue
(conversation.py). A word is a course term when it is a protected domain term (settings.SPELL_PROTECTED_TERMS), or when it is at
least settings.COURSE_TERM_MIN_RATIO times more frequent in the course chunks than in general English (pyspellchecker's English
word-frequency list, offline) and appears at least settings.COURSE_TERM_MIN_COUNT times in the chunks.

The counts are built once, on first use (about half a second), and only in a conversation, when a follow-up falls below the
similarity threshold. Single-turn chat never loads them.
"""

import json
from collections import Counter
from functools import lru_cache

from spellchecker import SpellChecker

from config import settings
from src.core.spelling import WORD


@lru_cache(maxsize=1)
def _frequencies() -> tuple[Counter, int, object, int]:
    """(course word counts, course total, English word frequencies, English total), built once."""
    chunks = json.loads(settings.CHUNKS_FILE.read_text(encoding="utf-8"))
    course = Counter(w.lower() for c in chunks for w in WORD.findall(c["text"]) if len(w) > 1)
    english = SpellChecker().word_frequency
    return course, sum(course.values()), english, english.total_words


def is_course_term(word: str) -> bool:
    """True for a protected domain term, or a word far more frequent in the course than in general English."""
    word = (word or "").lower()
    if word in settings.SPELL_PROTECTED_TERMS:
        return True
    course, course_total, english, english_total = _frequencies()
    if course[word] < settings.COURSE_TERM_MIN_COUNT:
        return False
    if not english[word]:
        return True                                       # frequent in the course, unknown to general English
    return (course[word] / course_total) / (english[word] / english_total) >= settings.COURSE_TERM_MIN_RATIO


def all_course_terms(words: list[str]) -> bool:
    """True when there is at least one word and every word is a course term."""
    return bool(words) and all(is_course_term(w) for w in words)
