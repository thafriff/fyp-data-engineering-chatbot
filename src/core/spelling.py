"""
Spelling corrector for the retrieval query.

Typos such as "whst is olap" lower the embedding similarity enough to fall below the relevance threshold, so the
question wrongly gets the "not in the course materials" fallback. This module corrects obvious misspellings in the
text that is EMBEDDED for the search; the original message is kept for display, history and the LLM prompt.

Offline: pyspellchecker's English word list, extended with every word that occurs in the course chunks (weighted up),
so domain terms are treated as correct and typos of them are corrected towards the course term.
"""

import json
import re
from collections import Counter

from spellchecker import SpellChecker

from config import settings

# Letters-only words (an apostrophe inside is allowed: "what's"); everything else is left exactly as typed.
_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


class SpellCorrector:
    """Built once (a few seconds: reads the course vocabulary), then corrects queries in about a millisecond."""

    def __init__(self):
        """Load the English dictionary and add the course vocabulary and the protected domain terms."""
        self.checker = SpellChecker(distance=settings.SPELL_MAX_EDIT_DISTANCE)
        chunks = json.loads(settings.CHUNKS_FILE.read_text(encoding="utf-8"))
        course_counts = Counter(w.lower() for c in chunks for w in _WORD.findall(c["text"]) if len(w) > 1)
        self.checker.word_frequency.load_json(
            {w: n * settings.SPELL_COURSE_WORD_WEIGHT for w, n in course_counts.items()})
        self.checker.word_frequency.load_words(settings.SPELL_PROTECTED_TERMS)

    def correct(self, text: str) -> str:
        """Return text with unknown words replaced by their most likely correction; known words, short words,
        words typed in ALL CAPS (acronyms such as CSS, FIFA) and words with no good candidate are left unchanged."""
        def fix(match: re.Match) -> str:
            word = match.group(0)
            lower = word.lower()
            if len(lower) < settings.SPELL_MIN_WORD_LENGTH or word.isupper() or lower in self.checker:
                return word
            best = self.checker.correction(lower)
            return best if best else word

        return _WORD.sub(fix, text)
