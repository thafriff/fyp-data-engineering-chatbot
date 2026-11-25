"""
Removes leading greetings and politeness from the text used for the retrieval search.

A greeting in front of a short question dilutes its embedding: "hi what is olap" scored 0.694, below the 0.70 relevance
threshold, while "what is olap" scores 0.708, so the student got the "not in the course materials" fallback. Like spelling
correction, this changes only the text that is EMBEDDED for the search; the original message is kept for display, history
and the LLM prompt. Only the start of the message is touched, never the middle or the end. Request verbs such as "explain"
are kept, because they carry meaning: "explain ETL" scores higher than "ETL" alone.

Called only through RagPipeline.search_text, the single function that builds the search text.
The phrase lists live in config/settings.py (SEARCH_LEADING_*, SEARCH_REQUEST_VERBS, SEARCH_STRIP_PROTECTED_STARTS).
"""

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from config import settings
from src.core.spelling import WORD


def _alternatives(phrases: list[str]) -> str:
    """One regex alternation for whole phrases: longest first, any run of spaces between words, either apostrophe."""
    def one(phrase: str) -> str:
        return r"\s+".join(re.escape(word).replace("'", "['’]") for word in phrase.split())
    return "|".join(one(p) for p in sorted(phrases, key=len, reverse=True))


# A phrase counts only as a whole word followed by space, punctuation or the end, so "hive", "high", "hi-res" and
# "pleased" are never touched. The spaces and punctuation after it, including a dash ("hiya - what is ..."), go with it.
# "can you" and the like count only when a request verb follows, optionally after "please".
_OPENER = (rf"(?:{_alternatives(settings.SEARCH_LEADING_REQUEST_OPENERS)})"
           rf"(?=\s+(?:please\s+)?(?:{_alternatives(settings.SEARCH_REQUEST_VERBS)})\b)")
_LEADING = re.compile(
    rf"^\s*(?!(?:{_alternatives(settings.SEARCH_STRIP_PROTECTED_STARTS)})\b)"
    rf"(?:{_alternatives(settings.SEARCH_LEADING_GREETINGS + settings.SEARCH_LEADING_FILLER)}|{_OPENER})"
    r"(?=[\s,.!?:;]|$)[\s,.!?:;\-–—]*",
    re.IGNORECASE)


def remove_leading(question: str) -> str:
    """The question with its leading greetings and politeness removed, or as typed if that step is switched off."""
    return strip_leading_filler(question) if settings.SEARCH_STRIP_LEADING_ENABLED else question


_BRITISH = [(re.compile(pattern), replacement) for pattern, replacement in settings.SPELLING_BRITISH_VARIANTS]
# Words never announced in the spelling hint (settings.SPELLING_HINT_EXTRA_COMMON_WORDS explains why).
_COMMON_WORDS = frozenset(ENGLISH_STOP_WORDS) | {w.lower() for w in settings.SPELLING_HINT_EXTRA_COMMON_WORDS}


def is_common_word(word: str) -> bool:
    """True for a common function or question word ("what", "the", "how", "difference"), which the hint never names."""
    return word.lower() in _COMMON_WORDS


def is_british_variant(typed: str, corrected: str) -> bool:
    """True when the typed word is the British spelling of the corrected word ("Normalise" -> "normalize", "colour" ->
    "color"), so the correction is not a typo. Each rule is a fixed British -> American rewrite; the word counts as a variant
    only if one rewrite, or all of them together, gives exactly the corrected word."""
    word, target = typed.lower(), corrected.lower()
    rewrites = {pattern.sub(replacement, word) for pattern, replacement in _BRITISH}
    combined = word
    for pattern, replacement in _BRITISH:
        combined = pattern.sub(replacement, combined)
    return target in rewrites | {combined}


def spelling_corrections(question: str, search_text: str, excerpts: list[str]) -> list[tuple[str, str]]:
    """The (typed, corrected) word pairs to show the model as a spelling hint, or [] for no hint.
    Spelling correction replaces words one for one, so the typed and the corrected words line up. British spellings are not
    typos and are left out ("Normalise" -> "normalize"), and so are corrections to common words ("wht" -> "what"). A hint is
    given only when EVERY remaining changed word's correction has at least SPELLING_HINT_MIN_LETTERS letters and appears, as a
    whole word, in the retrieved excerpts: "olpa" -> "olap" qualifies, "dat" -> "at" does not. Greeting removal alone changes no word, so it never produces a hint. Works from stored
    values, so the parity check can rebuild a frozen prompt without the spelling corrector."""
    typed, corrected = WORD.findall(remove_leading(question)), WORD.findall(search_text)
    if len(typed) != len(corrected):
        return []
    # Only content and technical words are announced: a common-word correction ("wht" -> "what") is left out here, so it
    # neither appears in the hint nor stops the hint for another word ("wht is olpa" -> only "olpa" -> "olap").
    pairs = [(t, c) for t, c in zip(typed, corrected)
             if t != c and not is_british_variant(t, c) and not is_common_word(c)]
    text = "\n".join(excerpts).lower()
    qualifies = all(len(c) >= settings.SPELLING_HINT_MIN_LETTERS and re.search(rf"\b{re.escape(c.lower())}\b", text)
                    for _, c in pairs)
    return pairs if pairs and qualifies else []


def strip_leading_filler(text: str) -> str:
    """Return text without its leading greetings and filler, removed repeatedly ("hi, please explain ..."). If nothing
    would be left, the text is returned as typed."""
    stripped = text
    while (match := _LEADING.match(stripped)) and match.end():
        stripped = stripped[match.end():]
    return stripped if stripped.strip() else text
