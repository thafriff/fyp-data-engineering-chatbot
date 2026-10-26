"""
Intent router: decides, before retrieval, whether a message is small talk, unclear, or a real question.

Rule-based on purpose (no LLM), so it can never answer a question from the model's own knowledge. Only a
message that is entirely a greeting/thanks/goodbye phrase, or that is gibberish / a stray character, is
handled here. Every other message is a "question" and goes through retrieval and the similarity-threshold
fallback exactly as before. All patterns and replies live in config/settings.py.
"""

import re
from dataclasses import dataclass

from config import settings

# Compiled once: each small-talk pattern must match the WHOLE normalised message.
_SMALL_TALK = {name: re.compile(rf"^(?:{pattern})$") for name, pattern in settings.SMALL_TALK_PATTERNS.items()}


@dataclass
class Intent:
    """The routing decision for one message."""
    kind: str                   # "small_talk", "unclear" or "question"
    reply: str | None = None    # fixed reply for small_talk / unclear; None for questions


def _normalise(message: str) -> str:
    """Lower case, replace punctuation with spaces and collapse whitespace ("Hi!!  there?" -> "hi there")."""
    text = re.sub(r"[^\w\s]", " ", message.lower().replace("'", ""))
    return " ".join(text.split())


def _is_keyboard_mash(word: str) -> bool:
    """True for one-word mashes such as "asdf", "qwerty" or "aaaa"."""
    if len(word) < settings.KEYBOARD_MASH_MIN_LENGTH:
        return False
    if len(set(word)) == 1:
        return True
    return any(word in row for row in settings.KEYBOARD_ROWS)


def route(message: str) -> Intent:
    """Classify the message: small talk and unclear input get a fixed reply; anything else is a question."""
    text = _normalise(message)

    # Small talk first, so one-letter phrases such as "k" are not caught by the unclear check below.
    for name, pattern in _SMALL_TALK.items():
        if pattern.match(text):
            return Intent("small_talk", settings.SMALL_TALK_RESPONSES[name])

    if sum(ch.isalnum() for ch in text) < settings.UNCLEAR_MIN_ALNUM_CHARS:
        return Intent("unclear", settings.UNCLEAR_RESPONSE)
    if " " not in text and _is_keyboard_mash(text):
        return Intent("unclear", settings.UNCLEAR_RESPONSE)

    return Intent("question")
