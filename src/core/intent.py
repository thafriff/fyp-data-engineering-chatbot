"""
Intent router: decides, before retrieval, whether a message is small talk, unclear, emotional, or a real question.

Order (cheapest first):
1. Rule-based small talk (whole message is a greeting / thanks / goodbye / acknowledgement) -> fixed reply.
2. Rule-based unclear input (gibberish, a stray character) -> fixed "please rephrase" reply.
3. Obvious question (ends with "?" or starts with a question word) -> retrieval, no LLM call.
4. LLM label classifier for what is left. The model may only output one label from settings.INTENT_LABELS;
   an EMOTIONAL_* label gets a canned reply chosen here from settings.EMOTIONAL_RESPONSES. Any other output,
   an error or a timeout is treated as REAL_QUESTION.
Every "question" goes through retrieval and the similarity-threshold fallback, so no path lets the model answer
from its own knowledge. All patterns, prompts and replies live in config/settings.py.
"""

import re
from dataclasses import dataclass

from config import settings
from src.core import llm

# Compiled once: each small-talk pattern must match the WHOLE normalised message.
_SMALL_TALK = {name: re.compile(rf"^(?:{pattern})$") for name, pattern in settings.SMALL_TALK_PATTERNS.items()}


@dataclass
class Intent:
    """The routing decision for one message."""
    kind: str                   # "small_talk", "unclear", "emotional" or "question"
    reply: str | None = None    # fixed reply for small_talk / unclear / emotional; None for questions
    path: str = "pattern"       # how it was decided: "pattern", "question_heuristic" or "classifier"
    label: str | None = None    # raw classifier output (only when path == "classifier")


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


def _is_obvious_question(message: str, text: str) -> bool:
    """True if the message ends with "?" or starts with a question/request word (no LLM call needed)."""
    return message.strip().endswith("?") or text.split()[0] in settings.QUESTION_START_WORDS


def classify(message: str) -> str:
    """Ask the LLM for ONE label from settings.INTENT_LABELS. Returns the label, or the fallback label
    (REAL_QUESTION) if the output is not exactly a known label or the call fails."""
    messages = [{"role": "system", "content": settings.INTENT_CLASSIFIER_PROMPT}]
    for example, label in settings.INTENT_CLASSIFIER_EXAMPLES:
        messages += [{"role": "user", "content": example}, {"role": "assistant", "content": label}]
    messages.append({"role": "user", "content": message})
    try:
        output = llm.complete(messages, max_tokens=settings.INTENT_CLASSIFIER_MAX_TOKENS,
                              timeout=settings.INTENT_CLASSIFIER_TIMEOUT_SECONDS).strip()
    except Exception:                 # server down, timeout, bad response: fail safe to retrieval
        return settings.INTENT_FALLBACK_LABEL
    return output if output in settings.INTENT_LABELS else settings.INTENT_FALLBACK_LABEL


def route(message: str) -> Intent:
    """Classify the message: small talk, unclear and emotional input get a fixed reply; anything else is a
    question that goes to retrieval."""
    text = _normalise(message)

    # Small talk first, so one-letter phrases such as "k" are not caught by the unclear check below.
    for name, pattern in _SMALL_TALK.items():
        if pattern.match(text):
            return Intent("small_talk", settings.SMALL_TALK_RESPONSES[name])

    if sum(ch.isalnum() for ch in text) < settings.UNCLEAR_MIN_ALNUM_CHARS:
        return Intent("unclear", settings.UNCLEAR_RESPONSE)
    if " " not in text and _is_keyboard_mash(text):
        return Intent("unclear", settings.UNCLEAR_RESPONSE)

    if not settings.INTENT_CLASSIFIER_ENABLED or _is_obvious_question(message, text):
        return Intent("question", path="question_heuristic")

    label = classify(message)
    if label in settings.EMOTIONAL_RESPONSES:
        return Intent("emotional", settings.EMOTIONAL_RESPONSES[label], path="classifier", label=label)
    return Intent("question", path="classifier", label=label)
