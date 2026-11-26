"""
App-layer answer formats, appended to the user message after the answer reminder:
- the OPENING RULE for every generated answer: start with the answer itself, never with "The course materials ...";
- the COMPARISON FORMAT for questions that compare things ("difference between", "vs", "same as", "which ... or ..."): an
  "In simple terms:" line with one plain-English bullet per item, then a markdown table (one column per item, one row per
  aspect, 3 to 5 rows, citations in the cells), then at most 1 sentence.

SYSTEM FEATURES of the chat prototype, evaluated in notebook 05, like coach mode (guardrail.py). They are NOT part of the
notebook 04 comparison: the runners never route, and llm.build_messages() is untouched, so every frozen prompt stays
byte-identical. The rules and the instructions live in config/settings.py (COMPARISON_*, ANSWER_OPENING_*).
"""

import re
from collections.abc import Callable

from config import settings
from src.core import guardrail

_PATTERNS = [re.compile(p, re.IGNORECASE) for p in settings.COMPARISON_PATTERNS]


def detect_comparison(*texts: str) -> str | None:
    """The first comparison phrase found in any of the texts (the message as typed, then its cleaned search text), or None.
    Returns None when the feature is switched off."""
    if not settings.COMPARISON_FORMAT_ENABLED:
        return None
    for text in texts:
        for pattern in _PATTERNS:
            if text and (match := pattern.search(text)):
                return match.group(0)
    return None


def _append(messages: list[dict], instruction: str) -> list[dict]:
    """The messages with one instruction appended to the user message; the system prompt is unchanged."""
    system, user = messages
    return [system, {"role": user["role"], "content": f"{user['content']}\n\n{instruction}"}]


def comparison_messages(messages: list[dict]) -> list[dict]:
    """The normal messages from llm.build_messages(), with the comparison format instruction appended to the user message."""
    return _append(messages, settings.COMPARISON_FORMAT_INSTRUCTION)


def opening_messages(messages: list[dict]) -> list[dict]:
    """The messages with the opening rule appended to the user message (for every generated answer in the app)."""
    return _append(messages, settings.ANSWER_OPENING_INSTRUCTION)


def app_transform(coach: bool = False, comparison: bool = False) -> Callable[[list[dict]], list[dict]] | None:
    """The function the pipeline passes to the generator: the opening rule first (if switched on), then the coach instruction
    for a task request or the comparison instruction for a comparison (coach wins). None when nothing is to be appended."""
    steps = [opening_messages] if settings.ANSWER_OPENING_RULE_ENABLED else []
    if coach:
        steps.append(guardrail.coach_messages)
    elif comparison:
        steps.append(comparison_messages)
    if not steps:
        return None

    def transform(messages: list[dict]) -> list[dict]:
        for step in steps:
            messages = step(messages)
        return messages
    return transform
