"""
Comparison answer format: detects questions that compare things ("difference between", "vs", "same as", "which ... or ...")
and asks for a fixed layout: a one-sentence direct answer, a markdown table (one column per item, one row per aspect, 3 to 5
rows, citations in the cells), then at most 2 sentences.

A SYSTEM FEATURE of the chat prototype, evaluated in notebook 05, like coach mode (guardrail.py). It is NOT part of the
notebook 04 comparison: the runners never route, and llm.build_messages() is untouched, so every frozen prompt stays
byte-identical. The rules and the instruction live in config/settings.py (COMPARISON_*).
"""

import re

from config import settings

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


def comparison_messages(messages: list[dict]) -> list[dict]:
    """The normal messages from llm.build_messages(), with the comparison format instruction appended to the user message.
    The system prompt and everything before the instruction are unchanged."""
    system, user = messages
    return [system, {"role": user["role"], "content": f"{user['content']}\n\n{settings.COMPARISON_FORMAT_INSTRUCTION}"}]
