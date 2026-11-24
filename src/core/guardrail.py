"""
Code-level guardrail router: detects "do my assignment / give me the full solution" requests and answers them in coach mode.

A SYSTEM FEATURE of the chat prototype, evaluated in notebook 05. It is NOT part of the notebook 04 comparison: the runners
never route, and llm.build_messages() is untouched, so every frozen prompt stays byte-identical.

Why it exists: with the frozen SYSTEM_PROMPT alone, llama3.2:3b handed over a complete solution on 7 of 9 standard guardrail
probes (Report TODO 12). Prompting alone does not hold on a small model, so the restriction is enforced in code:
1. detection by rules (settings.GUARDRAIL_*), on the cleaned search text;
2. a coach instruction appended to the normal prompt (the normal messages are built first and left as they are);
3. a hard cap on fenced code blocks in the answer, which holds even when the model ignores the instruction.

The rule lists and the held-out evaluation set (evaluation/guardrail_router_cases.json) are separate on purpose: the set was
written and hashed before this code and is never tuned to it.
"""

import re
from dataclasses import dataclass, field

from config import settings


def _phrase_pattern(phrases: list[str]) -> re.Pattern:
    """Whole-word, case-insensitive match of any phrase; "lab " with a trailing space is kept as typed."""
    parts = []
    for phrase in sorted(phrases, key=len, reverse=True):
        p = re.escape(phrase.strip()).replace(r"\ ", r"\s+")
        parts.append(rf"\b{p}\b" if phrase == phrase.strip() else rf"\b{p}(?=\s)")
    return re.compile("|".join(parts), re.IGNORECASE)


_VERBS = _phrase_pattern(settings.GUARDRAIL_PRODUCTION_VERBS)
_DELIVERABLES = _phrase_pattern(settings.GUARDRAIL_DELIVERABLES)
_ASSESSED = _phrase_pattern(settings.GUARDRAIL_ASSESSED_SIGNALS)
_COMPLETE = _phrase_pattern(settings.GUARDRAIL_COMPLETE_SIGNALS)
_EXAMPLE = _phrase_pattern(settings.GUARDRAIL_EXAMPLE_WORDS)
_FENCE = re.compile(r"```[^\n]*\n(.*?)(```|\Z)", re.DOTALL)


@dataclass
class TaskDecision:
    """Why a message was or was not treated as a task request, for the UI and the evaluation."""
    is_task: bool
    verb: str | None = None
    deliverable: str | None = None
    assessed: str | None = None
    complete: str | None = None
    example: str | None = None
    reasons: list[str] = field(default_factory=list)


def _overlaps(a: re.Match, b: re.Match) -> bool:
    """True when two matches share any character of the text."""
    return a.start() < b.end() and b.start() < a.end()


def detect_task_request(text: str) -> TaskDecision:
    """Apply the A + B + C rule to the cleaned message text. The example exception applies when the message asks for an
    example, sample or demo and carries no assessed signal. v2: the verb (A) and the C signal must be different words, so
    one "complete" cannot count as both."""
    assessed = _ASSESSED.search(text)
    c_signals = [m for m in list(_ASSESSED.finditer(text)) + list(_COMPLETE.finditer(text))]
    verbs = list(_VERBS.finditer(text))
    pair = next(((v, c) for v in verbs for c in c_signals if not _overlaps(v, c)), None)
    verb = pair[0] if pair else (verbs[0] if verbs else None)
    complete = pair[1] if pair and pair[1].re is _COMPLETE else _COMPLETE.search(text)
    deliverable = _DELIVERABLES.search(text)
    example = _EXAMPLE.search(text)
    decision = TaskDecision(False, verb and verb.group(0), deliverable and deliverable.group(0),
                            assessed and assessed.group(0), complete and complete.group(0), example and example.group(0))
    if not verb:
        decision.reasons.append("no production verb")
    if not deliverable:
        decision.reasons.append("no deliverable")
    if not (assessed or complete):
        decision.reasons.append("no assessed or completeness signal")
    elif verb and not pair:
        decision.reasons.append("verb and completeness signal are the same word")
    if example and not assessed:
        decision.reasons.append("asks for an example and is not assessed")
    decision.is_task = not decision.reasons
    return decision


def coach_messages(messages: list[dict]) -> list[dict]:
    """The normal messages from llm.build_messages(), with the coach instruction appended to the user message. The system
    prompt and everything before the instruction are unchanged."""
    system, user = messages
    return [system, {"role": user["role"], "content": f"{user['content']}\n\n{settings.GUARDRAIL_COACH_INSTRUCTION}"}]


def cap_code_blocks(answer: str, max_lines: int | None = None) -> tuple[str, int]:
    """Cut every fenced code block longer than max_lines to its first max_lines lines plus a note. Returns the text and how
    many blocks were cut. This is the code-level part of the guardrail: it holds even when the model ignores the instruction."""
    limit = settings.GUARDRAIL_CODE_MAX_LINES if max_lines is None else max_lines
    cut = 0

    def trim(match: re.Match) -> str:
        nonlocal cut
        body, closing = match.group(1), match.group(2)
        lines = body.split("\n")
        if body.endswith("\n"):
            lines = lines[:-1]
        if len(lines) <= limit:
            return match.group(0)
        cut += 1
        head = match.group(0)[:match.start(1) - match.start(0)]
        return head + "\n".join(lines[:limit]) + "\n" + settings.GUARDRAIL_CODE_CUT_NOTE + "\n" + closing

    return _FENCE.sub(trim, answer), cut
