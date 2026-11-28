"""
Intent router: decides, before retrieval, whether a message is small talk, unclear, emotional, or a real question.

Order (cheapest first):
1. Rule-based small talk (whole message is a greeting / thanks / goodbye / acknowledgement) -> fixed reply.
2. Rule-based unclear input (gibberish, a stray character) -> fixed "please rephrase" reply.
3. Obvious question (ends with "?" or starts with a question word) -> retrieval, no LLM call.
4. LLM label classifier for what is left. The model may only output one label from settings.INTENT_LABELS.
   Any other output, an error or a timeout is treated as REAL_QUESTION.
5. Course-content override: an EMOTIONAL_* label is accepted only if the message's top-1 similarity to the
   course index is below settings.EMOTIONAL_OVERRIDE_MIN_SIMILARITY; then it gets a canned reply chosen here
   from settings.EMOTIONAL_RESPONSES. Otherwise (or if the check is unavailable or fails) it is a question.
   If the emotional message names a course topic, one fixed line offering to start with that topic follows the reply
   (settings.EMOTIONAL_TOPIC_OFFERS); it is chosen here by pattern, with no model call.
Every "question" goes through retrieval and the similarity-threshold fallback, so no path lets the model answer
from its own knowledge. All patterns, prompts and replies live in config/settings.py.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

from config import settings
from src.core import guardrail, llm, quiz, study_plan
from src.core.search_text import remove_leading

# Compiled once: each small-talk pattern must match the WHOLE normalised message.
_SMALL_TALK = {name: re.compile(rf"^(?:{pattern})$") for name, pattern in settings.SMALL_TALK_PATTERNS.items()}
# Course topics an emotional message may name, matched as whole words (settings.EMOTIONAL_TOPIC_OFFERS).
_TOPICS = [(re.compile(rf"\b(?:{pattern})\b", re.IGNORECASE), topic, question)
           for pattern, topic, question in settings.EMOTIONAL_TOPIC_OFFERS]


@dataclass
class Intent:
    """The routing decision for one message."""
    kind: str                   # "small_talk", "unclear", "emotional", "question", "quiz" or "study_plan"
    reply: str | None = None    # fixed reply for small_talk / unclear / emotional; None for questions
    path: str = "pattern"       # "pattern", "task_request", "question_heuristic", "classifier" or "similarity_override"
    label: str | None = None    # raw classifier output (when the classifier was called)
    top_score: float | None = None   # top-1 course similarity (when the override check ran)
    task: object | None = None  # guardrail.TaskDecision when the message is a task request (answered in coach mode)
    offer_question: str | None = None   # starter question of a topic offer in an emotional reply ("What is SQL?")
    quiz_topic: str | None = None    # for kind "quiz": the topic named ("" for "quiz me"), see quiz.detect
    plan_request: object | None = None   # for kind "study_plan": the parsed study_plan.PlanRequest


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


def topic_offer(*texts: str) -> str | None:
    """The one-line offer for the first course topic named in the message, or None if it names none. texts are the message
    as typed and its cleaned search text (so a corrected typo such as "sqll" still counts); the earliest mention wins.
    Fixed text from settings: no model is called."""
    match = topic_offer_match(*texts)
    return match[0] if match else None


def topic_offer_match(*texts: str) -> tuple[str, str] | None:
    """(offer line, starter question) for the first course topic named in the texts, or None. Multi-turn chat keeps the
    starter question, so a following "yes" can be answered as that question (conversation.py)."""
    for text in texts:
        found = [(m.start(), topic, question) for pattern, topic, question in _TOPICS if (m := pattern.search(text or ""))]
        if found:
            _, topic, question = min(found)
            return settings.EMOTIONAL_TOPIC_OFFER_TEMPLATE.format(topic=topic, question=question), question
    return None


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


def fixed_reply(message: str) -> Intent | None:
    """The small-talk or unclear decision for a message, or None if it is neither (the first two steps of route()).
    Small talk comes first, so one-letter phrases such as "k" are not caught by the unclear check."""
    text = _normalise(message)
    for name, pattern in _SMALL_TALK.items():
        if pattern.match(text):
            return Intent("small_talk", settings.SMALL_TALK_RESPONSES[name])
    if sum(ch.isalnum() for ch in text) < settings.UNCLEAR_MIN_ALNUM_CHARS:
        return Intent("unclear", settings.UNCLEAR_RESPONSE)
    if " " not in text and _is_keyboard_mash(text):
        return Intent("unclear", settings.UNCLEAR_RESPONSE)
    return None


def route(message: str, top_score: Callable[[str], float] | None = None,
          clean: Callable[[str], str] | None = None, use_classifier: bool = True) -> Intent:
    """Classify the message: small talk, unclear and emotional input get a fixed reply; anything else is a
    question that goes to retrieval. top_score(message) returns the top-1 course similarity; it is needed to
    accept an emotional label (without it, emotional labels fall back to a question).
    clean(message) is the pipeline's search text (RagPipeline.search_text: greetings removed, spelling corrected). The
    obvious-question check reads it, so "hey what is olpa" is seen as "what is olap" and goes straight to retrieval
    without the classifier. Small talk and the unclear check still read the message as typed, so "hi" stays a greeting."""
    text = _normalise(message)
    fixed = fixed_reply(message)
    if fixed:
        return fixed

    cleaned = clean(message) if clean else message
    cleaned_text = _normalise(cleaned) or text
    # Task requests ("write the full code for my assignment") come before the obvious-question check, because "give"
    # and "show" are question-start words there. They are still questions: retrieval runs, then coach mode answers.
    if settings.GUARDRAIL_ROUTER_ENABLED:
        decision = guardrail.detect_task_request(cleaned)
        if decision.is_task:
            return Intent("question", path="task_request", task=decision)
    # Quiz requests (feature B, 2026-11-28): the whole cleaned message is "quiz me on X" or similar. After the task check, so an
    # assessed task stays coach mode; before the question checks, so "give me questions on Kafka" is not answered as a question.
    if settings.QUIZ_ENABLED:
        topic = quiz.detect(cleaned)
        if topic is not None:
            return Intent("quiz", path="quiz", quiz_topic=topic)
    # Study plans (feature A, 2026-11-28): a planning phrase and a duration, or an explicit "study plan" request. After the quiz.
    if settings.PLAN_ENABLED:
        # Parsed from the message with greetings removed but NOT spelling-corrected: the corrector turns Malay words such as
        # "untuk" (for) into English ones ("untuck"). The topic is still spelling-corrected when it is searched.
        request = study_plan.detect(remove_leading(message))
        if request is not None:
            return Intent("study_plan", path="study_plan", plan_request=request)
    if not settings.INTENT_CLASSIFIER_ENABLED or _is_obvious_question(cleaned, cleaned_text):
        return Intent("question", path="question_heuristic")
    # Multi-turn (conversation.is_short_follow_up): a short, not clearly emotional follow-up inside a conversation is a question.
    if not use_classifier:
        return Intent("question", path="followup")

    label = classify(message)
    if label not in settings.EMOTIONAL_RESPONSES:
        return Intent("question", path="classifier", label=label)

    # Emotional label: accept it only if the message is NOT close to course content (fail safe -> question).
    try:
        score = top_score(message) if top_score else None
    except Exception:
        score = None
    if score is None or score >= settings.EMOTIONAL_OVERRIDE_MIN_SIMILARITY:
        return Intent("question", path="similarity_override", label=label, top_score=score)
    reply = settings.EMOTIONAL_RESPONSES[label]
    offer = topic_offer_match(message, cleaned)
    if offer:
        reply = f"{reply}\n\n{offer[0]}"
    return Intent("emotional", reply, path="classifier", label=label, top_score=score,
                  offer_question=offer[1] if offer else None)
