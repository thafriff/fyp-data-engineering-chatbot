"""
Quiz me (feature B; docs/feature_design_2026-11-28.md, approved by the author on 2026-11-28). APP LAYER ONLY: the frozen
SYSTEM_PROMPT, the notebook 04 runners and single-turn chat are untouched.

"quiz me on star schema" -> 2 multiple-choice questions and 1 short-answer question, written by the answer model from the retrieved
chunks only, with answers, explanations and citations shown after the student answers. Code decides everything it can:
- detection: the whole cleaned message matches settings.QUIZ_REQUEST_PATTERNS (intent.route checks it after the guardrail router, so
  an assessed task still goes to coach mode);
- retrieval and the 0.70 threshold, exactly as for a question (RagPipeline does this before any quiz is written);
- every item is checked before it is shown (check_item): its shape; its cited IDs are in the retrieved set; at least
  QUIZ_SUPPORT_MIN_SHARE of the correct answer's content words appear in its cited chunks; the question, searched like a student
  message, retrieves one of its cited chunks in the top 5 at or above the threshold; a multiple-choice item has four distinct
  options and one valid correct letter;
- items that fail are written once more (one retry for the missing ones); with fewer than QUIZ_MIN_ITEMS left there is no quiz
  (QuizError), never invented filler.
Multiple choice is marked by code; the short answer is self-marked by the student against the model answer (no model grading).
"""

import json
import re
import time
from collections.abc import Callable

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from config import settings
from src.core import conversation as conversation_rules
from src.core import llm
from src.core.retriever import RetrievedChunk

LETTERS = ("A", "B", "C", "D")
_PATTERNS = [re.compile(rf"^{pattern}$") for pattern in settings.QUIZ_REQUEST_PATTERNS]
_TRAILING = re.compile(r"(?:[\s,]+(?:" + "|".join(re.escape(w) for w in settings.QUIZ_TRAILING_WORDS) + r"))+$")
_END_PUNCTUATION = re.compile(r"[\s.?!,]+$")
_WORD = re.compile(r"[a-z0-9]+")
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
_META = [re.compile(p, re.IGNORECASE) for p in settings.QUIZ_META_REFERENCE_PATTERNS]
_NEGATIVE = [re.compile(p) for p in settings.QUIZ_NEGATIVE_PATTERNS]


class QuizError(Exception):
    """No reliable quiz could be made. str(error) is the message for the student; rounds holds what was tried (for evaluation)."""
    def __init__(self, message: str, rounds: list | None = None):
        super().__init__(message)
        self.rounds = rounds or []


# ---------------------------------------------------------------------------
# Detection and topic
# ---------------------------------------------------------------------------

def detect(text: str) -> str | None:
    """The quiz topic when the whole (cleaned) message is a quiz request ("" when it names no topic), otherwise None."""
    t = _END_PUNCTUATION.sub("", text.lower().strip())
    t = _END_PUNCTUATION.sub("", _TRAILING.sub("", t))
    for pattern in _PATTERNS:
        match = pattern.match(t)
        if match:
            return (match.groupdict().get("topic") or "").strip()
    return None


def resolve_topic(topic: str, conversation=None) -> str | None:
    """The topic to quiz on: the one named, or for "quiz me" / "quiz me on that" the earlier topic of the conversation.
    None when there is neither (the student is asked for one)."""
    if topic and conversation_rules.topic_words(topic):
        return topic
    last = conversation.last() if conversation is not None else None
    return last.topic_text if last is not None else None


# ---------------------------------------------------------------------------
# Prompt and parsing
# ---------------------------------------------------------------------------

def build_messages(topic: str, chunks: list[RetrievedChunk], mcq: int, short: int, avoid: list[str] | None = None) -> list[dict]:
    """The quiz prompt: its own system message (the chat SYSTEM_PROMPT is for answers, not quizzes), the excerpts rendered exactly as
    for chat, the topic and the format instruction. On the retry, the questions already kept are listed so they are not repeated."""
    excerpts = "\n\n".join(settings.CONTEXT_CHUNK_TEMPLATE.format(chunk_id=c.chunk_id, source_name=c.source_name,
                                                                  section=c.section, text=c.text) for c in chunks)
    user = (f"Course excerpts:\n\n{excerpts}\n\nTopic: {topic}\n\n"
            f"{settings.QUIZ_FORMAT_INSTRUCTION.format(mcq=mcq, short=short)}")
    if avoid:
        user += "\n\n" + settings.QUIZ_RETRY_INSTRUCTION.format(mcq=mcq, short=short, avoid="\n".join(f"- {q}" for q in avoid))
    return [{"role": "system", "content": settings.QUIZ_SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse(text: str) -> list | None:
    """The list of items in the model's reply, or None when it is not the expected JSON."""
    text = _FENCE.sub("", text.strip())
    try:
        data = json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start:end + 1])
        except ValueError:
            return None
    items = data.get("questions") if isinstance(data, dict) else data
    return items if isinstance(items, list) else None


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _stem(word: str) -> str:
    """A light plural fold, so "tables" supports "table"."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def content_words(text: str) -> set[str]:
    """The words that carry meaning: lower case, no English stop words, no single letters."""
    return {_stem(w) for w in _WORD.findall(text.lower()) if len(w) > 1 and w not in ENGLISH_STOP_WORDS}


def support_share(answer: str, cited_text: str) -> float:
    """The share of the answer's content words that appear in the cited chunks (0 when the answer has none)."""
    words = content_words(answer)
    return len(words & content_words(cited_text)) / len(words) if words else 0.0


def _ids(item: dict) -> list[str]:
    raw = item.get("chunk_ids")
    return [str(i).strip().strip("[]").strip() for i in raw] if isinstance(raw, list) else []


def correct_text(item: dict) -> str:
    """The text that must be supported: the correct option of a multiple-choice item, or the short answer."""
    if item.get("type") == "mcq":
        return str(item["options"][str(item.get("correct", "")).strip().upper()[:1]])
    return str(item.get("answer") or "")


def check_item(item, retrieved: dict[str, RetrievedChunk], search: Callable[[str], list[RetrievedChunk]]) -> str | None:
    """Why an item must not be shown, or None when it passes every check. search(question) is the pipeline's own search."""
    if not isinstance(item, dict):
        return "not an object"
    kind, question = item.get("type"), str(item.get("question") or "").strip()
    if kind not in ("mcq", "short"):
        return "unknown type"
    if len(question) < 10:
        return "no question"
    if any(p.search(question) for p in _META):
        return "question refers to the excerpts, not the topic"
    if any(p.search(question) for p in _NEGATIVE):
        return "negative question"
    if not str(item.get("explanation") or "").strip():
        return "no explanation"
    ids = _ids(item)
    if not ids:
        return "no chunk IDs"
    outside = [i for i in ids if i not in retrieved]
    if outside:
        return f"cites outside the retrieved set: {outside}"
    if kind == "mcq":
        options = item.get("options")
        if not isinstance(options, dict) or set(options) != set(LETTERS):
            return "options are not exactly A, B, C and D"
        texts = [str(options[k]).strip() for k in LETTERS]
        if not all(texts) or len({t.lower() for t in texts}) < len(LETTERS):
            return "options are not four distinct texts"
        if str(item.get("correct", "")).strip().upper()[:1] not in LETTERS:
            return "no valid correct letter"
    else:
        points = item.get("key_points")
        if not str(item.get("answer") or "").strip():
            return "no answer"
        if not isinstance(points, list) or not [p for p in points if str(p).strip()]:
            return "no key points"
    cited = " ".join(retrieved[i].text for i in ids)
    if support_share(correct_text(item), cited) < settings.QUIZ_SUPPORT_MIN_SHARE:
        return "answer not supported by its cited chunks"
    if not any(h.chunk_id in ids and h.score >= settings.SIMILARITY_THRESHOLD for h in search(question)):
        return "question does not retrieve its cited chunk"
    return None


def clean_item(item: dict) -> dict:
    """The shown form of a passed item: only the fields the card uses, normalised."""
    out = {"type": item["type"], "question": str(item["question"]).strip(), "explanation": str(item["explanation"]).strip(),
           "chunk_ids": _ids(item)}
    if item["type"] == "mcq":
        out["options"] = {k: str(item["options"][k]).strip() for k in LETTERS}
        out["correct"] = str(item["correct"]).strip().upper()[:1]
    else:
        out["answer"] = str(item["answer"]).strip()
        out["key_points"] = [str(p).strip() for p in item["key_points"] if str(p).strip()]
    return out


def _same_question(a: str, b: str) -> bool:
    return " ".join(_WORD.findall(a.lower())) == " ".join(_WORD.findall(b.lower()))


# ---------------------------------------------------------------------------
# Making a quiz
# ---------------------------------------------------------------------------

def _generate(messages: list[dict], model: str, on_status) -> tuple[str, dict | None]:
    """One model call returning JSON: the cloud model in JSON mode, or Ollama structured output for a local model."""
    if llm.is_cloud_model(model):
        reply = llm.cloud_complete(model, messages, settings.QUIZ_MAX_OUTPUT_TOKENS, json_output=True, on_status=on_status)
        return reply.text, {"input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens,
                            "reasoning_tokens": reply.reasoning_tokens, "cost_usd_list": reply.cost_usd_list,
                            "latency_seconds": reply.latency_seconds, "retries": reply.retries,
                            "retry_wait_seconds": reply.retry_wait_seconds}
    try:
        return llm.local_complete_json(messages, model, settings.QUIZ_MAX_OUTPUT_TOKENS, settings.QUIZ_JSON_SCHEMA), None
    except Exception as error:          # Ollama down or too slow: a message, not a crash
        raise QuizError(f"The offline model could not be reached ({type(error).__name__}). Is the Ollama app running?") from error


def _add_usage(total: dict | None, usage: dict | None) -> dict | None:
    if usage is None:
        return total
    if total is None:
        return dict(usage)
    for key in ("input_tokens", "output_tokens", "reasoning_tokens", "cost_usd_list", "latency_seconds", "retries",
                "retry_wait_seconds"):
        total[key] = total[key] + usage[key]
    return total


def make_quiz(topic: str, chunks: list[RetrievedChunk], model: str, search: Callable[[str], list[RetrievedChunk]],
              on_status: Callable[[str], None] | None = None) -> dict:
    """Write and check a quiz on the topic from these chunks. Returns the quiz (items, the retrieved chunk IDs, every round's check
    results, usage and time); raises QuizError when fewer than QUIZ_MIN_ITEMS items pass after one retry, or llm.CloudError."""
    start = time.perf_counter()
    retrieved = {c.chunk_id: c for c in chunks}
    want = {"mcq": settings.QUIZ_MCQ_COUNT, "short": settings.QUIZ_SHORT_COUNT}
    kept: dict[str, list[dict]] = {"mcq": [], "short": []}
    rounds, usage = [], None
    for attempt in (1, 2):
        need = {kind: want[kind] - len(kept[kind]) for kind in want}
        if not any(need.values()):
            break
        avoid = [i["question"] for kind in kept for i in kept[kind]] if attempt == 2 else None
        text, call_usage = _generate(build_messages(topic, chunks, need["mcq"], need["short"], avoid), model, on_status)
        usage = _add_usage(usage, call_usage)
        items = parse(text)
        record = {"attempt": attempt, "requested": need, "parsed": items is not None, "items": []}
        for item in items or []:
            reason = check_item(item, retrieved, search)
            kind = item.get("type") if isinstance(item, dict) else None
            shown = [i["question"] for k in kept for i in kept[k]]
            if reason is None and any(_same_question(str(item["question"]), q) for q in shown):
                reason = "repeats a question already kept"
            used = reason is None and len(kept[kind]) < want[kind]
            if used:
                kept[kind].append(clean_item(item))
            question = str(item.get("question", ""))[:300] if isinstance(item, dict) else ""
            record["items"].append({"type": kind, "passed": reason is None, "used": used, "reason": reason, "question": question})
        rounds.append(record)
    items = kept["mcq"] + kept["short"]
    if len(items) < settings.QUIZ_MIN_ITEMS:
        message = settings.QUIZ_FAILED_MESSAGE if llm.is_cloud_model(model) else settings.QUIZ_FAILED_LOCAL_MESSAGE
        raise QuizError(message, rounds)
    return {"topic": topic, "model": model, "items": items, "chunk_ids": [c.chunk_id for c in chunks], "rounds": rounds,
            "usage": usage, "seconds": round(time.perf_counter() - start, 1)}


def mark(quiz: dict, answers: dict) -> dict:
    """Code marking of the multiple-choice items: {item index: True/False}; short answers are self-marked, so they are not here."""
    return {i: answers.get(i) == item["correct"] for i, item in enumerate(quiz["items"]) if item["type"] == "mcq"}
