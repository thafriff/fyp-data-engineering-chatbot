"""
Multi-turn chat: the conversation memory and the follow-up rules (author's approved design, 2026-11-26; proposal Section 2.4.3).

A SYSTEM FEATURE of the chat prototype, evaluated in notebook 05. NOT part of notebook 04: the runners never pass a conversation,
llm.build_messages() is untouched, and the first message of every conversation is answered exactly as a single-turn question.

What it does, all by rules (no extra model call):
- keeps the last model-answered exchanges and sends at most settings.HISTORY_MAX_EXCHANGES of them with a new message, each earlier
  answer trimmed (citations, tables and code removed, at most HISTORY_ANSWER_MAX_CHARS), and only as many as the token budget allows;
- decides the search text of a follow-up: COMBINED with the earlier topic when the follow-up has no topic word of its own or
  contains a refer-back pronoun ("how is it different from a data lake?"), otherwise ALONE, exactly as single-turn;
- remembers a topic offer from an emotional reply, so a following "yes" is answered as its starter question;
- keeps coach mode on after a coach-mode turn when the student refers back or pushes for the solution.
All word lists and limits live in config/settings.py (HISTORY_*, FOLLOWUP_*, OFFER_*, GUARDRAIL_PRESSURE_PATTERNS).
"""

import math
import re
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from config import settings

_WORD = re.compile(r"[a-z0-9]+")
_PRONOUNS = frozenset(settings.FOLLOWUP_PRONOUNS)
_NOT_TOPIC = (frozenset(ENGLISH_STOP_WORDS) | _PRONOUNS | frozenset(settings.FOLLOWUP_REFERBACK_WORDS)
              | frozenset(settings.FOLLOWUP_GENERIC_WORDS))
_AFFIRMATIVE = re.compile(rf"^(?:{settings.OFFER_AFFIRMATIVE_PATTERN})$")
_NEGATIVE = re.compile(rf"^(?:{settings.OFFER_NEGATIVE_PATTERN})$")
_PRESSURE = [re.compile(p, re.IGNORECASE) for p in settings.GUARDRAIL_PRESSURE_PATTERNS]
_EMOTIONAL_CUES = [re.compile(p, re.IGNORECASE) for p in settings.EMOTIONAL_CUE_PATTERNS]
# "one"/"ones" count as refer-back pronouns only as a whole word that is not part of a hyphenated term ("one-to-many").
_ONE = re.compile(r"(?<![\w-])ones?(?![\w-])", re.IGNORECASE)
_CITATION = re.compile(rf"\s?\[[^\]]*{settings.NB04_CITATION_ID_PATTERN}[^\]]*\]")
_FENCED = re.compile(r"```.*?(```|$)", re.DOTALL)


@dataclass
class Exchange:
    """One model-answered turn, as remembered for history and follow-up search."""
    question: str               # the question answered (an accepted topic offer's starter question, or the message as typed)
    topic_text: str             # the search text a refer-back follow-up is combined with
    chunk_ids: list[str]
    answer: str                 # the answer as the student saw it (code blocks already capped in coach mode)
    coach: bool = False
    task: object | None = None  # guardrail.TaskDecision of a coach-mode turn


@dataclass
class Conversation:
    """The state of one chat. A new, empty conversation behaves exactly like single-turn chat."""
    exchanges: list[Exchange] = field(default_factory=list)
    pending_offer: str | None = None    # starter question offered by the previous (emotional) reply, for one message only

    def clear(self) -> None:
        """New chat: forget every exchange and any pending offer."""
        self.exchanges.clear()
        self.pending_offer = None

    def last(self) -> Exchange | None:
        """The most recent model-answered exchange, or None."""
        return self.exchanges[-1] if self.exchanges else None

    def record(self, result, answer: str, failed: bool = False) -> None:
        """Remember a finished turn (call it after the answer is complete). An emotional reply's topic offer is kept for the
        next message; a model-answered turn becomes an exchange; fixed replies and the pipeline fallback add no exchange, so
        they do not break the chain ("thanks" then "and ELT?" still follows the ETL answer)."""
        routing = result.routing
        self.pending_offer = routing.offer_question if routing is not None and result.intent == "emotional" else None
        if result.model and answer and not failed and not result.used_fallback:
            self.exchanges.append(Exchange(question=result.asked or "", topic_text=result.topic_text or "",
                                           chunk_ids=[c.chunk_id for c in result.chunks], answer=answer,
                                           coach=result.coach_mode, task=getattr(routing, "task", None)))
            del self.exchanges[:-max(1, settings.HISTORY_MAX_EXCHANGES)]


def _words(text: str) -> list[str]:
    """Lower-case words of a text, apostrophes dropped ("what's" -> "whats")."""
    return _WORD.findall((text or "").lower().replace("'", "").replace("’", ""))


def topic_words(text: str) -> list[str]:
    """The words of a message that carry a topic: not stop words, refer-back words, generic aspect words or pronouns."""
    return [w for w in _words(text) if w not in _NOT_TOPIC]


def has_pronoun(text: str) -> bool:
    """True when the message contains one of the refer-back pronouns (settings.FOLLOWUP_PRONOUNS). "one"/"ones" count only as
    a whole word outside a hyphenated term, so "which one is better" counts and "a one-to-many relationship" does not."""
    words = _words(text)
    if any(w in _PRONOUNS and w not in ("one", "ones") for w in words):
        return True
    return bool({"one", "ones"} & _PRONOUNS) and bool(_ONE.search(text or ""))


def refers_back(text: str) -> bool:
    """True when the message has no topic word of its own ("explain that simpler", "give an example", "why?")."""
    return bool(_words(text)) and not topic_words(text)


def follow_up_search(own_search_text: str, conversation: "Conversation | None") -> tuple[str, str, str]:
    """(mode, search text to embed, topic text to remember) for a routed question. COMBINED with the earlier topic when the
    follow-up refers back (no topic word of its own) or contains a refer-back pronoun; otherwise ALONE, as single-turn. A
    pure refer-back keeps the earlier topic as the topic; a pronoun follow-up with its own words makes the combination the
    new topic ("data warehouse" + "how is it different from a data lake" covers both)."""
    last = conversation.last() if conversation is not None else None
    if last is None:
        return "alone", own_search_text, own_search_text
    if refers_back(own_search_text):
        return "combined", f"{last.topic_text} {own_search_text}", last.topic_text
    if has_pronoun(own_search_text):
        combined = f"{last.topic_text} {own_search_text}"
        return "combined", combined, combined
    return "alone", own_search_text, own_search_text


def _normalise(message: str) -> str:
    """Lower case, apostrophes dropped, punctuation as spaces, single spaces ("Yes please!" -> "yes please")."""
    return " ".join(_words(message))


def is_affirmative(message: str) -> bool:
    """True for a whole-message yes ("yes", "sure", "ok", "boleh") that accepts a pending topic offer."""
    return bool(_AFFIRMATIVE.match(_normalise(message)))


def is_negative(message: str) -> bool:
    """True for a whole-message no ("no", "not now", "tak nak") that declines a pending topic offer."""
    return bool(_NEGATIVE.match(_normalise(message)))


def is_pressure(message: str) -> bool:
    """True when the message pushes for the solution ("just give me the answer", "give me the full code")."""
    text = _normalise(message)
    return any(p.search(text) for p in _PRESSURE)


def is_clearly_emotional(message: str) -> bool:
    """True when the message carries a clear emotional cue ("give up", "stressed", "tired", "ugh" ...; settings
    EMOTIONAL_CUE_PATTERNS): such a message still goes to the classifier and can get the kind reply."""
    text = _normalise(message)
    return any(p.search(text) for p in _EMOTIONAL_CUES)


def is_short_follow_up(message: str, conversation: "Conversation | None") -> bool:
    """True inside a conversation (an answered turn exists) for a message of at most FOLLOWUP_SHORT_MAX_WORDS words without a
    clear emotional cue: it is treated as a follow-up question instead of being sent to the emotional classifier."""
    return bool(conversation is not None and conversation.last() is not None
                and len(_words(message)) <= settings.FOLLOWUP_SHORT_MAX_WORDS and not is_clearly_emotional(message))


def stays_in_coach(message: str, own_search_text: str, conversation: "Conversation | None") -> bool:
    """True when the previous exchange was in coach mode and this message refers back or pushes for the solution. Coach mode
    ends with a new question that has its own topic and is not a task (the router then decides as usual)."""
    last = conversation.last() if conversation is not None else None
    return bool(last and last.coach and (refers_back(own_search_text) or is_pressure(message)))


def trim_answer(answer: str) -> str:
    """An earlier answer as sent in the history: stop notes, citations, code blocks and table lines removed, whitespace
    collapsed, at most HISTORY_ANSWER_MAX_CHARS, cut at a sentence end when there is one in the second half."""
    text = answer or ""
    for note in (settings.ANSWER_CUT_NOTE, settings.ANSWER_REPEAT_NOTE, settings.GUARDRAIL_CODE_CUT_NOTE):
        text = text.replace(note, " ")
    text = _FENCED.sub(" ", text)
    text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("|"))
    text = " ".join(_CITATION.sub("", text).split())
    limit = settings.HISTORY_ANSWER_MAX_CHARS
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return cut[:end + 1] if end >= limit // 2 else cut[:cut.rfind(" ")].rstrip(",;:") + " ..."


def _trim_question(question: str) -> str:
    """An earlier question as sent in the history: as typed, cut to HISTORY_QUESTION_MAX_CHARS."""
    return (question or "")[:settings.HISTORY_QUESTION_MAX_CHARS]


def estimated_tokens(messages: list[dict]) -> int:
    """Conservative token estimate of chat messages: characters / HISTORY_CHARS_PER_TOKEN, plus a per-message overhead."""
    return sum(math.ceil(len(m["content"]) / settings.HISTORY_CHARS_PER_TOKEN) + settings.HISTORY_MESSAGE_OVERHEAD_TOKENS
               for m in messages)


def history_messages(exchanges: list[Exchange]) -> list[dict]:
    """The earlier exchanges as chat turns: the question as typed, then the trimmed answer."""
    out = []
    for e in exchanges:
        out += [{"role": "user", "content": _trim_question(e.question)}, {"role": "assistant", "content": trim_answer(e.answer)}]
    return out


def select_history(conversation: "Conversation | None", current_messages: list[dict]) -> list[Exchange]:
    """The earlier exchanges to send with this message: the most recent ones, at most HISTORY_MAX_EXCHANGES, and only as many
    as fit beside the current prompt (current_messages, with all its instructions), the history instruction and the answer cap
    inside DEV_LLM_NUM_CTX with HISTORY_TOKEN_MARGIN to spare. Oldest first. [] for an empty conversation."""
    if conversation is None or not settings.MULTITURN_ENABLED or not conversation.exchanges:
        return []
    budget = (settings.DEV_LLM_NUM_CTX - settings.DEV_LLM_MAX_ANSWER_TOKENS - settings.HISTORY_TOKEN_MARGIN
              - estimated_tokens(current_messages)
              - math.ceil((len(settings.HISTORY_INSTRUCTION) + 2) / settings.HISTORY_CHARS_PER_TOKEN))
    chosen = []
    for exchange in reversed(conversation.exchanges[-settings.HISTORY_MAX_EXCHANGES:]):
        cost = estimated_tokens(history_messages([exchange]))
        if cost > budget:
            break
        chosen.insert(0, exchange)
        budget -= cost
    return chosen


def with_history(messages: list[dict], exchanges: list[Exchange]) -> list[dict]:
    """The messages with the earlier exchanges inserted as chat turns before the current message, and the history instruction
    appended to the current message. Unchanged when there is no exchange."""
    if not exchanges:
        return messages
    *head, user = messages
    return [*head, *history_messages(exchanges),
            {"role": user["role"], "content": f"{user['content']}\n\n{settings.HISTORY_INSTRUCTION}"}]
