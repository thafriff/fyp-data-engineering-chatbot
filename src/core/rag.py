"""
RAG pipeline: question -> intent check -> search text (leading greetings removed, then spelling corrected) -> retrieve
chunks -> relevance check -> generate answer.

Multi-turn chat (conversation.py) is optional: answer(question) with no conversation is the single-turn pipeline, unchanged, and
the first message of a conversation is answered exactly the same way. Notebook 04 never passes a conversation.

This is the only entry point the user interface calls, so the UI never contains retrieval or prompt logic.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from config import settings
from src.core import answer_format, intent, llm
from src.core import conversation as conversation_rules
from src.core import course_terms
from src.core.conversation import Conversation
from src.core.intent import Intent
from src.core.retriever import RetrievedChunk, Retriever
from src.core.search_text import remove_leading, spelling_corrections
from src.core.spelling import SpellCorrector


@dataclass
class RagResult:
    """What the interface needs to show one answer."""
    chunks: list[RetrievedChunk]      # the retrieved chunks (shown as sources); empty if retrieval was skipped
    answer_stream: Iterator[str]      # the answer, piece by piece
    used_fallback: bool               # True if nothing relevant was found and the LLM was not called
    intent: str = "question"          # "small_talk", "unclear", "emotional" or "question" (src/core/intent.py)
    routing: Intent | None = None     # full routing decision (path taken, raw classifier label)
    model: str | None = None          # local model that generated the answer; None if no LLM answer was made
    search_query: str | None = None   # text actually embedded for the search (spelling-corrected); None if no search
    cloud_call: Callable[..., llm.CloudAnswer] | None = None   # for a cloud model: call it with on_status to get the answer
    coach_mode: bool = False          # True when the message was a task request and the answer is scaffolding (guardrail.py)
    comparison_format: bool = False   # True when the comparison format instruction was appended (answer_format.py)
    # Multi-turn chat (conversation.py). Single-turn values: "alone", 0, None, the question itself, the search text.
    search_mode: str | None = None    # "alone" or "combined" (with the earlier topic); None if no search
    history_exchanges: int = 0        # earlier exchanges sent to the model with this message
    resolved_question: str | None = None   # an accepted topic offer's starter question ("yes" -> "What is SQL?")
    asked: str | None = None          # the question actually answered (resolved_question, or the message as typed)
    topic_text: str | None = None     # what a later refer-back follow-up is combined with


class RagPipeline:
    """Holds the retriever (loaded once) and answers questions."""

    def __init__(self, retriever: Retriever | None = None):
        """Create the pipeline; loading the retriever takes a few seconds (embedding model + index)."""
        self.retriever = retriever or Retriever()
        self.speller = SpellCorrector() if settings.SPELL_CORRECTION_ENABLED else None

    def _top_score(self, message: str) -> float:
        """Top-1 course similarity of a message (embedding search only, no LLM); used by the intent router's emotional
        override. It searches the same cleaned text as retrieval, so a greeting cannot pull a course question under the
        override line ("hi what is olap" scores 0.694 raw, 0.708 cleaned)."""
        chunks = self.retriever.retrieve(self.search_text(message), top_k=1)
        return chunks[0].score if chunks else 0.0

    def search_text(self, question: str) -> str:
        """THE function that builds the text embedded for a search, used by retrieval, the emotional override and
        evaluation/notebook04/freeze_contexts.py: leading greetings and politeness removed, then spelling corrected.
        Greetings go first, because the corrector would otherwise change "salam" into "salad". The question itself,
        which the model receives, is not changed."""
        text = remove_leading(question)
        return self.speller.correct(text) if self.speller else text

    def answer(self, question: str, model: str | None = None, conversation: Conversation | None = None) -> RagResult:
        """Small talk, unclear and emotional input get a fixed reply without retrieval. Real questions (on- or
        off-topic) retrieve the top chunks; if the best one is not similar enough, return the fallback message
        without calling the LLM, otherwise stream the LLM's answer grounded in those chunks.
        model only changes the final answer generation (default settings.DEV_LLM_MODEL); routing is unaffected.
        Greeting removal and spelling correction change the text embedded for the search. The LLM always gets the original
        question; when spelling correction changed a word and the correction appears in the retrieved excerpts, it also gets a
        one-line "Assuming you mean ..." hint (search_text.spelling_corrections).
        conversation: the chat so far (conversation.py), or None for single-turn. The caller records the finished turn with
        conversation.record(result, answer). With None, or an empty conversation, the result is exactly the single-turn one."""
        multi = conversation is not None and settings.MULTITURN_ENABLED
        asked, routed = question, None
        if multi:
            pending, conversation.pending_offer = conversation.pending_offer, None    # an offer is valid for one message
            if pending and conversation_rules.is_affirmative(question):
                asked = pending                                   # "yes" after a topic offer -> its starter question
            elif pending and conversation_rules.is_negative(question):
                declined = Intent("offer_declined", settings.OFFER_DECLINED_RESPONSE, path="offer")
                return RagResult(chunks=[], answer_stream=iter([declined.reply]), used_fallback=False,
                                 intent=declined.kind, routing=declined)
            elif intent.fixed_reply(question) is None and conversation_rules.stays_in_coach(
                    question, self.search_text(question), conversation):
                routed = Intent("question", path="task_request_followup", task=conversation.last().task)
        if routed is None and multi:
            # A short, not clearly emotional follow-up inside a conversation skips the emotional classifier (path "followup").
            routed = intent.route(asked, top_score=self._top_score, clean=self.search_text,
                                  use_classifier=not conversation_rules.is_short_follow_up(asked, conversation))
        routed = routed or intent.route(asked, top_score=self._top_score, clean=self.search_text)
        if routed.kind != "question":
            return RagResult(chunks=[], answer_stream=iter([routed.reply]), used_fallback=False,
                             intent=routed.kind, routing=routed)

        own_search = self.search_text(asked)
        search_mode, search_query, topic_text = "alone", own_search, own_search
        if multi and asked == question:                           # an accepted offer's starter question is searched alone
            if routed.path == "task_request_followup":            # coach follow-up: always with the task's topic
                search_mode, search_query = "combined", f"{conversation.last().topic_text} {own_search}"
                topic_text = conversation.last().topic_text
            else:
                search_mode, search_query, topic_text = conversation_rules.follow_up_search(own_search, conversation)
        resolved = asked if asked != question else None
        chunks = self.retriever.retrieve(search_query)
        # Short-follow-up rescue: searched alone and below the threshold, but every topic word is a course term ("and ELT?")
        # -> search again combined with the earlier topic. "what about the weather today?" is not rescued.
        if (multi and asked == question and search_mode == "alone" and conversation.last() is not None
                and settings.FOLLOWUP_RESCUE_ENABLED and (not chunks or chunks[0].score < settings.SIMILARITY_THRESHOLD)
                and course_terms.all_course_terms(conversation_rules.topic_words(own_search))):
            search_mode = "combined"
            search_query = topic_text = f"{conversation.last().topic_text} {own_search}"
            chunks = self.retriever.retrieve(search_query)
        if not chunks or chunks[0].score < settings.SIMILARITY_THRESHOLD:
            return RagResult(chunks=chunks, answer_stream=iter([settings.FALLBACK_MESSAGE]), used_fallback=True,
                             routing=routed, search_query=search_query, search_mode=search_mode,
                             resolved_question=resolved, asked=asked, topic_text=topic_text)
        model = model or settings.DEV_LLM_MODEL
        corrections = spelling_corrections(asked, own_search, [c.text for c in chunks])
        # Coach mode for a task request (or a coach follow-up): the normal messages are built as usual, then the coach
        # instruction is appended.
        coach = routed.path in ("task_request", "task_request_followup")
        # Comparison format for a question that compares things (read from this message only); coach mode wins.
        comparison = not coach and answer_format.detect_comparison(asked, own_search) is not None
        # Earlier exchanges that fit the token budget beside this prompt and all its instructions ([] when single-turn).
        history = []
        if multi:
            base = llm.build_messages(asked, chunks, corrections=corrections)
            without_history = answer_format.app_transform(coach=coach, comparison=comparison)
            history = conversation_rules.select_history(conversation, without_history(base) if without_history else base)
        # Every generated answer also gets the opening rule (answer_format.app_transform), before any mode instruction.
        transform = answer_format.app_transform(coach=coach, comparison=comparison, history=history)
        extra = {"search_mode": search_mode, "history_exchanges": len(history), "resolved_question": resolved,
                 "asked": asked, "topic_text": topic_text}
        if llm.is_cloud_model(model):      # not streamed: the interface shows the waiting state, then the finished answer
            return RagResult(chunks=chunks, answer_stream=iter(()), used_fallback=False, routing=routed, model=model,
                             search_query=search_query, coach_mode=coach, comparison_format=comparison,
                             cloud_call=lambda on_status=None: llm.cloud_generate(asked, chunks, model, on_status,
                                                                                  corrections=corrections, transform=transform),
                             **extra)
        return RagResult(chunks=chunks,
                         answer_stream=llm.stream_generate(asked, chunks, model=model, corrections=corrections,
                                                           transform=transform),
                         used_fallback=False, routing=routed, model=model, search_query=search_query, coach_mode=coach,
                         comparison_format=comparison, **extra)
