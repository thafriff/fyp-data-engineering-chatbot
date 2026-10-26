"""
RAG pipeline: question -> intent check -> spelling correction of the search query -> retrieve chunks
-> relevance check -> generate answer.

This is the only entry point the user interface calls, so the UI never contains retrieval or prompt logic.
"""

from collections.abc import Iterator
from dataclasses import dataclass

from config import settings
from src.core import intent, llm
from src.core.intent import Intent
from src.core.retriever import RetrievedChunk, Retriever
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


class RagPipeline:
    """Holds the retriever (loaded once) and answers questions."""

    def __init__(self, retriever: Retriever | None = None):
        """Create the pipeline; loading the retriever takes a few seconds (embedding model + index)."""
        self.retriever = retriever or Retriever()
        self.speller = SpellCorrector() if settings.SPELL_CORRECTION_ENABLED else None

    def _top_score(self, message: str) -> float:
        """Top-1 course similarity of a message (embedding search only, no LLM); used by the intent router."""
        chunks = self.retriever.retrieve(message, top_k=1)
        return chunks[0].score if chunks else 0.0

    def answer(self, question: str, model: str | None = None) -> RagResult:
        """Small talk, unclear and emotional input get a fixed reply without retrieval. Real questions (on- or
        off-topic) retrieve the top chunks; if the best one is not similar enough, return the fallback message
        without calling the LLM, otherwise stream the LLM's answer grounded in those chunks.
        model only changes the final answer generation (default settings.DEV_LLM_MODEL); routing is unaffected.
        Spelling correction only changes the text embedded for the search; the LLM gets the original question."""
        routed = intent.route(question, top_score=self._top_score)
        if routed.kind != "question":
            return RagResult(chunks=[], answer_stream=iter([routed.reply]), used_fallback=False,
                             intent=routed.kind, routing=routed)

        search_query = self.speller.correct(question) if self.speller else question
        chunks = self.retriever.retrieve(search_query)
        if not chunks or chunks[0].score < settings.SIMILARITY_THRESHOLD:
            return RagResult(chunks=chunks, answer_stream=iter([settings.FALLBACK_MESSAGE]), used_fallback=True,
                             routing=routed, search_query=search_query)
        model = model or settings.DEV_LLM_MODEL
        return RagResult(chunks=chunks, answer_stream=llm.stream_generate(question, chunks, model=model),
                         used_fallback=False, routing=routed, model=model, search_query=search_query)
