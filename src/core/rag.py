"""
RAG pipeline: question -> intent check -> retrieve chunks -> relevance check -> generate answer.

This is the only entry point the user interface calls, so the UI never contains retrieval or prompt logic.
"""

from collections.abc import Iterator
from dataclasses import dataclass

from config import settings
from src.core import intent, llm
from src.core.retriever import RetrievedChunk, Retriever


@dataclass
class RagResult:
    """What the interface needs to show one answer."""
    chunks: list[RetrievedChunk]      # the retrieved chunks (shown as sources); empty if retrieval was skipped
    answer_stream: Iterator[str]      # the answer, piece by piece
    used_fallback: bool               # True if nothing relevant was found and the LLM was not called
    intent: str = "question"          # "small_talk", "unclear" or "question" (see src/core/intent.py)


class RagPipeline:
    """Holds the retriever (loaded once) and answers questions."""

    def __init__(self, retriever: Retriever | None = None):
        """Create the pipeline; loading the retriever takes a few seconds (embedding model + index)."""
        self.retriever = retriever or Retriever()

    def answer(self, question: str) -> RagResult:
        """Small talk and unclear input get a fixed reply without retrieval. Real questions (on- or off-topic)
        retrieve the top chunks; if the best one is not similar enough, return the fallback message without
        calling the LLM, otherwise stream the LLM's answer grounded in those chunks."""
        routed = intent.route(question)
        if routed.kind != "question":
            return RagResult(chunks=[], answer_stream=iter([routed.reply]), used_fallback=False, intent=routed.kind)

        chunks = self.retriever.retrieve(question)
        if not chunks or chunks[0].score < settings.SIMILARITY_THRESHOLD:
            return RagResult(chunks=chunks, answer_stream=iter([settings.FALLBACK_MESSAGE]), used_fallback=True)
        return RagResult(chunks=chunks, answer_stream=llm.stream_generate(question, chunks), used_fallback=False)
