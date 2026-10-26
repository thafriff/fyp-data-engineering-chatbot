"""
Retriever: finds the knowledge-base chunks that are most relevant to a student's question.

It loads the FAISS index built and locked in notebook 03, the chunk-id order saved next to it, and the
chunk text + metadata from data/chunks/chunks.json. Questions are embedded with the same model that
embedded the chunks (settings.EMBEDDING_MODEL), without a query prefix, exactly as in the benchmark.
"""

import json
from dataclasses import dataclass

import faiss
from sentence_transformers import SentenceTransformer

from config import settings


@dataclass
class RetrievedChunk:
    """One search result: the chunk, its metadata and its cosine similarity to the question."""
    chunk_id: str
    score: float
    text: str
    source_id: str
    source_name: str
    source_type: str
    section: str
    page: int | None
    file: str | None
    topic: str | None


class Retriever:
    """Loads the locked index once and answers 'top-k chunks for this question' requests."""

    def __init__(self):
        """Load the index, the chunk-id order and the chunk metadata, and check they belong together."""
        self.index = faiss.read_index(str(settings.FAISS_INDEX_FILE))
        id_map = json.loads(settings.FAISS_ID_MAP_FILE.read_text(encoding="utf-8"))
        self.chunk_ids = id_map["chunk_ids"]
        chunks = json.loads(settings.CHUNKS_FILE.read_text(encoding="utf-8"))
        self.chunks = {c["chunk_id"]: c for c in chunks}

        # Refuse to run with a mismatched index: wrong model or a different chunk set gives wrong results.
        if id_map["embedding_model"] != settings.EMBEDDING_MODEL:
            raise RuntimeError(f"The saved index was built with {id_map['embedding_model']}, but settings.py "
                               f"says {settings.EMBEDDING_MODEL}. Re-run notebook 03.")
        if self.index.ntotal != len(self.chunk_ids) or set(self.chunk_ids) != set(self.chunks):
            raise RuntimeError("The FAISS index and data/chunks/chunks.json do not match. Re-run notebook 03.")

        self.query_prefix = settings.EMBEDDING_QUERY_PREFIX_IN_USE
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return the top_k chunks most similar to the question, best first."""
        top_k = top_k or settings.TOP_K
        vector = self.model.encode([self.query_prefix + question], normalize_embeddings=True).astype("float32")
        scores, rows = self.index.search(vector, top_k)
        results = []
        for score, row in zip(scores[0], rows[0]):
            if row < 0:                       # FAISS returns -1 when fewer results exist than requested
                continue
            chunk = self.chunks[self.chunk_ids[row]]
            results.append(RetrievedChunk(
                chunk_id=chunk["chunk_id"], score=float(score), text=chunk["text"],
                source_id=chunk["source_id"], source_name=chunk["source_name"], source_type=chunk["source_type"],
                section=chunk["section"], page=chunk.get("page"), file=chunk.get("file"), topic=chunk.get("topic"),
            ))
        return results
