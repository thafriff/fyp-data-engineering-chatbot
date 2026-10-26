"""
LLM client: builds the course-assistant prompt from a question and its retrieved chunks, and sends it to
the language model.

DEV PROTOTYPE: the model is a small local model served by Ollama (settings.DEV_LLM_MODEL), used only to
see the chatbot working end to end. The formal LLM is chosen in notebook 04 and will replace this backend;
the prompt-building part stays the same.
"""

import json
from collections.abc import Iterator

import requests

from config import settings
from src.core.retriever import RetrievedChunk


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict]:
    """Build the chat messages: the system instructions, then the context excerpts and the question."""
    context = "\n\n".join(
        settings.CONTEXT_CHUNK_TEMPLATE.format(chunk_id=c.chunk_id, source_name=c.source_name,
                                               section=c.section, text=c.text)
        for c in chunks
    )
    user_message = (f"Context excerpts from the course materials:\n\n{context}\n\n"
                    f"Student question: {question}\n\n{settings.ANSWER_REMINDER}")
    return [{"role": "system", "content": settings.SYSTEM_PROMPT},
            {"role": "user", "content": user_message}]


def _request_body(messages: list[dict], stream: bool) -> dict:
    """Ollama /api/chat request body with the generation settings from settings.py."""
    return {
        "model": settings.DEV_LLM_MODEL,
        "messages": messages,
        "stream": stream,
        "keep_alive": settings.DEV_LLM_KEEP_ALIVE,
        "options": {"temperature": settings.LLM_TEMPERATURE, "num_ctx": settings.DEV_LLM_NUM_CTX},
    }


def generate(question: str, chunks: list[RetrievedChunk]) -> str:
    """Return the full answer in one piece (used for scripted tests)."""
    response = requests.post(f"{settings.OLLAMA_BASE_URL}/api/chat",
                             json=_request_body(build_messages(question, chunks), stream=False),
                             timeout=settings.DEV_LLM_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()["message"]["content"]


def stream_generate(question: str, chunks: list[RetrievedChunk]) -> Iterator[str]:
    """Yield the answer piece by piece as the model writes it, so the interface can show it immediately."""
    with requests.post(f"{settings.OLLAMA_BASE_URL}/api/chat",
                       json=_request_body(build_messages(question, chunks), stream=True),
                       stream=True, timeout=settings.DEV_LLM_TIMEOUT_SECONDS) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            part = json.loads(line)
            if part.get("message", {}).get("content"):
                yield part["message"]["content"]
            if part.get("done"):
                break


def check_backend() -> tuple[bool, str]:
    """Check that the Ollama server is running and the configured model is downloaded."""
    try:
        tags = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5).json()
    except requests.RequestException:
        return False, f"Ollama is not running at {settings.OLLAMA_BASE_URL}. Start the Ollama app and reload."
    names = {m["name"] for m in tags.get("models", [])}
    if settings.DEV_LLM_MODEL not in names:
        return False, f"Model {settings.DEV_LLM_MODEL} is not downloaded. Run: ollama pull {settings.DEV_LLM_MODEL}"
    return True, f"Ollama is running with {settings.DEV_LLM_MODEL}."
