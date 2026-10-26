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


def _request_body(messages: list[dict], stream: bool, model: str | None = None) -> dict:
    """Ollama /api/chat request body with the generation settings from settings.py.
    model defaults to settings.DEV_LLM_MODEL."""
    return {
        "model": model or settings.DEV_LLM_MODEL,
        "messages": messages,
        "stream": stream,
        "keep_alive": settings.DEV_LLM_KEEP_ALIVE,
        "options": {"temperature": settings.LLM_TEMPERATURE, "num_ctx": settings.DEV_LLM_NUM_CTX},
    }


def generate(question: str, chunks: list[RetrievedChunk], model: str | None = None) -> str:
    """Return the full answer in one piece (used for scripted tests)."""
    response = requests.post(f"{settings.OLLAMA_BASE_URL}/api/chat",
                             json=_request_body(build_messages(question, chunks), stream=False, model=model),
                             timeout=settings.DEV_LLM_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()["message"]["content"]


def stream_generate(question: str, chunks: list[RetrievedChunk], model: str | None = None) -> Iterator[str]:
    """Yield the answer piece by piece as the model writes it, so the interface can show it immediately."""
    with requests.post(f"{settings.OLLAMA_BASE_URL}/api/chat",
                       json=_request_body(build_messages(question, chunks), stream=True, model=model),
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


def complete(messages: list[dict], max_tokens: int, timeout: float, temperature: float = 0.0) -> str:
    """Send arbitrary chat messages and return the reply text (used by the intent classifier).
    max_tokens caps the length of the output; errors are raised to the caller."""
    body = _request_body(messages, stream=False)
    body["options"].update({"temperature": temperature, "num_predict": max_tokens})
    response = requests.post(f"{settings.OLLAMA_BASE_URL}/api/chat", json=body, timeout=timeout)
    response.raise_for_status()
    return response.json()["message"]["content"]


def check_backend(model: str | None = None) -> tuple[bool, str]:
    """Check that the Ollama server is running and that both the answer model (default
    settings.DEV_LLM_MODEL) and the intent classifier's model (settings.DEV_LLM_MODEL) are downloaded."""
    model = model or settings.DEV_LLM_MODEL
    try:
        tags = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5).json()
    except requests.RequestException:
        return False, f"Ollama is not running at {settings.OLLAMA_BASE_URL}. Start the Ollama app and reload."
    names = {m["name"] for m in tags.get("models", [])}
    for needed in dict.fromkeys([model, settings.DEV_LLM_MODEL]):
        if needed not in names:
            return False, f"Model {needed} is not downloaded. Run: ollama pull {needed}"
    return True, f"Ollama is running with {model}."
