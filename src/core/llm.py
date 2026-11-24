"""
LLM client: builds the course-assistant prompt from a question and its retrieved chunks, and sends it to
the language model.

DEV PROTOTYPE: the default model is a small local model served by Ollama (settings.DEV_LLM_MODEL), used only to
see the chatbot working end to end. The formal LLM is chosen in notebook 04 and will replace this backend;
the prompt-building part stays the same.

Two cloud models (settings.DEV_CLOUD_MODELS) can also be picked in the sidebar for casual viewing. They get exactly the
same prompt from build_messages(), through cloud_generate() below, and nothing they do touches the notebook 04 caches.
"""

import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import requests
from dotenv import dotenv_values

from config import settings
from src.core.retriever import RetrievedChunk
from src.core.timeouts import RequestTimedOut


def spelling_hint(corrections: list[tuple[str, str]] | None) -> str:
    """The one hint line for (typed, corrected) word pairs, or "" when there are none."""
    if not corrections:
        return ""
    typos = "; ".join(settings.SPELLING_HINT_TYPO.format(typed=t, corrected=c) for t, c in corrections)
    return settings.SPELLING_HINT_TEMPLATE.format(typos=typos, meant=" ".join(c for _, c in corrections))


def build_messages(question: str, chunks: list[RetrievedChunk],
                   corrections: list[tuple[str, str]] | None = None) -> list[dict]:
    """Build the chat messages: the system instructions, then the context excerpts and the question.
    corrections are the (typed, corrected) word pairs from search_text.spelling_corrections; when there are any, one hint
    line follows the question, which is still sent exactly as typed. Without them the messages are exactly as before the
    hint existed."""
    context = "\n\n".join(
        settings.CONTEXT_CHUNK_TEMPLATE.format(chunk_id=c.chunk_id, source_name=c.source_name,
                                               section=c.section, text=c.text)
        for c in chunks
    )
    line = spelling_hint(corrections)
    hint = f"\n{line}" if line else ""
    user_message = (f"Context excerpts from the course materials:\n\n{context}\n\n"
                    f"Student question: {question}{hint}\n\n{settings.ANSWER_REMINDER}")
    return [{"role": "system", "content": settings.SYSTEM_PROMPT},
            {"role": "user", "content": user_message}]


def _request_body(messages: list[dict], stream: bool, model: str | None = None) -> dict:
    """Ollama /api/chat request body with the generation settings from settings.py.
    model defaults to settings.DEV_LLM_MODEL. Models in settings.DEV_LLM_THINK_OFF_MODELS get "think": false (thinking off);
    every other model's body has no "think" field."""
    body = {
        "model": model or settings.DEV_LLM_MODEL,
        "messages": messages,
        "stream": stream,
        "keep_alive": settings.DEV_LLM_KEEP_ALIVE,
        "options": {"temperature": settings.LLM_TEMPERATURE, "num_ctx": settings.DEV_LLM_NUM_CTX},
    }
    if body["model"] in settings.DEV_LLM_THINK_OFF_MODELS:
        body["think"] = False
    return body


def generate(question: str, chunks: list[RetrievedChunk], model: str | None = None,
             corrections: list[tuple[str, str]] | None = None) -> str:
    """Return the full answer in one piece (used for scripted tests)."""
    response = requests.post(f"{settings.OLLAMA_BASE_URL}/api/chat",
                             json=_request_body(build_messages(question, chunks, corrections=corrections), stream=False, model=model),
                             timeout=settings.DEV_LLM_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()["message"]["content"]


def stream_generate(question: str, chunks: list[RetrievedChunk], model: str | None = None,
                    corrections: list[tuple[str, str]] | None = None,
                    transform: Callable[[list[dict]], list[dict]] | None = None) -> Iterator[str]:
    """Yield the answer piece by piece as the model writes it, so the interface can show it immediately.
    Stops at settings.DEV_LLM_MAX_ANSWER_TOKENS, or as soon as the model writes the same line (of at least
    settings.DEV_LLM_REPEAT_MIN_CHARS characters) settings.DEV_LLM_REPEAT_LINE_LIMIT times, and then adds a
    short note saying why it stopped. corrections: the spelling hint pairs (see build_messages). transform, when given,
    is applied to the built messages (coach mode appends its instruction this way); the default sends them as built."""
    messages = build_messages(question, chunks, corrections=corrections)
    body = _request_body(transform(messages) if transform else messages, stream=True, model=model)
    body["options"]["num_predict"] = settings.DEV_LLM_MAX_ANSWER_TOKENS
    line_counts: dict[str, int] = {}
    current_line = ""
    with requests.post(f"{settings.OLLAMA_BASE_URL}/api/chat", json=body,
                       stream=True, timeout=settings.DEV_LLM_TIMEOUT_SECONDS) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            part = json.loads(line)
            text = part.get("message", {}).get("content", "")
            if text:
                yield text
                # Repetition guard: count every finished, long-enough line (whitespace-insensitive).
                current_line += text
                *finished, current_line = current_line.split("\n")
                for done_line in finished:
                    key = " ".join(done_line.split())
                    if len(key) >= settings.DEV_LLM_REPEAT_MIN_CHARS:
                        line_counts[key] = line_counts.get(key, 0) + 1
                        if line_counts[key] >= settings.DEV_LLM_REPEAT_LINE_LIMIT:
                            yield settings.ANSWER_REPEAT_NOTE
                            return            # closing the request makes Ollama stop generating
            if part.get("done"):
                if part.get("done_reason") == "length":
                    yield settings.ANSWER_CUT_NOTE
                break


def complete(messages: list[dict], max_tokens: int, timeout: float, temperature: float = 0.0) -> str:
    """Send arbitrary chat messages and return the reply text (used by the intent classifier).
    max_tokens caps the length of the output; errors are raised to the caller."""
    body = _request_body(messages, stream=False)
    body["options"].update({"temperature": temperature, "num_predict": max_tokens})
    response = requests.post(f"{settings.OLLAMA_BASE_URL}/api/chat", json=body, timeout=timeout)
    response.raise_for_status()
    return response.json()["message"]["content"]


# ---------------------------------------------------------------------------
# Cloud generation (Google Gemini API), for the two free models in the sidebar
# ---------------------------------------------------------------------------

@dataclass
class CloudAnswer:
    """One finished cloud answer and what it took."""
    text: str
    input_tokens: int
    output_tokens: int               # includes any billed reasoning tokens
    reasoning_tokens: int
    latency_seconds: float           # the successful request only, as in the notebook 04 runner
    retries: int                     # failed attempts before the successful one
    retry_wait_seconds: float        # time spent in back-off between those attempts
    cost_usd_list: float             # tokens x list price; a free-tier request is not actually billed


class CloudError(Exception):
    """The cloud model could not answer. str(error) is a complete message for the student and never holds the key."""


@dataclass
class _Reply:
    """What one successful API call returned."""
    text: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    finish: str


_google_clients: dict[str, object] = {}


def is_cloud_model(model: str | None) -> bool:
    """True for the models served by the Google API rather than by Ollama."""
    return model in settings.DEV_CLOUD_MODELS


def _google_api_key() -> str | None:
    """The key from .env, read each time so that a key added while the app runs is picked up. Never logged or displayed."""
    name = settings.API_KEY_ENV_VARS["google"]
    return dotenv_values(settings.PROJECT_ROOT / ".env").get(name) or os.environ.get(name) or None


def _google_client(key: str):
    """One SDK client per key, created on first use. The SDK's own retries stay off, as in the notebook 04 runner."""
    if key not in _google_clients:
        from google import genai
        _google_clients[key] = genai.Client(api_key=key,
                                            http_options={"timeout": settings.NB04_REQUEST_TIMEOUT_SECONDS * 1000})
    return _google_clients[key]


def _cloud_request(model: str, messages: list[dict]) -> dict:
    """The SDK call's arguments, built the same way as in the notebook 04 integration trial. A model that cannot turn thinking
    off gets a reasoning allowance on top of the answer cap, so that reasoning cannot take answer tokens."""
    config = settings.DEV_CLOUD_MODELS[model]
    return {"model": config["model_id"], "contents": messages[1]["content"],
            "config": {"system_instruction": messages[0]["content"], "temperature": config["temperature"],
                       "max_output_tokens": settings.DEV_LLM_MAX_ANSWER_TOKENS + config["reasoning_allowance_tokens"],
                       "thinking_config": {"thinking_level": config["request_options"]["thinking_level"]}}}


def _send_google(key: str, request: dict) -> _Reply:
    """Make one API call. Thinking tokens are reported apart and billed as output."""
    response = _google_client(key).models.generate_content(**request)
    usage = response.usage_metadata
    thoughts = usage.thoughts_token_count or 0
    finish = response.candidates[0].finish_reason.name if response.candidates else "NO_CANDIDATES"
    return _Reply(text=response.text or "", input_tokens=usage.prompt_token_count or 0,
                  output_tokens=(usage.candidates_token_count or 0) + thoughts, reasoning_tokens=thoughts, finish=finish)


def _classify_error(error: Exception) -> tuple[str, int | None, float | None]:
    """Sort an SDK error by its HTTP status: (kind, status, retry_after_seconds). kind is "rate_limited", "temporary",
    "invalid_key", "timeout" or "other". The status is read because the message text may not be safe to show."""
    status = getattr(error, "code", None) or getattr(error, "status_code", None)
    status = status if isinstance(status, int) else None
    name = type(error).__name__.lower()
    if status == 429:
        headers = getattr(getattr(error, "response", None), "headers", None) or {}
        try:
            return "rate_limited", status, float(headers.get("retry-after"))
        except (TypeError, ValueError):
            return "rate_limited", status, None
    if status in (401, 403) or (status == 400 and "api key" in str(error).lower().replace("_", " ")):
        return "invalid_key", status, None
    if status is None and "timeout" in name:
        return "timeout", status, None
    if (status is not None and status >= 500) or (status is None and "connect" in name):
        return "temporary", status, None
    return "other", status, None


def _wait_for_call(work: Callable[[], _Reply], limit: float, tick: Callable[[], None]) -> _Reply:
    """Run one request in a background thread and call tick() on this thread while it runs, so that the interface can show the
    elapsed time. Raises RequestTimedOut at the limit; the abandoned thread cannot be killed but its result is ignored."""
    outcome: dict = {}

    def run():
        try:
            outcome["value"] = work()
        except BaseException as error:      # handed to the waiting caller below
            outcome["error"] = error

    start = time.perf_counter()
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    while True:
        thread.join(settings.DEV_CLOUD_STATUS_TICK_SECONDS)
        if not thread.is_alive():
            break
        if time.perf_counter() - start >= limit:
            raise RequestTimedOut(time.perf_counter() - start)
        tick()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def _call_with_retries(model: str, key: str, request: dict, on_status: Callable[[str], None]) -> tuple[_Reply, float, int, float]:
    """One answer: retry 503s, connection failures and 429s with the notebook 04 runner's back-off. Returns the reply, the
    latency of the successful request, the number of retries and the time spent waiting. Anything else raises CloudError."""
    total = settings.DEV_CLOUD_BACKOFF_MAX_RETRIES + 1
    begun, waited = time.perf_counter(), 0.0

    def elapsed() -> int:
        return int(time.perf_counter() - begun)

    def show_waiting() -> None:
        on_status(settings.DEV_CLOUD_MSG_WAITING.format(model=model, elapsed=elapsed()))

    for attempt in range(total):
        show_waiting()
        start = time.perf_counter()
        try:
            reply = _wait_for_call(lambda: _send_google(key, request), settings.NB04_REQUEST_TIMEOUT_SECONDS, show_waiting)
            return reply, time.perf_counter() - start, attempt, waited
        except RequestTimedOut:
            raise CloudError(settings.DEV_CLOUD_MSG_TIMEOUT.format(
                model=model, seconds=settings.NB04_REQUEST_TIMEOUT_SECONDS)) from None
        except CloudError:
            raise
        except Exception as error:
            kind, status, retry_after = _classify_error(error)
        if kind == "invalid_key":
            raise CloudError(settings.DEV_CLOUD_MSG_INVALID_KEY.format(model=model, status=status)) from None
        if kind == "timeout":
            raise CloudError(settings.DEV_CLOUD_MSG_TIMEOUT.format(
                model=model, seconds=settings.NB04_REQUEST_TIMEOUT_SECONDS)) from None
        if kind == "other":
            detail = f"HTTP {status}" if status else "connection or client error"
            raise CloudError(settings.DEV_CLOUD_MSG_OTHER.format(model=model, detail=detail)) from None
        if attempt == total - 1:
            message = settings.DEV_CLOUD_MSG_RATE_LIMITED if kind == "rate_limited" else settings.DEV_CLOUD_MSG_OVERLOADED
            raise CloudError(message.format(model=model, attempts=total)) from None
        wait = retry_after if retry_after is not None else min(settings.NB04_BACKOFF_BASE_SECONDS * 2 ** attempt,
                                                               settings.NB04_BACKOFF_MAX_SECONDS)
        reason = "free-tier rate limit" if kind == "rate_limited" else "high demand"
        wait_start = time.perf_counter()
        while (remaining := wait - (time.perf_counter() - wait_start)) > 0:
            on_status(settings.DEV_CLOUD_MSG_RETRYING.format(model=model, reason=reason, remaining=int(remaining) + 1,
                                                             attempt=attempt + 2, attempts=total, elapsed=elapsed()))
            time.sleep(min(settings.DEV_CLOUD_STATUS_TICK_SECONDS, remaining))
        waited += wait
    raise AssertionError("unreachable: the last attempt returns or raises")


def _cut_at_repeated_line(text: str) -> tuple[str, bool]:
    """The repetition guard of stream_generate, applied to a finished answer: cut just after a line of at least
    DEV_LLM_REPEAT_MIN_CHARS characters appears for the DEV_LLM_REPEAT_LINE_LIMIT-th time. Returns the text and whether it was cut."""
    counts: dict[str, int] = {}
    position = 0
    for line in text.splitlines(keepends=True):
        position += len(line)
        if not line.endswith(("\n", "\r")):      # same as the streamed guard: only finished lines count
            continue
        key = " ".join(line.split())
        if len(key) >= settings.DEV_LLM_REPEAT_MIN_CHARS:
            counts[key] = counts.get(key, 0) + 1
            if counts[key] >= settings.DEV_LLM_REPEAT_LINE_LIMIT:
                return text[:position].rstrip(), True
    return text, False


def _apply_answer_guards(reply: _Reply) -> str:
    """The same runaway guards as local answers: the repetition guard and the answer-token cap, each with its short note.
    A model with a reasoning allowance can run past the cap, so its text is cut in proportion to its token count."""
    text, repeated = _cut_at_repeated_line(reply.text)
    if repeated:
        return text + settings.ANSWER_REPEAT_NOTE
    answer_tokens = reply.output_tokens - reply.reasoning_tokens
    if answer_tokens > settings.DEV_LLM_MAX_ANSWER_TOKENS:
        keep = int(len(text) * settings.DEV_LLM_MAX_ANSWER_TOKENS / answer_tokens)
        return text[:keep].rsplit(" ", 1)[0].rstrip() + settings.ANSWER_CUT_NOTE
    return text + settings.ANSWER_CUT_NOTE if reply.finish == "MAX_TOKENS" else text


def cloud_generate(question: str, chunks: list[RetrievedChunk], model: str,
                   on_status: Callable[[str], None] | None = None,
                   corrections: list[tuple[str, str]] | None = None,
                   transform: Callable[[list[dict]], list[dict]] | None = None) -> CloudAnswer:
    """Answer with a cloud model, from the same prompt as a local answer. on_status(text) is called on the calling thread
    about twice a second with a line for the student (waiting, or retrying), so a slow reply does not look like a hang.
    Raises CloudError, whose text can be shown as it is, for a missing or rejected key, overload, timeout or an empty reply."""
    key = _google_api_key()
    if not key:
        raise CloudError(settings.DEV_CLOUD_MSG_MISSING_KEY.format(model=model))
    messages = build_messages(question, chunks, corrections=corrections)
    request = _cloud_request(model, transform(messages) if transform else messages)
    reply, latency, retries, waited = _call_with_retries(model, key, request, on_status or (lambda text: None))
    if not reply.text.strip():
        raise CloudError(settings.DEV_CLOUD_MSG_BLOCKED.format(model=model, finish=reply.finish))
    config = settings.DEV_CLOUD_MODELS[model]
    cost = (reply.input_tokens * config["input_usd_per_m"] + reply.output_tokens * config["output_usd_per_m"]) / 1_000_000
    return CloudAnswer(text=_apply_answer_guards(reply), input_tokens=reply.input_tokens, output_tokens=reply.output_tokens,
                       reasoning_tokens=reply.reasoning_tokens, latency_seconds=round(latency, 1), retries=retries,
                       retry_wait_seconds=waited, cost_usd_list=cost)


def check_backend(model: str | None = None) -> tuple[bool, str]:
    """Check that the chosen model can be used. A cloud model only needs its key in .env; this makes no API call, because it
    runs on every page refresh. A local model needs the Ollama server running, with both the answer model (default
    settings.DEV_LLM_MODEL) and the intent classifier's model (settings.DEV_LLM_MODEL) downloaded."""
    model = model or settings.DEV_LLM_MODEL
    if is_cloud_model(model):
        if _google_api_key():
            return True, f"{model} answers through the Google Gemini API. Replies can take a few minutes when Google is busy."
        return False, settings.DEV_CLOUD_MSG_MISSING_KEY.format(model=model)
    try:
        tags = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5).json()
    except requests.RequestException:
        return False, f"Ollama is not running at {settings.OLLAMA_BASE_URL}. Start the Ollama app and reload."
    names = {m["name"] for m in tags.get("models", [])}
    for needed in dict.fromkeys([model, settings.DEV_LLM_MODEL]):
        if needed not in names:
            return False, f"Model {needed} is not downloaded. Run: ollama pull {needed}"
    return True, f"Ollama is running with {model}."
