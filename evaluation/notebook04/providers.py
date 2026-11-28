"""
Model adapters for notebook 04: one way to send the same chat messages to every candidate.

Each API style has three parts:
- build_request: turns the shared messages into that API's request, without touching the network;
- request_messages: reads the messages back out of a built request, so check_parity.py can prove that what would
  be SENT is identical for every model;
- send: makes the call and returns a GenerationResult.

The official SDKs are used directly, with no LangChain (author's decision, Report TODO 27). SDK retries are turned off
because run_generation.py does the retrying, so that waiting can be logged separately from latency.
Claude's refusal fallback option is deliberately not used: it would let a different model answer for the candidate.
"""

import os
import re
from dataclasses import dataclass

from config import settings
from src.core.timeouts import RequestTimedOut


class RateLimited(Exception):
    """HTTP 429. retry_after is the host's requested wait in seconds, when it sends one, and retry_after_source says
    where it came from: "header" (Retry-After) or "body" (Google's retryDelay)."""
    def __init__(self, message: str, retry_after: float | None = None, retry_after_source: str | None = None):
        super().__init__(message)
        self.retry_after = retry_after
        self.retry_after_source = retry_after_source


class TemporaryFailure(Exception):
    """Server error (5xx) or connection failure: worth retrying."""


class AccessBlocked(Exception):
    """401, 403 or 404: bad key, no access (e.g. Enterprise-only) or unknown model. Retrying cannot help."""


@dataclass
class GenerationResult:
    """What one successful API call returned. output_tokens includes any billed reasoning tokens. cached_input_tokens is the
    part of input_tokens the host served from its prompt cache (billed at a lower rate); cost still counts every input token at
    the full list price, so it is recorded only for the cost summary's "actually charged" column."""
    text: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    finish: str          # "stop", "length", "refusal" or the host's own word for anything else
    served_model: str | None
    cached_input_tokens: int = 0
    reasoning_tokens_estimated: bool = False     # True when the host gave no figure and it was split from the text


def api_key(candidate: dict) -> str | None:
    """The candidate's key from the environment (.env is loaded by the runner). Never stored in code or caches."""
    return os.environ.get(settings.API_KEY_ENV_VARS[candidate["key"]])


# ---------------------------------------------------------------------------
# Building requests (no network)
# ---------------------------------------------------------------------------

def _split_messages(messages: list[dict]) -> tuple[str, str]:
    """The shared prompt is always exactly [system, user]."""
    if [m["role"] for m in messages] != ["system", "user"]:
        raise ValueError(f"Expected system + user messages, got {[m['role'] for m in messages]}")
    return messages[0]["content"], messages[1]["content"]


def build_request(candidate: dict, messages: list[dict]) -> dict:
    """The keyword arguments for this candidate's SDK call."""
    system, user = _split_messages(messages)
    api = candidate["api"]
    if api == "gemini":
        return {"model": candidate["model_id"], "contents": user,
                "config": {"system_instruction": system, "temperature": candidate["temperature"],
                           # Thinking shares this budget on Gemini, so the allowance keeps 800 tokens for the answer.
                           "max_output_tokens": settings.NB04_MAX_ANSWER_TOKENS
                           + candidate["reasoning_allowance_tokens"],
                           "thinking_config": {"thinking_level": candidate["request_options"]["thinking_level"]}}}
    if api == "openai_compatible":
        # A model that always reasons (Mistral Large 4) gets its reasoning allowance on top, as on Gemini; 0 for the others.
        request = {"model": candidate["model_id"],
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                   candidate["max_tokens_param"]: settings.NB04_MAX_ANSWER_TOKENS + candidate["reasoning_allowance_tokens"]}
    elif api == "anthropic":
        request = {"model": candidate["model_id"], "system": system,
                   "messages": [{"role": "user", "content": user}],
                   "max_tokens": settings.NB04_MAX_ANSWER_TOKENS}
    else:
        raise ValueError(f"Unknown api style: {api}")
    if candidate["temperature"] is not None:
        request["temperature"] = candidate["temperature"]
    request.update(candidate["request_options"])
    return request


def request_messages(candidate: dict, request: dict) -> list[dict]:
    """Read the [system, user] messages back out of a built request (used by the parity check)."""
    api = candidate["api"]
    if api == "openai_compatible":
        return [{"role": m["role"], "content": m["content"]} for m in request["messages"]]
    if api == "anthropic":
        return [{"role": "system", "content": request["system"]},
                {"role": "user", "content": request["messages"][0]["content"]}]
    if api == "gemini":
        return [{"role": "system", "content": request["config"]["system_instruction"]},
                {"role": "user", "content": request["contents"]}]
    raise ValueError(f"Unknown api style: {api}")


# ---------------------------------------------------------------------------
# Sending requests (network; only run_generation.py reaches this)
# ---------------------------------------------------------------------------

_clients: dict[str, object] = {}


def _client(candidate: dict):
    """One SDK client per host and key, created on first use, with SDK retries off."""
    cache_key = f"{candidate['api']}|{candidate['base_url']}|{candidate['key']}"
    if cache_key in _clients:
        return _clients[cache_key]
    key = api_key(candidate)
    timeout = settings.NB04_REQUEST_TIMEOUT_SECONDS
    if candidate["api"] == "openai_compatible":
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=candidate["base_url"], max_retries=0, timeout=timeout)
    elif candidate["api"] == "anthropic":
        from anthropic import Anthropic
        client = Anthropic(api_key=key, max_retries=0, timeout=timeout)
    else:
        from google import genai
        client = genai.Client(api_key=key, http_options={"timeout": timeout * 1000})
    _clients[cache_key] = client
    return client


def _status_code(error: Exception) -> int | None:
    """The HTTP status of an SDK error, or None when there was no HTTP response (a timeout or dropped connection)."""
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    return status if isinstance(status, int) else None


def _retry_delay_from_body(error: Exception) -> float | None:
    """Google's requested wait in seconds, from the RetryInfo entry in the error body (retryDelay, e.g. "38s"), or None.
    Google sends this in the body, so a client that reads only the Retry-After header would miss it."""
    body = getattr(error, "details", None)
    inner = body.get("error", body) if isinstance(body, dict) else None
    entries = inner.get("details") if isinstance(inner, dict) else None
    for entry in entries if isinstance(entries, list) else []:
        delay = entry.get("retryDelay") if isinstance(entry, dict) else None
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)s\s*", delay) if isinstance(delay, str) else None
        if match:
            return float(match.group(1))
    return None


def is_free_tier_quota(error: Exception) -> bool:
    """True for Google's free-tier quota 429 (metric generate_content_free_tier_requests, quota id ...-FreeTier)."""
    text = str(error.__cause__ or error).lower() + " " + str(error).lower()
    return "free_tier" in text or "freetier" in text


def describe_error(error: Exception) -> dict:
    """What the runner saves about one failed call: the SDK's own error type, the HTTP status where there was one, and its
    text with any API key removed and the length capped. Enough to tell a 5xx from a dropped connection later."""
    source = error.__cause__ or error       # the adapter's translated errors keep the SDK's original as their cause
    text = str(source)
    for variable in settings.API_KEY_ENV_VARS.values():
        key = os.environ.get(variable)
        if key:
            text = text.replace(key, "[redacted]")
    # Hosts echo masked fragments of a rejected key (DeepSeek "****1ed5", OpenAI "sk-ab***xyz"): never saved either.
    text = re.sub(r"\*{2,}[A-Za-z0-9_\-]{1,12}", "[redacted]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_\-*]{2,}", "[redacted]", text)
    return {"error_type": type(source).__name__, "status_code": _status_code(source),
            "error_text": text[:settings.NB04_ERROR_TEXT_MAX_CHARS]}


def _translate_error(error: Exception) -> Exception:
    """Map any SDK's exception to RateLimited / AccessBlocked / TemporaryFailure by its HTTP status.
    The SDKs name their exceptions differently, so the status code is read rather than matching each class."""
    status = _status_code(error)
    if status == 429:
        headers = getattr(getattr(error, "response", None), "headers", None) or {}
        try:
            return RateLimited(str(error), float(headers.get("retry-after")), "header")
        except (TypeError, ValueError):
            delay = _retry_delay_from_body(error)
            return RateLimited(str(error), delay, "body" if delay is not None else None)
    if status in (401, 403, 404):
        return AccessBlocked(f"HTTP {status}: {error}")
    name = type(error).__name__.lower()
    if status is None and "timeout" in name:     # the SDK's own limit: handled like the runner's wall-clock limit
        return RequestTimedOut(detail=type(error).__name__)
    if (status is not None and status >= 500) or (status is None and "connection" in name):
        return TemporaryFailure(f"{type(error).__name__}: {error}")
    return error      # anything else (e.g. a 400 for a rejected setting) is recorded against the question


def _field(obj, name: str):
    """A field of an SDK object or of a plain dict (hosts' extra fields can arrive as either)."""
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _message_text(content) -> str:
    """The answer text of a chat message. Most hosts return a string. Mistral's reasoning models return a list of chunks, a
    "thinking" chunk before the "text" chunk; only the text chunks are the answer, so the thinking is never shown or judged."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(_field(part, "text") or "" for part in content if _field(part, "type") == "text")


def _thinking_text(content) -> str:
    """The text inside Mistral's "thinking" chunks ([{"type": "thinking", "thinking": [{"type": "text", "text": ...}]}]),
    used only to estimate its token count when the host reports no reasoning-token figure."""
    if not isinstance(content, list):
        return ""
    out = []
    for part in content:
        if _field(part, "type") != "thinking":
            continue
        inner = _field(part, "thinking")
        out += [inner] if isinstance(inner, str) else [_field(p, "text") or "" for p in inner or []]
    return "".join(out)


def _reasoning_tokens(usage, content, output_tokens: int) -> tuple[int, bool]:
    """Reasoning tokens inside output_tokens, and whether the figure is an estimate. The host's own figure is used when it
    reports one (OpenAI-style completion_tokens_details.reasoning_tokens). Otherwise, when the message carries thinking chunks,
    the output tokens are split in proportion to the characters of thinking and answer text, so the 800-token answer cap is
    applied to the answer alone. That split is an estimate and is recorded as one; cost always uses every output token."""
    details = _field(usage, "completion_tokens_details")
    reported = _field(details, "reasoning_tokens") if details is not None else None
    if reported:
        return int(reported), False
    thinking, answer = len(_thinking_text(content)), len(_message_text(content))
    if not thinking:
        return 0, False
    return round(output_tokens * thinking / (thinking + answer)), True


def _cached_tokens(usage) -> int:
    """Prompt tokens served from the host's cache: OpenAI-style prompt_tokens_details.cached_tokens, or DeepSeek's
    prompt_cache_hit_tokens. 0 when the host reports neither."""
    details = _field(usage, "prompt_tokens_details")
    cached = _field(details, "cached_tokens") if details is not None else None
    return int(cached or _field(usage, "prompt_cache_hit_tokens") or 0)


def _send_openai(candidate: dict, request: dict) -> GenerationResult:
    """OpenAI, DeepSeek and Mistral all accept this request style (DeepSeek's thinking switch rides in extra_body)."""
    response = _client(candidate).chat.completions.create(**request)
    choice = response.choices[0]
    content = choice.message.content
    reasoning, estimated = _reasoning_tokens(response.usage, content, response.usage.completion_tokens)
    finish = {"length": "length", "stop": "stop", "content_filter": "refusal"}
    return GenerationResult(
        text=_message_text(content),
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,      # already includes reasoning tokens
        reasoning_tokens=reasoning,
        finish=finish.get(choice.finish_reason, choice.finish_reason),
        served_model=response.model,
        cached_input_tokens=_cached_tokens(response.usage),
        reasoning_tokens_estimated=estimated)


def _send_anthropic(candidate: dict, request: dict) -> GenerationResult:
    """Anthropic Messages API. The stop reason decides the outcome, because a refusal returns HTTP 200."""
    response = _client(candidate).messages.create(**request)
    text = "".join(block.text for block in response.content if block.type == "text")
    finish = {"end_turn": "stop", "max_tokens": "length", "refusal": "refusal"}
    return GenerationResult(
        text=text,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        reasoning_tokens=0,                                  # not reported separately; billed inside output_tokens
        finish=finish.get(response.stop_reason, response.stop_reason),
        served_model=response.model,
        cached_input_tokens=getattr(response.usage, "cache_read_input_tokens", None) or 0)


def _send_gemini(candidate: dict, request: dict) -> GenerationResult:
    """Gemini API (Gemini and Gemma models). Thinking tokens are reported apart and billed as output."""
    response = _client(candidate).models.generate_content(**request)
    usage = response.usage_metadata
    thoughts = usage.thoughts_token_count or 0
    reason = response.candidates[0].finish_reason.name if response.candidates else "NO_CANDIDATES"
    finish = {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "refusal"}
    return GenerationResult(
        text=response.text or "",
        input_tokens=usage.prompt_token_count or 0,
        output_tokens=(usage.candidates_token_count or 0) + thoughts,
        reasoning_tokens=thoughts,
        finish=finish.get(reason, reason),
        served_model=getattr(response, "model_version", None),
        cached_input_tokens=getattr(usage, "cached_content_token_count", None) or 0)


_SENDERS = {"openai_compatible": _send_openai, "anthropic": _send_anthropic, "gemini": _send_gemini}


def send(candidate: dict, request: dict) -> GenerationResult:
    """Make one API call. Raises RateLimited, TemporaryFailure or AccessBlocked, or the SDK's own error."""
    try:
        return _SENDERS[candidate["api"]](candidate, request)
    except Exception as error:
        raise _translate_error(error) from error
