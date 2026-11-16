"""
Wall-clock timeout for one LLM request, shared by the evaluation runners.

HTTP read timeouts only limit the gap between received bytes, so a slow or stalled request can run far past them. These
helpers limit the total time instead: the request runs in a background thread, and the caller stops waiting when time is
up. A plain function cannot be killed, so its thread is abandoned; a stream is also closed at its next piece, which
ends the request.
"""

import threading
import time
from collections.abc import Callable, Iterable


class RequestTimedOut(Exception):
    """The request passed its time limit. elapsed_seconds is how long the caller waited."""
    def __init__(self, elapsed_seconds: float | None = None, detail: str = ""):
        self.elapsed_seconds = elapsed_seconds
        waited = f"{elapsed_seconds:.1f} s" if elapsed_seconds is not None else "the client's limit"
        super().__init__(f"timeout after {waited}" + (f" ({detail})" if detail else ""))


def call_with_timeout(work: Callable, seconds: float, stop: threading.Event | None = None):
    """Return work(), or raise RequestTimedOut after the given seconds. Errors raised by work() reach the caller.
    If a stop event is given, it is set on timeout so that work() can end itself."""
    outcome = {}

    def run():
        try:
            outcome["value"] = work()
        except BaseException as error:      # handed to the waiting caller below
            outcome["error"] = error

    start = time.perf_counter()
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        if stop:
            stop.set()
        raise RequestTimedOut(time.perf_counter() - start)
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def collect_stream(make_stream: Callable[[], Iterable[str]], seconds: float) -> str:
    """Join a streamed answer into one string, within the time limit. After a timeout the stream is closed at its next
    piece, so an abandoned answer does not keep generating."""
    stop = threading.Event()

    def work():
        stream = make_stream()
        pieces = []
        try:
            for piece in stream:
                if stop.is_set():
                    return None
                pieces.append(piece)
        finally:
            close = getattr(stream, "close", None)
            if close:
                close()
        return "".join(pieces)

    return call_with_timeout(work, seconds, stop)
