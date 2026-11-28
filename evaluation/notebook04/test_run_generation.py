"""
Offline tests for the timing and retry behaviour of run_generation.py, with a fake model. NOTHING HERE CAN REACH A REAL API:
every call goes to a fake providers.send, the SDK client factory is blocked, .env is never loaded, the keys are fake strings,
sleeping is recorded instead of done, and all caches go to temporary folders. The real contexts are only read.

Usage (from the project root):
    venv\\Scripts\\python evaluation\\notebook04\\test_run_generation.py
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for extra_path in (PROJECT_ROOT, Path(__file__).resolve().parent):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

import providers  # noqa: E402
import run_generation  # noqa: E402
from config import settings  # noqa: E402
from src.core.timeouts import RequestTimedOut  # noqa: E402

MODEL = "gemma-4-31b-it"
CANDIDATE = settings.NB04_CANDIDATES[MODEL]
OK = providers.GenerationResult("An answer [S06-0204].", 1000, 100, 0, "stop", MODEL)


def timed_out():
    return RequestTimedOut(detail="fake")


def rate_limited(retry_after=None):
    return providers.RateLimited("429", retry_after)


def overloaded():
    return providers.TemporaryFailure("503")


class RunnerTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.raw = Path(self.tmp.name) / "raw"
        self.outcomes, self.sleeps, self.calls = [], [], 0

        def send(candidate, request):
            self.calls += 1
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        def blocked_client(candidate):
            raise AssertionError("a real SDK client was requested in an offline test")

        patches = [mock.patch.object(settings, "NB04_RAW_DIR", self.raw),
                   mock.patch.object(settings, "NB04_JUDGE_DIR", Path(self.tmp.name) / "judge"),
                   mock.patch.object(providers, "send", send), mock.patch.object(providers, "_client", blocked_client),
                   mock.patch.object(run_generation.time, "sleep", self.sleeps.append),
                   # Pacing off here, so self.sleeps holds back-off waits only; FreeTierPacing tests it on its own.
                   mock.patch.object(settings, "NB04_FREE_TIER_MIN_INTERVAL_SECONDS", {}),
                   mock.patch("dotenv.load_dotenv", lambda *a, **k: False),
                   mock.patch.dict(os.environ, {"GOOGLE_API_KEY": "fake-key-not-real"}),
                   mock.patch.object(run_generation.check_parity, "run_checks", lambda **kwargs: [])]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def run_main(self, *argv):
        out = io.StringIO()
        code = 0
        with mock.patch.object(sys, "argv", ["run_generation.py", *argv]), contextlib.redirect_stdout(out):
            try:
                run_generation.main()
            except SystemExit as stop:
                code = stop.code or 0
        return code, out.getvalue()

    def generate(self, *outcomes, only="Q01"):
        self.outcomes = list(outcomes)
        return self.run_main("--model", MODEL, "--only", only)

    def cache(self):
        return json.loads((self.raw / f"{MODEL}.json").read_text(encoding="utf-8"))

    def timing(self):
        return run_generation.timing_summary(self.cache())


class TimedOutAttempts(RunnerTestCase):
    def test_timeout_is_kept_after_a_later_attempt_succeeds(self):
        self.generate(timed_out())
        first = self.cache()["results"]["Q01"]
        self.assertEqual(first["stopped_by"], "timeout")
        self.assertEqual(len(first["timed_out_attempts"]), 1)
        self.generate(OK)
        record = self.cache()["results"]["Q01"]
        self.assertIsNone(record["error"], "the later attempt succeeded")
        self.assertEqual(record["stopped_by"], "finished")
        self.assertEqual(len(record["timed_out_attempts"]), 1, "the stall stays on the record")
        self.assertIn("elapsed_seconds", record["timed_out_attempts"][0])

    def test_several_timeouts_accumulate(self):
        for _ in range(3):
            self.generate(timed_out())
        self.generate(OK)
        self.assertEqual(len(self.cache()["results"]["Q01"]["timed_out_attempts"]), 3)
        self.assertEqual(self.timing()["timed_out_attempts"], 3)

    def test_clean_answer_has_an_empty_list(self):
        self.generate(OK)
        self.assertEqual(self.cache()["results"]["Q01"]["timed_out_attempts"], [])

    def test_report_gives_all_three_figures_by_hand(self):
        self.generate(OK, OK, OK, only="Q01,Q02,Q03")
        path = self.raw / f"{MODEL}.json"
        cache = self.cache()
        for key, seconds in (("Q01", 10.0), ("Q02", 20.0), ("Q03", 90.0)):
            cache["results"][key]["latency_seconds"] = seconds
        cache["results"]["Q02"]["timed_out_attempts"] = [{"at": "x", "elapsed_seconds": 180.0, "waits": {}},
                                                         {"at": "y", "elapsed_seconds": 181.5, "waits": {}}]
        path.write_text(json.dumps(cache), encoding="utf-8")
        timing = self.timing()
        self.assertEqual(timing["latency_mean_seconds"], 40.0, "the timed-out attempts stay out of latency")
        self.assertEqual(timing["latency_p95_seconds"], 90.0)
        self.assertEqual(timing["timed_out_attempts"], 2)
        self.assertEqual(timing["timed_out_seconds"], 361.5)
        self.assertEqual(timing["total_seconds_including_abandoned"], 120.0 + 361.5)
        print(f"\n  [by hand] {timing}")

    def test_unresolved_timeout_counts_once_and_its_waits_once(self):
        self.generate(overloaded(), timed_out())
        timing = self.timing()
        self.assertEqual(timing["timed_out_attempts"], 1)
        self.assertEqual(timing["temporary_failure_retries"], 1, "not counted from both the record and the attempt")
        self.assertEqual(timing["temporary_failure_wait_seconds"], 5)
        self.assertEqual(timing["question_answers"], 0)
        self.assertIsNone(timing["latency_mean_seconds"])

    def test_waits_of_a_resolved_timeout_survive(self):
        self.generate(overloaded(), timed_out())
        self.generate(OK)
        self.assertEqual(self.timing()["temporary_failure_wait_seconds"], 5)

    def test_probes_are_outside_the_timing(self):
        self.generate(OK, OK, OK, only="G01")
        self.assertEqual(self.timing()["question_answers"], 0)

    def test_report_prints_timing_and_calls_nothing(self):
        self.generate(OK)
        with mock.patch.object(providers, "send", side_effect=AssertionError("the report must not call a model")):
            code, out = self.run_main("--report")
        self.assertEqual(code, 0)
        data = json.loads(out)
        row = data["timing_question_answers"][0]
        for field in ("latency_mean_seconds", "latency_p95_seconds", "timed_out_attempts", "total_seconds_including_abandoned"):
            self.assertIn(field, row)
        self.assertIn("truncation", data)


class Backoff(RunnerTestCase):
    def test_429_uses_the_longer_back_off(self):
        self.generate(rate_limited(), rate_limited(), rate_limited(), OK)
        self.assertEqual(self.sleeps, [60, 120, 240])

    def test_429_back_off_is_capped(self):
        self.generate(*[rate_limited() for _ in range(5)], OK)
        self.assertEqual(self.sleeps, [60, 120, 240, 300, 300])

    def test_retry_after_is_respected(self):
        self.generate(rate_limited(retry_after=17), OK)
        self.assertEqual(self.sleeps, [17])
        self.assertEqual(self.cache()["results"]["Q01"]["rate_limit_wait_seconds"], 17)

    def test_retry_after_longer_than_the_cap_is_still_respected(self):
        self.generate(rate_limited(retry_after=900), OK)
        self.assertEqual(self.sleeps, [900])

    def test_a_very_long_host_wait_is_cut_to_the_ceiling(self):
        # Google's daily-reset delay (57,954 s on 27 Nov) is re-checked after the ceiling; the host's figure stays in the log.
        self.generate(rate_limited(retry_after=57954), OK)
        self.assertEqual(self.sleeps, [settings.NB04_RATE_LIMIT_HOST_WAIT_MAX_SECONDS])
        record = self.cache()["results"]["Q01"]
        self.assertEqual(record["rate_limit_wait_seconds"], settings.NB04_RATE_LIMIT_HOST_WAIT_MAX_SECONDS)
        entry = record["call_log"][0]
        self.assertEqual((entry["retry_after_seconds"], entry["waited_after_seconds"]),
                         (57954, settings.NB04_RATE_LIMIT_HOST_WAIT_MAX_SECONDS))
        self.assertEqual(self.cache()["run_settings"]["rate_limit_host_wait_max_seconds"],
                         settings.NB04_RATE_LIMIT_HOST_WAIT_MAX_SECONDS)

    def test_5xx_back_off_is_unchanged(self):
        self.generate(*[overloaded() for _ in range(7)], OK)
        self.assertEqual(self.sleeps, [5, 10, 20, 40, 80, 120, 120])

    def test_429_and_5xx_waits_are_recorded_apart(self):
        self.generate(overloaded(), overloaded(), rate_limited(), OK)
        record = self.cache()["results"]["Q01"]
        self.assertEqual((record["temporary_failure_retries"], record["temporary_failure_wait_seconds"]), (2, 15))
        self.assertEqual((record["rate_limit_retries"], record["rate_limit_wait_seconds"]), (1, 60))

    def test_a_429_after_503s_starts_at_its_own_base(self):
        self.generate(overloaded(), overloaded(), overloaded(), rate_limited(), OK)
        self.assertEqual(self.sleeps, [5, 10, 20, 60])

    def test_ten_retries_are_allowed(self):
        self.assertEqual(settings.NB04_BACKOFF_MAX_RETRIES, 10)
        code, _ = self.generate(*[overloaded() for _ in range(10)], OK)
        self.assertEqual((code, self.calls), (0, 11))
        self.assertEqual(self.cache()["results"]["Q01"]["temporary_failure_retries"], 10)
        self.assertEqual(sum(self.sleeps), 755)

    def test_retries_exhausted_stops_and_keeps_the_waits(self):
        code, out = self.generate(*[overloaded() for _ in range(11)])
        self.assertEqual((code, self.calls), (run_generation.EXIT_BLOCKED, 11))
        self.assertIn("not cleared after 10 retries", out)
        cache = self.cache()
        self.assertEqual(cache["results"], {}, "no answer is invented")
        self.assertEqual(len(cache["retries_exhausted"]), 1)
        timing = self.timing()
        self.assertEqual((timing["retries_exhausted"], timing["temporary_failure_wait_seconds"]), (1, 755))

    def test_failed_attempt_time_is_recorded(self):
        self.generate(overloaded(), OK)
        self.assertIn("failed_attempt_seconds", self.cache()["results"]["Q01"])

    def test_retries_do_not_change_what_is_billed(self):
        self.generate(overloaded(), rate_limited(), OK)
        self.assertEqual(self.cache()["spend"]["billed_calls"], 1)


class Settings(RunnerTestCase):
    def test_timeout_is_180(self):
        self.assertEqual(settings.NB04_REQUEST_TIMEOUT_SECONDS, 180)

    def test_timeout_is_resume_locked(self):
        self.generate(OK)
        calls = self.calls
        with mock.patch.object(settings, "NB04_REQUEST_TIMEOUT_SECONDS", 120):
            code, out = self.generate(OK, only="Q02")
        self.assertEqual(code, run_generation.EXIT_SETUP)
        self.assertIn("request_timeout_seconds", out)
        self.assertEqual(self.calls, calls, "no model was called")

    def test_back_off_settings_are_not_resume_locked(self):
        self.generate(OK)
        with mock.patch.object(settings, "NB04_BACKOFF_MAX_RETRIES", 4):
            code, _ = self.generate(OK, only="Q02")
        self.assertEqual(code, 0)

    def test_back_off_is_recorded_in_the_run_settings(self):
        self.generate(OK)
        backoff = self.cache()["run_settings"]["backoff"]
        self.assertEqual(backoff, {"max_retries": 10, "temporary_failure_base_and_cap_seconds": [5, 120],
                                   "rate_limit_base_and_cap_seconds": [60, 300]})


class RealSdkErrorTypes(unittest.TestCase):
    """The adapter's error mapping, on the real exception classes of the SDK that serves the Google models."""

    def translate(self, code, headers=None):
        from google.genai import errors
        kind = errors.ClientError if code < 500 else errors.ServerError
        body = {"error": {"code": code, "message": "x", "status": "S"}}
        return providers._translate_error(kind(code, body, response=SimpleNamespace(headers=headers or {})))

    def test_429_is_rate_limited(self):
        self.assertIsInstance(self.translate(429), providers.RateLimited)
        self.assertIsNone(self.translate(429).retry_after)

    def test_retry_after_header_is_read(self):
        self.assertEqual(self.translate(429, {"retry-after": "17"}).retry_after, 17.0)

    def test_500_503_504_are_temporary(self):
        for code in (500, 503, 504):
            self.assertIsInstance(self.translate(code), providers.TemporaryFailure, code)

    def test_404_is_blocked(self):
        self.assertIsInstance(self.translate(404), providers.AccessBlocked)


class GoogleRetryDelay(unittest.TestCase):
    """retryDelay in the error body, in the shape Google's API returns it, on the real SDK exception class."""

    def rate_limited(self, details, headers=None):
        from google.genai import errors
        body = {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED", "details": details}}
        return providers._translate_error(errors.ClientError(429, body, response=SimpleNamespace(headers=headers or {})))

    RETRY_INFO = {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "38s"}
    QUOTA_INFO = {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": []}

    def test_body_delay_is_read(self):
        error = self.rate_limited([self.QUOTA_INFO, self.RETRY_INFO])
        self.assertEqual((error.retry_after, error.retry_after_source), (38.0, "body"))

    def test_fractional_delay_is_read(self):
        self.assertEqual(self.rate_limited([{"retryDelay": "1.5s"}]).retry_after, 1.5)

    def test_header_is_used_when_both_are_sent(self):
        error = self.rate_limited([self.RETRY_INFO], {"retry-after": "17"})
        self.assertEqual((error.retry_after, error.retry_after_source), (17.0, "header"))

    def test_no_hint_leaves_the_back_off_to_decide(self):
        for details in ([], [self.QUOTA_INFO], [{"retryDelay": "soon"}], "not a list", [None]):
            error = self.rate_limited(details)
            self.assertIsInstance(error, providers.RateLimited, details)
            self.assertIsNone(error.retry_after, details)
            self.assertIsNone(error.retry_after_source, details)


class CallLog(RunnerTestCase):
    """Every call is timestamped and every failure can be identified later."""

    def sdk_error(self, code):
        """A translated error whose cause is the real SDK exception, as providers.send raises it."""
        from google.genai import errors
        kind = errors.ClientError if code < 500 else errors.ServerError
        original = kind(code, {"error": {"code": code, "message": "busy", "status": "UNAVAILABLE"}}, response=SimpleNamespace(headers={}))
        translated = providers._translate_error(original)
        translated.__cause__ = original
        return translated

    def log(self, key="Q01"):
        return self.cache()["results"][key]["call_log"]

    def test_a_clean_answer_logs_one_timestamped_call(self):
        self.generate(OK)
        [call] = self.log()
        self.assertEqual(call["outcome"], "success")
        self.assertRegex(call["started_at"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00$")
        self.assertRegex(call["finished_at"], r"\+00:00$")
        self.assertNotIn("error_text", call)

    def test_each_retry_saves_its_error_type_status_and_text(self):
        self.generate(self.sdk_error(503), self.sdk_error(500), OK)
        first, second, last = self.log()
        self.assertEqual((first["outcome"], first["error_type"], first["status_code"]), ("temporary_failure", "ServerError", 503))
        self.assertEqual(second["status_code"], 500)
        self.assertIn("busy", first["error_text"])
        self.assertEqual((first["waited_after_seconds"], second["waited_after_seconds"]), (5, 10))
        self.assertEqual(last["outcome"], "success")

    def test_a_dropped_connection_is_told_apart_from_a_5xx(self):
        original = ConnectionResetError("reset by peer")
        translated = providers._translate_error(original)
        translated.__cause__ = original
        self.generate(translated, OK)
        call = self.log()[0]
        self.assertEqual((call["error_type"], call["status_code"]), ("ConnectionResetError", None))

    def test_a_429_saves_where_its_wait_came_from(self):
        self.generate(providers.RateLimited("429", 38.0, "body"), OK)
        call = self.log()[0]
        self.assertEqual(self.sleeps, [38.0])
        self.assertEqual((call["retry_after_seconds"], call["retry_after_source"], call["waited_after_seconds"]), (38.0, "body", 38.0))

    def test_an_api_key_never_reaches_the_cache(self):
        self.generate(providers.TemporaryFailure("503 for key fake-key-not-real"), OK)
        self.assertNotIn("fake-key-not-real", (self.raw / f"{MODEL}.json").read_text(encoding="utf-8"))
        self.assertIn("[redacted]", self.log()[0]["error_text"])

    def test_long_error_text_is_capped(self):
        self.generate(providers.TemporaryFailure("x" * 5000), OK)
        self.assertEqual(len(self.log()[0]["error_text"]), settings.NB04_ERROR_TEXT_MAX_CHARS)

    def test_the_log_survives_a_later_run(self):
        self.generate(timed_out())
        self.generate(OK)
        self.assertEqual([c["outcome"] for c in self.log()], ["timeout", "success"])
        self.assertIn("started_at", self.cache()["results"]["Q01"]["timed_out_attempts"][0])
        self.assertNotIn("call_log", self.cache()["results"]["Q01"]["timed_out_attempts"][0]["waits"])

    def test_an_error_not_retried_is_logged_and_later_kept(self):
        self.generate(ValueError("400 rejected"))
        self.generate(OK)
        first, last = self.log()
        self.assertEqual((first["outcome"], first["error_type"], last["outcome"]), ("not_retried", "ValueError", "success"))

    def test_retries_used_up_keeps_every_call(self):
        code, _ = self.generate(*[overloaded() for _ in range(settings.NB04_BACKOFF_MAX_RETRIES + 1)])
        self.assertEqual(code, run_generation.EXIT_BLOCKED)
        entry = self.cache()["retries_exhausted"][0]
        self.assertEqual(len(entry["call_log"]), settings.NB04_BACKOFF_MAX_RETRIES + 1)
        self.assertNotIn("call_log", entry["waits"])

    def test_an_access_block_is_logged(self):
        code, _ = self.generate(providers.AccessBlocked("HTTP 403"))
        self.assertEqual(code, run_generation.EXIT_BLOCKED)
        self.assertEqual(self.cache()["access_blocked"][0]["call_log"][0]["outcome"], "not_retried")


class TrueElapsedAndWindow(RunnerTestCase):
    @staticmethod
    def call(start, end, seconds, waited=0.0):
        return {"started_at": f"2026-11-09T{start}+00:00", "finished_at": f"2026-11-09T{end}+00:00", "elapsed_seconds": seconds,
                "outcome": "x", "waited_after_seconds": waited}

    def test_the_saved_gemini_trial_gives_990_14_by_hand(self):
        """The 8 Nov re-run's real figures, in the format saved before call_log existed."""
        none = {"failed_attempt_seconds": 0.0, "rate_limit_wait_seconds": 0.0, "temporary_failure_wait_seconds": 0.0}
        def record(latency, failed, waited, timeouts=()):
            return {"kind": "question", "error": None, "latency_seconds": latency, "failed_attempt_seconds": failed,
                    "temporary_failure_wait_seconds": waited, "timed_out_attempts": list(timeouts)}
        cache = {"results": {"Q01": record(175.678, 0.0, 0.0, [{"elapsed_seconds": 180.0, "waits": none}]),
                             "Q02": record(126.653, 178.25, 5.0), "Q03": record(141.51, 178.05, 5.0)}}
        summary = run_generation.timing_summary(cache)
        self.assertEqual(summary["successful_request_seconds"], 443.84)
        self.assertEqual(summary["total_seconds_including_abandoned"], 623.84)
        self.assertEqual(summary["true_elapsed_seconds"], 990.14)

    def test_true_elapsed_adds_every_call_and_wait_by_hand(self):
        log = [self.call("01:00:00", "01:03:00", 180.0, 5.0), self.call("01:03:05", "01:04:05", 60.0, 10.0),
               self.call("01:04:15", "01:04:55", 40.0)]
        cache = {"results": {"Q01": {"kind": "question", "error": None, "latency_seconds": 40.0, "failed_attempt_seconds": 240.0,
                                     "temporary_failure_wait_seconds": 15.0, "timed_out_attempts": [], "call_log": log}}}
        self.assertEqual(run_generation.timing_summary(cache)["true_elapsed_seconds"], 295.0)

    def test_a_logged_timeout_is_not_counted_twice(self):
        self.generate(timed_out())
        self.generate(OK)
        cache = self.cache()
        logged = sum(c["elapsed_seconds"] for c in cache["results"]["Q01"]["call_log"])
        self.assertAlmostEqual(run_generation.timing_summary(cache)["true_elapsed_seconds"], logged, places=1)

    def test_retries_used_up_counts_towards_true_elapsed(self):
        cache = {"results": {}, "retries_exhausted": [{"kind": "question", "waits": {}, "call_log": [self.call("01:00:00", "01:01:00", 60.0, 5.0)]}]}
        self.assertEqual(run_generation.timing_summary(cache)["true_elapsed_seconds"], 65.0)

    def test_the_window_spans_first_start_to_last_finish(self):
        cache = {"results": {"Q01": {"call_log": [self.call("02:00:00", "02:01:00", 60.0)]},
                             "G01#r1": {"call_log": [self.call("05:00:00", "05:01:30", 90.0)]},
                             "Q02": {"call_log": [self.call("01:00:00", "01:01:00", 60.0)]}},
                 "retries_exhausted": [{"call_log": [self.call("06:00:00", "06:02:00", 120.0)]}]}
        window = run_generation.run_window(cache)
        self.assertEqual(window["first_call_utc"], "2026-11-09T01:00:00+00:00")
        self.assertEqual(window["last_call_utc"], "2026-11-09T06:02:00+00:00")
        self.assertEqual((window["calls_timestamped"], window["results_without_timestamps"]), (4, 0))

    def test_older_results_are_counted_not_guessed(self):
        window = run_generation.run_window({"results": {"Q01": {"error": None}, "Q02": {"error": None}}})
        self.assertEqual((window["first_call_utc"], window["last_call_utc"], window["results_without_timestamps"]), (None, None, 2))

    def test_report_prints_the_window_and_calls_nothing(self):
        self.generate(OK)
        calls = self.calls
        code, out = self.run_main("--report")
        report = json.loads(out)
        self.assertEqual((code, self.calls), (0, calls))
        self.assertEqual(report["run_window_all_calls"][0]["calls_timestamped"], 1)
        self.assertIn("true_elapsed_seconds", report["timing_question_answers"][0])


class PerItemResume(RunnerTestCase):
    """A cached answer stays valid while its OWN prompt hash matches its item; other items' prompts do not matter."""

    def setUp(self):
        super().setUp()
        self.contexts_path = Path(self.tmp.name) / "contexts.json"
        self.contexts = json.loads(settings.NB04_CONTEXTS_FILE.read_text(encoding="utf-8"))
        self.write_contexts()
        patch = mock.patch.object(settings, "NB04_CONTEXTS_FILE", self.contexts_path)
        patch.start()
        self.addCleanup(patch.stop)

    def write_contexts(self):
        self.contexts_path.write_text(json.dumps(self.contexts), encoding="utf-8")

    def change_prompt(self, item_id):
        item = next(i for i in self.contexts["items"] if i["id"] == item_id)
        item["prompt_sha256"] = "0" * 64
        self.write_contexts()

    def test_changing_one_item_refuses_only_its_answer(self):
        self.generate(OK, OK, OK, only="Q01,Q02,Q03")
        before = self.cache()["results"]
        self.change_prompt("Q02")
        calls = self.calls
        code, out = self.generate(OK, only="Q01,Q02,Q03")
        self.assertEqual(code, 0, out)
        self.assertEqual(self.calls - calls, 1)                 # only Q02 was generated again
        self.assertIn("2 already done, 1 to run (1 made from an earlier prompt: ['Q02'])", out)
        after = self.cache()
        self.assertEqual(after["results"]["Q01"], before["Q01"])
        self.assertEqual(after["results"]["Q03"], before["Q03"])
        self.assertEqual(after["results"]["Q02"]["prompt_sha256"], "0" * 64)
        self.assertEqual(after["superseded"][0]["prompt_sha256"], before["Q02"]["prompt_sha256"])
        self.assertEqual(len(after["results"]["Q02"]["call_log"]), 1)    # nothing carried over from the stale answer

    def test_a_change_elsewhere_in_the_prompt_set_keeps_the_cache(self):
        self.generate(OK, only="Q01")
        self.change_prompt("Q100")                               # the whole prompt-set hash now differs
        calls = self.calls
        code, out = self.generate(only="Q01")
        self.assertEqual((code, self.calls - calls), (0, 0), out)
        self.assertIn("1 already done, 0 to run", out)

    def test_the_prompt_set_is_no_longer_resume_locked(self):
        self.assertNotIn("prompt_set_sha256", run_generation.RESUME_MUST_MATCH)


class ParityCheck8(unittest.TestCase):
    """Check 8 per item: strict for everything except the generation runner, which replaces stale answers."""

    def test_stale_answer_fails_strict_and_passes_tolerant_unknown_fails_both(self):
        items = {i["id"]: i for i in json.loads(settings.NB04_CONTEXTS_FILE.read_text(encoding="utf-8"))["items"]}
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            (raw / "fake.json").write_text(json.dumps({"results": {
                "Q01": {"prompt_sha256": items["Q01"]["prompt_sha256"]},
                "Q02": {"prompt_sha256": "stale"}}}), encoding="utf-8")
            with mock.patch.object(settings, "NB04_RAW_DIR", raw):
                strict = [f for f in run_generation.check_parity.run_checks() if f.startswith("Check 8")]
                tolerant = [f for f in run_generation.check_parity.run_checks(allow_stale_answers=True) if f.startswith("Check 8")]
                (raw / "fake.json").write_text(json.dumps({"results": {"Q999": {"prompt_sha256": "x"}}}), encoding="utf-8")
                unknown = [f for f in run_generation.check_parity.run_checks(allow_stale_answers=True) if f.startswith("Check 8")]
        self.assertEqual(len(strict), 1)
        self.assertIn("Q02", strict[0])
        self.assertNotIn("Q01", strict[0])
        self.assertEqual(tolerant, [])
        self.assertEqual(len(unknown), 1)
        self.assertIn("Q999", unknown[0])


MESSAGES = [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "USER"}]


def fake_openai_response(content, completion_tokens=1000, details=None, prompt_details=None, **usage_extra):
    """An object shaped like the OpenAI SDK's chat completion, for the adapter's parsing (no network)."""
    usage = SimpleNamespace(prompt_tokens=1600, completion_tokens=completion_tokens, completion_tokens_details=details,
                            prompt_tokens_details=prompt_details, **usage_extra)
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
                           usage=usage, model="served-model")


class CandidateLineup(unittest.TestCase):
    """2026-11-27: DeepSeek replaces the LLaMA slot; Groq and its fallback hosts are gone; Mistral is DROPPED (its free mode has no
    API access and paid access needs a $10 minimum top-up), so five candidates remain."""

    def test_five_candidates_one_per_slot_no_groq_no_mistral(self):
        self.assertEqual(list(settings.NB04_CANDIDATES), ["gpt-4o", "gemini-3.8-flash", "claude-sonnet-5-5", "deepseek-flash",
                                                          "gemma-4-31b-it"])
        self.assertEqual({c["slot"] for c in settings.NB04_CANDIDATES.values()}, {"GPT", "Gemini", "Claude", "DeepSeek", "Gemma"})
        self.assertEqual(run_generation.check_parity.all_models(), settings.NB04_CANDIDATES)
        self.assertFalse(hasattr(settings, "NB04_FALLBACK_HOSTS"))
        self.assertEqual(set(settings.API_KEY_ENV_VARS), {"google", "openai", "anthropic", "deepseek"})
        self.assertEqual(settings.API_KEY_ENV_VARS["deepseek"], "DEEPSEEK_API_KEY")
        # 2026-11-28: gemini-3.8-flash moved to Google's paid tier; only Gemma (no paid tier) is still paced and free.
        self.assertEqual(set(settings.NB04_FREE_TIER_MIN_INTERVAL_SECONDS), {"gemma-4-31b-it"})
        self.assertEqual({m for m, c in settings.NB04_CANDIDATES.items() if c["free_tier"]}, {"gemma-4-31b-it"})
        for candidate in settings.NB04_CANDIDATES.values():
            self.assertNotIn("groq", (candidate["base_url"] or "") + candidate["key"])
            self.assertNotIn("mistral", (candidate["base_url"] or "") + candidate["key"])

    def test_judge_calls_follow_the_five_candidates(self):
        self.assertEqual({job: entry["calls"] for job, entry in settings.NB04_JUDGE_JOBS.items()},
                         {"rubric": 500, "ragas": 500, "guardrail": 150, "crosscheck": 100})

    def test_deepseek_request_turns_thinking_off(self):
        candidate = settings.NB04_CANDIDATES["deepseek-flash"]
        request = providers.build_request(candidate, MESSAGES)
        self.assertEqual(request, {"model": "deepseek-flash", "messages": MESSAGES, "max_tokens": 800, "temperature": 0.2,
                                   "extra_body": {"thinking": {"type": "disabled"}}})
        self.assertEqual(candidate["base_url"], "https://api.deepseek.com")
        self.assertEqual(providers.request_messages(candidate, request), MESSAGES)

    def test_an_allowance_is_added_on_the_openai_style_path_too(self):
        """A model that cannot turn reasoning off gets its allowance on top of the 800, on any API style (none does now)."""
        candidate = {**settings.NB04_CANDIDATES["deepseek-flash"], "reasoning_allowance_tokens": 1200}
        self.assertEqual(providers.build_request(candidate, MESSAGES)["max_tokens"], 2000)

    def test_gpt_4o_request_is_unchanged(self):
        request = providers.build_request(settings.NB04_CANDIDATES["gpt-4o"], MESSAGES)
        self.assertEqual(request, {"model": "gpt-4o", "messages": MESSAGES, "max_completion_tokens": 800, "temperature": 0.2})

    def test_estimates_follow_the_new_prices(self):
        est = {name: round(run_generation.estimated_spend_usd(c), 4) for name, c in settings.NB04_CANDIDATES.items()}
        self.assertAlmostEqual(est["deepseek-flash"], 130 * (1605 * 0.30 + 400 * 1.20) / 1e6, places=4)
        self.assertEqual(est["gemma-4-31b-it"], 0.0)


class OpenAiStyleParsing(unittest.TestCase):
    """The adapter reads a plain string answer, and Mistral's [thinking chunk, text chunk] answer, without showing the thinking."""

    def send(self, response, model="deepseek-flash"):
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response)))
        with mock.patch.object(providers, "_client", lambda candidate: client):
            return providers.send(settings.NB04_CANDIDATES[model], {"model": "x"})

    def test_plain_string(self):
        result = self.send(fake_openai_response("An answer [S06-0204].", completion_tokens=100), model="deepseek-flash")
        self.assertEqual((result.text, result.reasoning_tokens, result.reasoning_tokens_estimated), ("An answer [S06-0204].", 0, False))

    def test_mistral_chunks_keep_only_the_text_and_estimate_the_reasoning(self):
        thinking = "t" * 300
        content = [{"type": "thinking", "thinking": [{"type": "text", "text": thinking}]}, {"type": "text", "text": "a" * 100}]
        result = self.send(fake_openai_response(content, completion_tokens=1000))
        self.assertEqual(result.text, "a" * 100)
        self.assertEqual((result.reasoning_tokens, result.reasoning_tokens_estimated), (750, True))
        self.assertEqual(result.output_tokens, 1000, "cost still uses every output token")

    def test_a_reported_reasoning_figure_wins(self):
        content = [{"type": "thinking", "thinking": [{"type": "text", "text": "t" * 300}]}, {"type": "text", "text": "answer"}]
        result = self.send(fake_openai_response(content, details=SimpleNamespace(reasoning_tokens=420)))
        self.assertEqual((result.reasoning_tokens, result.reasoning_tokens_estimated), (420, False))

    def test_sdk_objects_as_chunks(self):
        content = [SimpleNamespace(type="thinking", thinking=[SimpleNamespace(type="text", text="hmm")]),
                   SimpleNamespace(type="text", text="Answer.")]
        self.assertEqual(self.send(fake_openai_response(content)).text, "Answer.")

    def test_masked_key_fragments_in_host_errors_are_redacted(self):
        for text in ("Authentication Fails, Your api key: ****1ed5 is invalid", "Incorrect API key provided: sk-ab***wxyz."):
            saved = providers.describe_error(Exception(text))["error_text"]
            self.assertNotIn("1ed5", saved)
            self.assertNotIn("wxyz", saved)
            self.assertIn("[redacted]", saved)

    def test_cached_tokens_are_read_from_either_field(self):
        openai_style = self.send(fake_openai_response("x", prompt_details=SimpleNamespace(cached_tokens=1024)), model="gpt-4o")
        deepseek_style = self.send(fake_openai_response("x", prompt_cache_hit_tokens=896), model="deepseek-flash")
        self.assertEqual((openai_style.cached_input_tokens, deepseek_style.cached_input_tokens), (1024, 896))


class ParallelRunsOnWindows(RunnerTestCase):
    """Several models run at once; a cache being read by another process cannot be replaced for a moment on Windows."""

    def test_a_busy_rename_is_retried_and_the_cache_saved(self):
        real_replace, calls = Path.replace, []

        def flaky_replace(self_path, target):
            calls.append(target)
            if len(calls) <= 2:
                raise PermissionError("[WinError 5] Access is denied")
            return real_replace(self_path, target)
        with mock.patch.object(Path, "replace", flaky_replace):
            run_generation.save_cache(self.raw / "x.json", {"results": {}, "spend": {"list_usd_total": 1.5}})
        self.assertEqual(len(calls), 3)
        self.assertEqual(json.loads((self.raw / "x.json").read_text(encoding="utf-8"))["spend"]["list_usd_total"], 1.5)
        self.assertEqual(len(self.sleeps), 2, "two short waits, recorded by the patched sleep")

    def test_a_rename_that_never_clears_still_raises(self):
        with mock.patch.object(Path, "replace", side_effect=PermissionError("busy")):
            with self.assertRaises(PermissionError):
                run_generation.save_cache(self.raw / "x.json", {"results": {}})


class FreeTierPacing(RunnerTestCase):
    def test_pace_waits_only_the_rest_of_the_interval(self):
        with mock.patch.object(settings, "NB04_FREE_TIER_MIN_INTERVAL_SECONDS", {MODEL: 4.0}), \
                mock.patch.object(run_generation.time, "monotonic", lambda: 101.0):
            self.assertEqual(run_generation.pace(MODEL, None), 0.0)
            self.assertEqual(run_generation.pace(MODEL, 100.0), 3.0)
            self.assertEqual(run_generation.pace(MODEL, 90.0), 0.0)
            self.assertEqual(run_generation.pace("gpt-4o", 100.0), 0.0)
        self.assertEqual(self.sleeps, [3.0])

    def test_a_run_records_the_pacing_and_keeps_it_out_of_latency(self):
        with mock.patch.object(settings, "NB04_FREE_TIER_MIN_INTERVAL_SECONDS", {MODEL: 4.0}):
            self.outcomes = [OK, OK]
            self.run_main("--model", MODEL, "--only", "Q01,Q02")
        results = self.cache()["results"]
        self.assertEqual(results["Q01"]["pacing_wait_seconds"], 0.0)
        self.assertGreater(results["Q02"]["pacing_wait_seconds"], 3.0)
        self.assertEqual(len(self.sleeps), 1)
        self.assertLess(results["Q02"]["latency_seconds"], 1.0)
        self.assertEqual(self.cache()["run_settings"]["min_interval_seconds"], 4.0)


class AccessTier(RunnerTestCase):
    """Author, 2026-11-28: a model moving from Google's free tier to the paid tier keeps both kinds of answer, labelled per answer."""

    def paid(self):
        return mock.patch.dict(settings.NB04_CANDIDATES, {MODEL: {**CANDIDATE, "free_tier": False, "access_note": "paid tier"}})

    def test_each_answer_records_the_tier_it_was_made_on(self):
        self.generate(OK, only="Q01")
        with self.paid():
            self.generate(OK, only="Q02")
        results, run = self.cache()["results"], self.cache()["run_settings"]
        self.assertEqual((results["Q01"]["access_tier"], results["Q02"]["access_tier"]), ("free", "paid"))
        self.assertTrue(run["free_tier"], "run_settings keeps the tier the run started on")
        self.assertEqual([c["access_tier"] for c in run["access_changes"]], ["paid"])
        self.assertEqual(run["access_changes"][0]["access_note"], "paid tier")

    def test_the_change_is_logged_once(self):
        self.generate(OK, only="Q01")
        with self.paid():
            self.generate(OK, only="Q02")
            self.generate(OK, only="Q03")
        self.assertEqual(len(self.cache()["run_settings"]["access_changes"]), 1)

    FREE_TIER_429 = ("429 RESOURCE_EXHAUSTED. Quota exceeded for metric: generativelanguage.googleapis.com/"
                     "generate_content_free_tier_requests, limit: 20 (quotaId GenerateRequestsPerDayPerProjectPerModel-FreeTier)")

    def test_a_free_tier_quota_error_stops_a_model_set_to_paid(self):
        with self.paid():
            code, out = self.generate(providers.RateLimited(self.FREE_TIER_429, 13099.0), OK)
        self.assertEqual((code, self.calls, self.sleeps), (run_generation.EXIT_BLOCKED, 1, []), out)
        cache = self.cache()
        self.assertEqual(cache["results"], {}, "no answer is labelled paid")
        self.assertIn("free_tier_requests", cache["access_blocked"][0]["call_log"][0]["error_text"])   # Google's own text
        self.assertIn("set to the paid tier", out)

    def test_on_the_free_tier_the_same_error_is_waited_out(self):
        code, _ = self.generate(providers.RateLimited(self.FREE_TIER_429, 13099.0), OK)
        self.assertEqual((code, self.sleeps), (0, [settings.NB04_RATE_LIMIT_HOST_WAIT_MAX_SECONDS]))
        self.assertEqual(self.cache()["results"]["Q01"]["access_tier"], "free")

    def test_answers_saved_before_the_field_get_their_runs_tier(self):
        self.generate(OK, only="Q01")
        path = self.raw / f"{MODEL}.json"
        cache = json.loads(path.read_text(encoding="utf-8"))
        del cache["results"]["Q01"]["access_tier"]
        run_generation.save_cache(path, cache)
        with self.paid():
            self.generate(OK, only="Q02")
        self.assertEqual(self.cache()["results"]["Q01"]["access_tier"], "free")
        self.assertEqual(self.cache()["results"]["Q02"]["access_tier"], "paid")


class RemeasureLatency(RunnerTestCase):
    """Pre-registration (a): the first 20 frozen questions again, into raw/remeasure/; the answer cache is never touched."""

    def remeasure(self, *outcomes):
        self.outcomes = list(outcomes)
        return self.run_main("--model", MODEL, "--remeasure-latency")

    def test_first_twenty_questions_go_to_their_own_file(self):
        self.generate(OK, only="Q01")
        before = (self.raw / f"{MODEL}.json").read_bytes()
        code, out = self.remeasure(*[OK] * settings.NB04_REMEASURE_QUESTIONS)
        self.assertEqual(code, 0, out)
        self.assertEqual((self.raw / f"{MODEL}.json").read_bytes(), before, "the answer cache is untouched")
        saved = json.loads((run_generation.remeasure_dir() / f"{MODEL}.json").read_text(encoding="utf-8"))
        questions = [i["id"] for i in json.loads(settings.NB04_CONTEXTS_FILE.read_text(encoding="utf-8"))["items"]
                     if i["kind"] == "question"][:settings.NB04_REMEASURE_QUESTIONS]
        self.assertEqual(list(saved["results"]), questions)
        self.assertEqual(questions[0], "Q01")
        self.assertIn("pre-registration (a)", saved["run_settings"]["purpose"])
        self.assertTrue(all(r["access_tier"] == "free" for r in saved["results"].values()))   # this test model is Gemma

    def test_its_spend_counts_towards_the_total_and_it_is_never_read_as_answers(self):
        self.remeasure(*[OK] * settings.NB04_REMEASURE_QUESTIONS)
        saved = json.loads((run_generation.remeasure_dir() / f"{MODEL}.json").read_text(encoding="utf-8"))
        with mock.patch.dict(settings.NB04_CANDIDATES, {MODEL: {**CANDIDATE, "input_usd_per_m": 1.0, "output_usd_per_m": 1.0}}):
            self.remove_remeasure()
            self.remeasure(*[OK] * settings.NB04_REMEASURE_QUESTIONS)
        priced = json.loads((run_generation.remeasure_dir() / f"{MODEL}.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["spend"]["list_usd_total"], 0.0)
        self.assertAlmostEqual(priced["spend"]["list_usd_total"], 20 * 1100 / 1e6)
        self.assertAlmostEqual(run_generation.total_spend_usd(), 20 * 1100 / 1e6)
        self.assertFalse((self.raw / f"{MODEL}.json").exists())

    def remove_remeasure(self):
        (run_generation.remeasure_dir() / f"{MODEL}.json").unlink()

    def test_only_and_items_are_refused(self):
        with contextlib.redirect_stderr(io.StringIO()):
            code, _ = self.run_main("--model", MODEL, "--remeasure-latency", "--only", "Q01")
        self.assertEqual(code, 2)
        self.assertEqual(self.calls, 0)


if __name__ == "__main__":
    unittest.main(verbosity=1)
