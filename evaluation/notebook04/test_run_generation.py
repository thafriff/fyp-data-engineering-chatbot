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


if __name__ == "__main__":
    unittest.main(verbosity=1)
