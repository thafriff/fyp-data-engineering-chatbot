"""
Offline tests of the chat prototype's cloud path: deepseek-flash (the app's default since 2026-11-28, DeepSeek's OpenAI-compatible
API) and gemini-3.8-flash and gemma-4-31b-it (Google Gemini API).

The API calls (llm._send_google, llm._send_openai) are always faked: no network, no real key, and no notebook 04 file is read for
writing.
The interface tests drive the real Streamlit page with streamlit.testing (AppTest) and the real retrieval, which loads the
embedding model from the local cache.

Run from the project root:  venv\\Scripts\\python tests\\test_cloud_app.py
"""

import hashlib
import os
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "evaluation" / "notebook04")]
sys.stdout.reconfigure(encoding="utf-8")
os.environ["HF_HUB_OFFLINE"] = "1"

from streamlit.testing.v1 import AppTest  # noqa: E402

import providers  # noqa: E402
from config import settings  # noqa: E402
from src.core import llm  # noqa: E402

APP = str(ROOT / "src" / "app" / "app.py")
CHUNK = SimpleNamespace(chunk_id="S05-0138", source_name="DE Wiki", section="OLAP", text="OLAP is a data model.", score=0.9,
                        page=None, file="x.md", topic="Data Warehousing", source_id="S05", source_type="course_notes")
GOOD = llm._Reply("OLAP is a data model [S05-0138].", 1100, 300, 0, "STOP")
FAST = {"NB04_BACKOFF_BASE_SECONDS": 0, "NB04_BACKOFF_MAX_SECONDS": 0, "DEV_CLOUD_STATUS_TICK_SECONDS": 0.01}


class ApiError(Exception):
    """Stands in for the SDK's error: the status is in .code, as in google.genai.errors.APIError."""
    def __init__(self, code, message="error"):
        super().__init__(message)
        self.code = code


class ConnectError(Exception):
    """A dropped connection: no status, and a name containing "connect"."""


def patched(**extra):
    return mock.patch.multiple(settings, **{**FAST, **extra})


class CloudGeneration(unittest.TestCase):
    def setUp(self):
        for p in (mock.patch.object(llm, "_google_api_key", lambda: "fake-key-not-real"), patched()):
            p.start()
            self.addCleanup(p.stop)

    def run_with(self, *outcomes, model="gemini-3.8-flash"):
        """cloud_generate with the Google call answering from outcomes in order. Returns (answer or error, statuses, calls)."""
        queue = list(outcomes)

        def send(key, request):
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        statuses = []
        with mock.patch.object(llm, "_send_google", send):
            try:
                result = llm.cloud_generate("what is OLAP?", [CHUNK], model, statuses.append)
            except llm.CloudError as error:
                result = error
        return result, statuses, len(outcomes) - len(queue)

    def test_request_is_identical_to_the_trial_runner(self):
        messages = llm.build_messages("what is OLAP?", [CHUNK])
        for name in settings.DEV_CLOUD_MODELS:
            self.assertEqual(llm._cloud_request(name, messages),
                             providers.build_request(settings.NB04_CANDIDATES[name], messages), name)
            if settings.DEV_CLOUD_MODELS[name]["api"] == "gemini":
                self.assertEqual(llm._cloud_request(name, messages)["config"]["thinking_config"],
                                 {"thinking_level": settings.NB04_CANDIDATES[name]["request_options"]["thinking_level"]})
        self.assertEqual(llm._cloud_request("deepseek-flash", messages)["extra_body"], {"thinking": {"type": "disabled"}})

    def test_request_with_a_spelling_hint_is_identical_too(self):
        messages = llm.build_messages("what is olpa", [CHUNK], corrections=[("olpa", "olap")])
        self.assertIn("Assuming you mean olap", messages[1]["content"])
        for name in settings.DEV_CLOUD_MODELS:
            self.assertEqual(llm._cloud_request(name, messages),
                             providers.build_request(settings.NB04_CANDIDATES[name], messages), name)

    def test_success_and_cost(self):
        answer, _, calls = self.run_with(GOOD)
        self.assertEqual((calls, answer.input_tokens, answer.output_tokens, answer.retries), (1, 1100, 300, 0))
        config = settings.DEV_CLOUD_MODELS["gemini-3.8-flash"]
        self.assertAlmostEqual(answer.cost_usd_list, (1100 * config["input_usd_per_m"] + 300 * config["output_usd_per_m"]) / 1e6)
        self.assertEqual(self.run_with(GOOD, model="gemma-4-31b-it")[0].cost_usd_list, 0.0)

    def test_503_retried_then_succeeds(self):
        answer, _, calls = self.run_with(ApiError(503), ApiError(503), GOOD)
        self.assertEqual((calls, answer.retries), (3, 2))

    def test_connection_failure_retried(self):
        answer, _, calls = self.run_with(ConnectError("dropped"), GOOD)
        self.assertEqual((calls, answer.retries), (2, 1))

    def test_six_retries_then_overloaded_message(self):
        error, _, calls = self.run_with(*[ApiError(503, "secret detail") for _ in range(settings.DEV_CLOUD_BACKOFF_MAX_RETRIES + 1)])
        self.assertIsInstance(error, llm.CloudError)
        self.assertEqual(calls, settings.DEV_CLOUD_BACKOFF_MAX_RETRIES + 1)
        self.assertEqual(str(error), settings.DEV_CLOUD_MSG_OVERLOADED.format(
            model="gemini-3.8-flash", attempts=settings.DEV_CLOUD_BACKOFF_MAX_RETRIES + 1, host="Google"))
        self.assertNotIn("secret detail", str(error))

    def test_429_message(self):
        error, _, _ = self.run_with(*[ApiError(429) for _ in range(settings.DEV_CLOUD_BACKOFF_MAX_RETRIES + 1)])
        self.assertIn("hit the Google rate limit", str(error))

    def test_invalid_key_not_retried(self):
        error, _, calls = self.run_with(ApiError(400, "API_KEY_INVALID"), GOOD)
        self.assertEqual(calls, 1)
        self.assertIn("rejected the API key", str(error))
        self.assertNotIn("fake-key-not-real", str(error))

    def test_401_and_403_are_key_problems(self):
        for code in (401, 403):
            error, _, calls = self.run_with(ApiError(code), GOOD)
            self.assertEqual(calls, 1, code)
            self.assertIn(f"HTTP {code}", str(error))

    def test_other_400_is_not_called_a_key_problem(self):
        error, _, calls = self.run_with(ApiError(400, "Invalid argument"), GOOD)
        self.assertEqual(calls, 1)
        self.assertNotIn("API key", str(error))
        self.assertIn("HTTP 400", str(error))

    def test_missing_key(self):
        with mock.patch.object(llm, "_google_api_key", lambda: None), \
                mock.patch.object(llm, "_send_google", side_effect=AssertionError("must not be called")):
            with self.assertRaises(llm.CloudError) as raised:
                llm.cloud_generate("what is OLAP?", [CHUNK], "gemma-4-31b-it")
        self.assertIn("GOOGLE_API_KEY is not set", str(raised.exception))

    def test_check_backend_reports_a_missing_key(self):
        with mock.patch.object(llm, "_google_api_key", lambda: None):
            self.assertFalse(llm.check_backend("gemma-4-31b-it")[0])

    def test_timeout_message(self):
        def slow(key, request):
            time.sleep(0.5)
            return GOOD
        with mock.patch.object(llm, "_send_google", slow), patched(NB04_REQUEST_TIMEOUT_SECONDS=0.05):
            with self.assertRaises(llm.CloudError) as raised:
                llm.cloud_generate("what is OLAP?", [CHUNK], "gemini-3.8-flash")
        self.assertIn("did not answer within 0.05 seconds", str(raised.exception))

    def test_waiting_status_shows_model_and_elapsed(self):
        def slow(key, request):
            time.sleep(0.1)
            return GOOD
        statuses = []
        with mock.patch.object(llm, "_send_google", slow):
            llm.cloud_generate("what is OLAP?", [CHUNK], "gemini-3.8-flash", statuses.append)
        self.assertTrue(statuses)
        self.assertTrue(all(s.startswith("Waiting for gemini-3.8-flash...") for s in statuses), statuses[:3])

    def test_retry_status_visible(self):
        with patched(NB04_BACKOFF_BASE_SECONDS=0.03, NB04_BACKOFF_MAX_SECONDS=0.03):
            _, statuses, _ = self.run_with(ApiError(503), GOOD)
        self.assertTrue(any("is busy (high demand)" in s and "attempt 2 of" in s for s in statuses), statuses)

    def test_repetition_guard(self):
        line = "OLAP stands for online analytical processing [S05-0138].\n"
        answer, _, _ = self.run_with(llm._Reply(line * 5, 1100, 300, 0, "STOP"))
        self.assertTrue(answer.text.endswith(settings.ANSWER_REPEAT_NOTE))
        self.assertEqual(answer.text.count("online analytical processing"), settings.DEV_LLM_REPEAT_LINE_LIMIT)

    def test_length_cap(self):
        answer, _, _ = self.run_with(llm._Reply("word " * 2000, 1100, 1000, 0, "STOP"))
        self.assertTrue(answer.text.endswith(settings.ANSWER_CUT_NOTE))
        self.assertLess(len(answer.text), len("word " * 2000))

    def test_reasoning_does_not_count_against_the_cap(self):
        answer, _, _ = self.run_with(llm._Reply("A short answer [S05-0138].", 1100, 1500, 900, "STOP"))
        self.assertEqual(answer.text, "A short answer [S05-0138].")

    def test_cut_note_on_max_tokens(self):
        answer, _, _ = self.run_with(llm._Reply("Cut here", 1100, 300, 0, "MAX_TOKENS"))
        self.assertEqual(answer.text, "Cut here" + settings.ANSWER_CUT_NOTE)

    def test_empty_reply_message(self):
        error, _, _ = self.run_with(llm._Reply("  ", 1100, 0, 0, "SAFETY"))
        self.assertIn("finish reason: SAFETY", str(error))

    def test_ollama_path_is_not_taken_for_cloud_models(self):
        with mock.patch.object(llm.requests, "get", side_effect=AssertionError("no Ollama for a cloud model")):
            self.assertTrue(llm.check_backend("gemini-3.8-flash")[0])


class Interface(unittest.TestCase):
    """The real page with the API calls faked (DeepSeek's fails the test unless a test replaces it)."""

    def setUp(self):
        for p in (patched(), mock.patch.object(llm, "_google_api_key", lambda: "fake-key-not-real"),
                  mock.patch.object(llm, "_api_key", lambda provider: "fake-key-not-real"),
                  mock.patch.object(llm, "_send_openai", side_effect=AssertionError("DeepSeek must not be called here"))):
            p.start()
            self.addCleanup(p.stop)

    def app(self):
        at = AppTest.from_file(APP, default_timeout=180).run()
        self.assertFalse(at.exception, at.exception)
        return at

    def ask(self, at, model, question):
        at.selectbox[0].set_value(model).run()
        at.chat_input[0].set_value(question).run()
        self.assertFalse(at.exception, at.exception)
        return at.chat_message[-1]

    def test_default_and_grouping(self):
        at = self.app()
        self.assertEqual(at.selectbox[0].value, settings.DEV_DEFAULT_ANSWER_MODEL)
        self.assertEqual(settings.DEV_DEFAULT_ANSWER_MODEL, "deepseek-flash")
        self.assertEqual(at.selectbox[0].options,
                         [f"{settings.DEV_MODEL_GROUP_LOCAL} · {m}" for m in settings.DEV_LLM_MODEL_OPTIONS]
                         + [f"{settings.DEV_MODEL_GROUP_CLOUD} · {m}" for m in settings.DEV_CLOUD_MODELS])

    def test_cloud_answer_captions_and_numbered_citation(self):
        with mock.patch.object(llm, "_send_google", lambda key, request: GOOD):
            message = self.ask(self.app(), "gemini-3.8-flash", "what is OLAP?")
        captions = [c.value for c in message.caption]
        self.assertIn("Generated by gemini-3.8-flash", captions)
        self.assertTrue(any(c.startswith("Input 1,100 tokens · output 300 tokens") for c in captions), captions)
        self.assertEqual(message.markdown[0].value, "OLAP is a data model [1].")

    def test_captions_survive_a_rerun(self):
        with mock.patch.object(llm, "_send_google", lambda key, request: GOOD):
            at = self.app()
            self.ask(at, "gemma-4-31b-it", "what is OLAP?")
            before = [c.value for c in at.chat_message[-1].caption]
            at.run()
        self.assertEqual([c.value for c in at.chat_message[-1].caption], before)
        self.assertIn("Generated by gemma-4-31b-it", before)

    def test_overload_shows_a_message_not_a_crash(self):
        with mock.patch.object(llm, "_send_google", side_effect=ApiError(503)):
            message = self.ask(self.app(), "gemini-3.8-flash", "what is OLAP?")
        self.assertIn("overloaded", message.error[0].value)
        self.assertFalse([c.value for c in message.caption if c.value.startswith("Generated by")])

    def test_invalid_key_message(self):
        with mock.patch.object(llm, "_send_google", side_effect=ApiError(400, "API_KEY_INVALID")):
            message = self.ask(self.app(), "gemini-3.8-flash", "what is OLAP?")
        self.assertIn("rejected the API key", message.error[0].value)

    def test_missing_key_message_in_chat(self):
        with mock.patch.object(llm, "_google_api_key", lambda: None):
            message = self.ask(self.app(), "gemma-4-31b-it", "what is OLAP?")
        self.assertIn("GOOGLE_API_KEY is not set", message.error[0].value)

    def test_below_threshold_calls_no_model(self):
        with mock.patch.object(llm, "_send_google", side_effect=AssertionError("must not be called")):
            message = self.ask(self.app(), "gemini-3.8-flash", "What is the capital of France?")
        self.assertEqual(message.markdown[0].value, settings.FALLBACK_MESSAGE)


class DeepSeekAndHistory(unittest.TestCase):
    """deepseek-flash in the app (author, 2026-11-28) and conversation history on both cloud APIs."""

    def setUp(self):
        for p in (patched(), mock.patch.object(llm, "_api_key", lambda provider: f"fake-{provider}-key"),
                  mock.patch.object(llm, "_google_api_key", lambda: "fake-google-key")):
            p.start()
            self.addCleanup(p.stop)

    def test_deepseek_answer_and_cost(self):
        sent = []

        def fake(model, key, request):
            sent.append((model, key, request))
            return GOOD
        with mock.patch.object(llm, "_send_openai", fake), \
                mock.patch.object(llm, "_send_google", side_effect=AssertionError("DeepSeek never calls Google")):
            answer = llm.cloud_generate("what is OLAP?", [CHUNK], "deepseek-flash")
        model, key, request = sent[0]
        self.assertEqual((model, key, request["model"]), ("deepseek-flash", "fake-deepseek-key", "deepseek-flash"))
        config = settings.DEV_CLOUD_MODELS["deepseek-flash"]
        self.assertAlmostEqual(answer.cost_usd_list, (1100 * config["input_usd_per_m"] + 300 * config["output_usd_per_m"]) / 1e6)

    def test_deepseek_missing_key_names_its_own_key(self):
        with mock.patch.object(llm, "_api_key", lambda provider: None), \
                mock.patch.object(llm, "_send_openai", side_effect=AssertionError("must not be called")):
            with self.assertRaises(llm.CloudError) as raised:
                llm.cloud_generate("what is OLAP?", [CHUNK], "deepseek-flash")
            ok, message = llm.check_backend("deepseek-flash")
        self.assertIn("DEEPSEEK_API_KEY is not set", str(raised.exception))
        self.assertFalse(ok)
        self.assertIn("DEEPSEEK_API_KEY", message)

    def test_the_backend_line_says_what_is_sent(self):
        ok, message = llm.check_backend("deepseek-flash")
        self.assertTrue(ok)
        self.assertIn("sent to DeepSeek", message)

    def test_openai_reply_is_read_like_the_runner(self):
        usage = SimpleNamespace(prompt_tokens=1200, completion_tokens=250, completion_tokens_details=None)
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="An answer [S05-0138]."),
                                                            finish_reason="length")], usage=usage)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response)))
        cache_key = ("k", settings.DEV_CLOUD_MODELS["deepseek-flash"]["base_url"])
        with mock.patch.dict(llm._openai_clients, {cache_key: client}):
            reply = llm._send_openai("deepseek-flash", "k", {"model": "deepseek-flash"})
        self.assertEqual((reply.text, reply.input_tokens, reply.output_tokens, reply.reasoning_tokens, reply.finish),
                         ("An answer [S05-0138].", 1200, 250, 0, "MAX_TOKENS"))

    def history_messages(self):
        messages = llm.build_messages("and ELT?", [CHUNK])
        return [messages[0], {"role": "user", "content": "what is ETL?"},
                {"role": "assistant", "content": "ETL is extract, transform, load."}, messages[1]]

    def test_gemini_with_history_answers_the_current_question(self):
        """Before 2026-11-28 the Gemini request sent only messages[1]: on a follow-up, the FIRST earlier question."""
        messages = self.history_messages()
        contents = llm._cloud_request("gemini-3.8-flash", messages)["contents"]
        self.assertEqual([c["role"] for c in contents], ["user", "model", "user"])
        self.assertEqual(contents[-1]["parts"][0]["text"], messages[-1]["content"])
        self.assertEqual(contents[0]["parts"][0]["text"], "what is ETL?")

    def test_deepseek_with_history_sends_every_turn_in_order(self):
        messages = self.history_messages()
        sent = llm._cloud_request("deepseek-flash", messages)["messages"]
        self.assertEqual([m["role"] for m in sent], ["system", "user", "assistant", "user"])
        self.assertEqual(sent[-1]["content"], messages[-1]["content"])

    def test_single_turn_gemini_request_is_unchanged(self):
        messages = llm.build_messages("what is OLAP?", [CHUNK])
        self.assertEqual(llm._cloud_request("gemma-4-31b-it", messages)["contents"], messages[1]["content"])

    def test_interface_default_answers_with_deepseek(self):
        with mock.patch.object(llm, "_send_openai", lambda model, key, request: GOOD), \
                mock.patch.object(llm, "_send_google", side_effect=AssertionError("the default is not a Google model")):
            at = AppTest.from_file(APP, default_timeout=180).run()
            at.chat_input[0].set_value("what is OLAP?").run()
        self.assertFalse(at.exception, at.exception)
        captions = [c.value for c in at.chat_message[-1].caption]
        self.assertIn("Generated by deepseek-flash", captions)


class Frozen(unittest.TestCase):
    def test_frozen_prompt_unchanged(self):
        self.assertTrue(hashlib.sha256(settings.SYSTEM_PROMPT.encode("utf-8")).hexdigest().startswith("677e8b3b"))   # re-registered 2026-11-24 (was fe8912d1)

    def test_stream_generate_still_uses_ollama(self):
        urls = []

        def fake_post(url, **kwargs):
            urls.append(url)
            raise RuntimeError("stop here")
        with mock.patch.object(llm.requests, "post", fake_post), \
                mock.patch.object(llm, "_send_google", side_effect=AssertionError("local models never call Google")):
            with self.assertRaises(RuntimeError):
                list(llm.stream_generate("what is OLAP?", [CHUNK], model=settings.DEV_LLM_MODEL))
        self.assertEqual(urls, [f"{settings.OLLAMA_BASE_URL}/api/chat"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
