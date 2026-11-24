"""
Offline tests of the Ollama thinking switch (settings.DEV_LLM_THINK_OFF_MODELS, llm._request_body).

qwen3:1.7b and qwen3.5:2b think by default in Ollama, and thinking tokens count against the answer cap, so every request
for them must send "think": false. Every other model (llama3.2:3b) must get NO "think" field, so its request body stays
exactly as it was before the switch was added. requests.post is faked: nothing leaves the machine.

Run from the project root:  venv\\Scripts\\python tests\\test_think_switch.py
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from config import settings  # noqa: E402
from src.core import intent, llm  # noqa: E402

CHUNK = SimpleNamespace(chunk_id="S05-0138", source_name="DE Wiki", section="OLAP", text="OLAP is a data model.", score=0.9,
                        page=None, file="x.md", topic="Data Warehousing", source_id="S05", source_type="course_notes")
MESSAGES = [{"role": "user", "content": "what is OLAP?"}]
QWEN = ["qwen3:1.7b", "qwen3.5:2b"]


class FakeResponse:
    """Enough of a requests.Response for llm.generate / stream_generate / complete."""
    def __init__(self, stream):
        self.stream = stream

    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": "REAL_QUESTION"}, "done": True, "done_reason": "stop"}

    def iter_lines(self):
        yield b'{"message": {"content": "OLAP is a data model [S05-0138]."}, "done": false}'
        yield b'{"message": {"content": ""}, "done": true, "done_reason": "stop"}'

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def sent_bodies(call):
    """Run call() with requests.post faked and return every JSON body it sent."""
    bodies = []

    def fake_post(url, json=None, stream=False, **kwargs):
        bodies.append(json)
        return FakeResponse(stream)

    with mock.patch.object(llm.requests, "post", side_effect=fake_post):
        call()
    return bodies


class RequestBody(unittest.TestCase):
    def test_llama_has_no_think_field(self):
        body = llm._request_body(MESSAGES, stream=False, model="llama3.2:3b")
        self.assertNotIn("think", body)

    def test_llama_body_unchanged(self):
        """The exact keys llama3.2:3b got before the switch existed."""
        body = llm._request_body(MESSAGES, stream=True, model="llama3.2:3b")
        self.assertEqual(set(body), {"model", "messages", "stream", "keep_alive", "options"})
        self.assertEqual(body["options"], {"temperature": settings.LLM_TEMPERATURE, "num_ctx": settings.DEV_LLM_NUM_CTX})

    def test_qwen_models_send_think_false(self):
        for model in QWEN:
            with self.subTest(model=model):
                body = llm._request_body(MESSAGES, stream=False, model=model)
                self.assertIs(body["think"], False)      # the boolean False, not a missing key or "false"

    def test_qwen_models_are_listed(self):
        self.assertTrue(set(QWEN) <= settings.DEV_LLM_THINK_OFF_MODELS)
        self.assertNotIn("llama3.2:3b", settings.DEV_LLM_THINK_OFF_MODELS)

    def test_default_model_follows_dev_llm_model(self):
        body = llm._request_body(MESSAGES, stream=False)
        self.assertEqual(body["model"], settings.DEV_LLM_MODEL)
        self.assertEqual("think" in body, settings.DEV_LLM_MODEL in settings.DEV_LLM_THINK_OFF_MODELS)


class EveryCallPath(unittest.TestCase):
    """The switch must reach the wire on every path that calls Ollama: answer (streamed and not) and the classifier."""

    def check(self, call, model):
        bodies = sent_bodies(call)
        self.assertEqual(len(bodies), 1)
        if model in settings.DEV_LLM_THINK_OFF_MODELS:
            self.assertIs(bodies[0]["think"], False)
        else:
            self.assertNotIn("think", bodies[0])

    def test_stream_generate(self):
        for model in ["llama3.2:3b"] + QWEN:
            with self.subTest(model=model):
                self.check(lambda: list(llm.stream_generate("what is OLAP?", [CHUNK], model=model)), model)

    def test_generate(self):
        for model in ["llama3.2:3b"] + QWEN:
            with self.subTest(model=model):
                self.check(lambda: llm.generate("what is OLAP?", [CHUNK], model=model), model)

    def test_intent_classifier_uses_dev_llm_model(self):
        """The classifier always uses DEV_LLM_MODEL, so it gets think false exactly when that model is listed."""
        for model in ["llama3.2:3b"] + QWEN:
            with self.subTest(model=model), mock.patch.object(settings, "DEV_LLM_MODEL", model):
                bodies = sent_bodies(lambda: intent.classify("im sick"))
                self.assertEqual(bodies[0]["model"], model)
                self.assertEqual(bodies[0]["options"]["num_predict"], settings.INTENT_CLASSIFIER_MAX_TOKENS)
                if model in QWEN:
                    self.assertIs(bodies[0]["think"], False)
                else:
                    self.assertNotIn("think", bodies[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
