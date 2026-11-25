"""
Offline tests of the intent-classifier fix for off-topic requests (author, 2026-11-24): "best nasi lemak in Shah Alam" (no
question word) was labelled EMOTIONAL_OTHER and got the empathy reply. Off-topic requests must go to retrieval and get the course
fallback; emotional messages and small talk must be unchanged.

These tests fake the classifier's label and the retriever, so they check the ROUTING given a label, the prompt and examples the
classifier receives, and that the fixed case set was not tuned on. Whether the real model gives the right label is measured live
by evaluation/intent_classifier_check.py on the hashed set evaluation/intent_classifier_cases.json. No model, no network.

Run from the project root:  venv\\Scripts\\python tests\\test_intent_offtopic.py
"""

import hashlib
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from config import settings  # noqa: E402
from src.core import intent, llm, rag  # noqa: E402

CASES = ROOT / "evaluation" / "intent_classifier_cases.json"


def no_model(*args, **kwargs):
    raise AssertionError("a model was called")


def pipeline(top_score: float):
    """A RagPipeline with a stand-in retriever whose best chunk scores top_score; no spelling corrector, no index."""
    p = rag.RagPipeline.__new__(rag.RagPipeline)
    p.speller = None
    p.retriever = SimpleNamespace(retrieve=lambda text, top_k=None: [
        SimpleNamespace(chunk_id="S05-0138", score=top_score, text="OLAP is a data model.", source_name="DE Wiki",
                        section="OLAP")])
    return p


def answer(message: str, label: str | None, top_score: float = 0.50):
    """The student's reply for a message, with the classifier faked to `label` (None: it must not be called) and no LLM."""
    classify = no_model if label is None else (lambda m: label)
    with mock.patch.object(intent, "classify", classify), mock.patch.object(llm, "stream_generate", no_model), \
            mock.patch.object(llm.requests, "post", no_model):
        result = pipeline(top_score).answer(message)
        return result, "".join(result.answer_stream)


class OffTopicGoesToTheFallback(unittest.TestCase):
    def test_the_authors_three(self):
        for message in ("best nasi lemak in Shah Alam", "recommend me a movie", "who won the world cup"):
            result, text = answer(message, label=None if message.startswith("who") else "REAL_QUESTION")
            self.assertTrue(result.used_fallback, message)
            self.assertEqual(text, settings.FALLBACK_MESSAGE, message)

    def test_who_won_the_world_cup_skips_the_classifier(self):
        routed = intent.route("who won the world cup", top_score=lambda m: 0.4, clean=lambda m: m)
        self.assertEqual((routed.kind, routed.path), ("question", "question_heuristic"))   # "who" is a question word

    def test_a_wrong_emotional_label_on_course_content_is_still_overridden(self):
        """Course content stays a question (and is answered, above the threshold) even if the label is emotional."""
        with mock.patch.object(intent, "classify", lambda m: "EMOTIONAL_OTHER"):
            routed = intent.route("star schema", top_score=lambda m: 0.78, clean=lambda m: m)
        self.assertEqual((routed.kind, routed.path), ("question", "similarity_override"))


class EmotionalAndSmallTalkUnchanged(unittest.TestCase):
    def test_im_sick(self):
        result, text = answer("im sick", label="EMOTIONAL_SICK")
        self.assertEqual((result.intent, text), ("emotional", settings.EMOTIONAL_RESPONSES["EMOTIONAL_SICK"]))

    def test_rough_day(self):
        result, text = answer("ugh today has been rough", label="EMOTIONAL_STRESSED")
        self.assertEqual((result.intent, text), ("emotional", settings.EMOTIONAL_RESPONSES["EMOTIONAL_STRESSED"]))

    def test_thanks_is_small_talk_without_the_classifier(self):
        result, text = answer("thanks", label=None)
        self.assertEqual(result.intent, "small_talk")
        self.assertNotEqual(text, settings.FALLBACK_MESSAGE)


class ClassifierInput(unittest.TestCase):
    def test_real_question_covers_off_topic_search_phrases(self):
        prompt = settings.INTENT_CLASSIFIER_PROMPT
        self.assertIn("including topics outside the course", prompt)
        self.assertIn("no question word and no question mark", prompt)
        self.assertIn("asks for nothing", prompt)                        # EMOTIONAL_OTHER is only a statement

    def test_examples_include_off_topic_requests(self):
        off_topic = [m for m, label in settings.INTENT_CLASSIFIER_EXAMPLES
                     if label == "REAL_QUESTION" and m in ("cheap hotels near KLCC", "good songs to study to")]
        self.assertEqual(len(off_topic), 2)
        self.assertTrue(all(label in settings.INTENT_LABELS for _, label in settings.INTENT_CLASSIFIER_EXAMPLES))

    def test_the_case_set_is_fixed_and_not_used_as_examples(self):
        recorded = (ROOT / "evaluation" / "intent_classifier_cases.sha256").read_text(encoding="utf-8").split()[0]
        self.assertEqual(hashlib.sha256(CASES.read_bytes()).hexdigest(), recorded)
        examples = {m.lower() for m, _ in settings.INTENT_CLASSIFIER_EXAMPLES}
        for case in json.loads(CASES.read_text(encoding="utf-8"))["cases"]:
            self.assertNotIn(case["message"].lower(), examples, case["message"])

    def test_the_classifier_still_sends_one_label_request(self):
        """Same request shape as before: system prompt, the examples as turns, then the message; 8-token cap."""
        sent = []
        with mock.patch.object(llm, "complete", lambda messages, max_tokens, timeout: sent.append((messages, max_tokens)) or
                               "REAL_QUESTION"):
            self.assertEqual(intent.classify("best nasi lemak in Shah Alam"), "REAL_QUESTION")
        messages, max_tokens = sent[0]
        self.assertEqual(max_tokens, settings.INTENT_CLASSIFIER_MAX_TOKENS)
        self.assertEqual(len(messages), 1 + 2 * len(settings.INTENT_CLASSIFIER_EXAMPLES) + 1)
        self.assertEqual(messages[-1], {"role": "user", "content": "best nasi lemak in Shah Alam"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
