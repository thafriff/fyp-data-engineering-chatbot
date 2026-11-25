"""
Offline tests of the topic offer after an emotional reply (settings.EMOTIONAL_TOPIC_OFFERS, intent.topic_offer).

The kind fixed reply is kept; when the message names a course topic, one fixed line follows it, offering to start with that
topic and giving a starter question. No model writes any of it: the classifier is faked here, and every Ollama call is made to
fail, so a test would break if the offer called a model. The last class loads the real retriever (local embedding model and
FAISS index, no network) to prove every starter question is actually answered.

Run from the project root:  venv\\Scripts\\python tests\\test_emotional_topic_offer.py
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
os.environ["HF_HUB_OFFLINE"] = "1"

from config import settings  # noqa: E402
from src.core import intent, llm  # noqa: E402

SQL_LINE = 'Want me to start with the basics of SQL? Just ask: "What is SQL?"'


def no_model(*args, **kwargs):
    raise AssertionError("a model was called")


def route(message: str, label: str = "EMOTIONAL_STRESSED", score: float = 0.60):
    """Route a message with the classifier faked to `label` and the course similarity faked to `score`."""
    with mock.patch.object(intent, "classify", lambda m: label), mock.patch.object(llm, "complete", no_model), \
            mock.patch.object(llm.requests, "post", no_model):
        return intent.route(message, top_score=lambda m: score, clean=lambda m: m)


class Offer(unittest.TestCase):
    def test_the_authors_example(self):
        r = route("ugh sql is so hard i want to give up")
        self.assertEqual(r.kind, "emotional")
        self.assertEqual(r.reply, settings.EMOTIONAL_RESPONSES["EMOTIONAL_STRESSED"] + "\n\n" + SQL_LINE)

    def test_the_kind_reply_comes_first_and_is_unchanged(self):
        for label in ("EMOTIONAL_SICK", "EMOTIONAL_TIRED", "EMOTIONAL_STRESSED", "EMOTIONAL_OTHER"):
            r = route("im so tired of sql", label=label)
            self.assertTrue(r.reply.startswith(settings.EMOTIONAL_RESPONSES[label] + "\n\n"), label)
            self.assertEqual(r.reply.count("Want me to start"), 1)

    def test_no_topic_no_extra_line(self):
        for message in ("ugh today has been rough", "im sick", "lol", "i want to give up"):
            r = route(message)
            self.assertEqual(r.reply, settings.EMOTIONAL_RESPONSES["EMOTIONAL_STRESSED"], message)

    def test_each_topic(self):
        cases = {"ugh joins confuse me so much": "SQL", "normalisation is killing me": "SQL",
                 "i hate ETL": "ETL and ELT", "elt makes me sad": "ETL and ELT",
                 "my pipeline keeps breaking and im done": "data pipelines",
                 "data cleaning is so boring": "data cleaning", "data quality stresses me out": "data quality",
                 "batch processing makes no sense to me": "data processing",
                 "datawarehouse stuff is too much": "data warehousing", "star schema makes me cry": "data warehousing",
                 "spark hates me": "big data", "kafka is too hard": "big data", "big data scares me": "big data"}
        for message, topic in cases.items():
            self.assertIn(f"the basics of {topic}?", route(message).reply, message)

    def test_the_first_topic_mentioned_wins_and_only_one_line(self):
        r = route("kafka and sql are both too hard")
        self.assertIn("the basics of big data?", r.reply)
        self.assertNotIn("SQL", r.reply)

    def test_whole_words_only(self):
        for message in ("my squirrel ran away", "the elite squad lost", "i feel sparkly today", "mysql"):
            self.assertIsNone(intent.topic_offer(message), message)

    def test_the_cleaned_text_counts_too(self):
        """A typo corrected in the search text ("sqll" -> "sql") still names the topic."""
        with mock.patch.object(intent, "classify", lambda m: "EMOTIONAL_STRESSED"):
            r = intent.route("ugh sqll is so hard", top_score=lambda m: 0.6, clean=lambda m: m.replace("sqll", "sql"))
        self.assertTrue(r.reply.endswith(SQL_LINE))

    def test_a_course_question_is_still_a_question(self):
        """The override is unchanged: close to course content -> a question, no emotional reply at all."""
        r = route("ugh sql joins confuse me", score=0.78)
        self.assertEqual((r.kind, r.path, r.reply), ("question", "similarity_override", None))

    def test_offer_text_is_fixed(self):
        self.assertEqual(intent.topic_offer("ugh sql"), SQL_LINE)


class StarterQuestionsAreAnswered(unittest.TestCase):
    """Every starter question must reach retrieval and score at or above the threshold, or the offer would lead to the
    'not in the course materials' fallback."""

    @classmethod
    def setUpClass(cls):
        from src.core.rag import RagPipeline
        cls.pipeline = RagPipeline()

    def test_each_starter_question(self):
        for _, topic, question in settings.EMOTIONAL_TOPIC_OFFERS:
            with self.subTest(topic=topic):
                with mock.patch.object(intent, "classify", no_model):
                    routed = intent.route(question, top_score=self.pipeline._top_score, clean=self.pipeline.search_text)
                self.assertEqual((routed.kind, routed.path), ("question", "question_heuristic"))
                score = self.pipeline.retriever.retrieve(self.pipeline.search_text(question), top_k=1)[0].score
                self.assertGreaterEqual(score, settings.SIMILARITY_THRESHOLD, f"{question} scores {score:.3f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
