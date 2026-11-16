"""Offline tests of the search text (leading greeting and filler removal), the routing that reads it, the spelling hint (incl.
British spellings) and the freeze script's use of the shared function. No model, no network; writes nothing.

Run from the project root:  venv\\Scripts\\python tests\\test_search_text.py
"""
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
os.environ["HF_HUB_OFFLINE"] = "1"

from config import settings   # noqa: E402
from src.core import intent, llm, rag   # noqa: E402
from src.core.search_text import strip_leading_filler as strip   # noqa: E402


class Stripped(unittest.TestCase):
    def test_greetings(self):
        cases = {
            "hi what is olap": "what is olap",
            "Hello, what is a star schema?": "what is a star schema?",
            "hey can you explain ETL": "explain ETL",
            "salam what is a data lake": "what is a data lake",
            "Assalamualaikum! what is dbt?": "what is dbt?",
            "assalamu alaikum, what is Kafka": "what is Kafka",
            "Assalamu\u2019alaikum what is Kafka": "what is Kafka",
            "Good morning... what is a fact table?": "what is a fact table?",
            "HI   THERE!!  what is OLTP": "what is OLTP",
            "hiya - what is BigQuery": "what is BigQuery",
        }
        for text, expected in cases.items():
            self.assertEqual(strip(text), expected, text)

    def test_filler(self):
        cases = {
            "please explain normalisation": "explain normalisation",
            "Can you please explain joins?": "explain joins?",
            "could you tell me what a data mart is": "tell me what a data mart is",
            "Could you describe a star schema": "describe a star schema",
            "can u compare ETL and ELT": "compare ETL and ELT",
            "would you please define a fact table": "define a fact table",
            "I want to know what OLAP is": "what OLAP is",
            "i'd like to know about partitioning": "about partitioning",
            "kindly, what is Terraform": "what is Terraform",
        }
        for text, expected in cases.items():
            self.assertEqual(strip(text), expected, text)

    def test_stacked_phrases_are_all_removed(self):
        self.assertEqual(strip("hi, good morning, please can you explain ETL?"), "explain ETL?")
        self.assertEqual(strip("Hello there! I would like to know what dbt is"), "what dbt is")

    def test_request_verbs_are_kept(self):
        for text in ("explain ETL", "describe a data lake", "compare OLAP and OLTP", "Explain the star schema"):
            self.assertEqual(strip(text), text, text)


class NotStripped(unittest.TestCase):
    def test_words_that_only_start_like_a_greeting(self):
        for text in ("hive partitioning in Hadoop", "high availability in HDFS", "Hive vs Spark SQL",
                     "hierarchies in a dimension table", "heyday of MapReduce", "hi-res images in a data lake",
                     "pleased with dbt tests?", "Salamander schema?", "goodness of fit", "hello world in PySpark",
                     "Hello World example in Spark"):
            self.assertEqual(strip(text), text, text)

    def test_middle_and_end_are_never_touched(self):
        for text in ("what is olap, hi", "what is ETL please", "explain hi and hello", "what is OLAP? thanks hello",
                     "is it ok to say hi to the DBA", "what is a data lake salam"):
            self.assertEqual(strip(text), text, text)

    def test_a_message_that_is_only_greeting_is_left_as_typed(self):
        for text in ("hi", "Hello!", "hey there", "good morning, please", "salam"):
            self.assertEqual(strip(text), text, text)

    def test_question_words_and_course_terms_stay(self):
        for text in ("what is OLAP?", "How does Kafka work?", "Explain the star schema", "Can you use dbt with BigQuery?",
                     "Could you partition a table by date?", "can you explainthis", "Tell me about ETL",
                     "Why use a data warehouse?"):
            self.assertEqual(strip(text), text, text)

    def test_none_of_the_110_frozen_items_change(self):
        items = json.loads(settings.NB04_MANIFEST_FILE.read_text(encoding="utf-8"))["items"]
        questions = {q["id"]: q["question"] for q in json.loads(settings.TEST_QUESTIONS_FILE.read_text(encoding="utf-8"))["questions"]}
        probes = {p["id"]: p["request"] for p in json.loads(settings.NB04_PROBES_FILE.read_text(encoding="utf-8"))["probes"]}
        texts = {**questions, **probes}
        self.assertEqual(len(texts), 110)
        changed = [i for i, t in texts.items() if strip(t) != t]
        self.assertEqual(changed, [])


class SpellingHint(unittest.TestCase):
    CHUNK = SimpleNamespace(chunk_id="S05-0138", source_name="DE Wiki", section="OLAP", text="OLAP is a data model.")
    OLAP = ["OLAP is a data model for analysis.", "OLTP vs OLAP"]

    def test_hint_only_for_qualifying_corrections(self):
        from src.core.search_text import spelling_corrections as sc
        self.assertEqual(sc("hey what is olpa", "what is olap", self.OLAP), [("olpa", "olap")])
        self.assertEqual(sc("how do I load dat into BigQuery?", "how do I load at into BigQuery?", ["load data at once"]), [])  # 2 letters
        self.assertEqual(sc("can a wifi network fail", "can a wife network fail", ["a network"]), [])   # not in the excerpts
        self.assertEqual(sc("hi what is olap", "what is olap", self.OLAP), [])                          # greeting removal only
        self.assertEqual(sc("what is OLAP?", "what is OLAP?", self.OLAP), [])
        self.assertEqual(sc("whst is olpa", "what is olap", self.OLAP), [])     # "what" is not in the excerpts: no hint at all
        self.assertEqual(sc("whst is olpa", "what is olap", self.OLAP + ["what it is"]), [("whst", "what"), ("olpa", "olap")])
        self.assertEqual(sc("what is olpap", "what is olap", ["OLAPS only"]), [])     # whole words only

    def test_british_spellings_are_not_typos(self):
        from src.core.search_text import is_british_variant, spelling_corrections as sc
        for typed, corrected in (("Normalise", "normalize"), ("colour", "color"), ("analyse", "analyze"),
                                 ("organisation", "organization"), ("organisations", "organizations"), ("behaviour", "behavior"),
                                 ("centre", "center"), ("catalogue", "catalog"), ("modelling", "modeling"),
                                 ("licence", "license"), ("programme", "program"), ("normalised", "normalized")):
            self.assertTrue(is_british_variant(typed, corrected), typed)
        for typed, corrected in (("olpa", "olap"), ("dat", "at"), ("wifi", "wife"), ("whst", "what"), ("colur", "color")):
            self.assertFalse(is_british_variant(typed, corrected), typed)
        excerpts = ["normalize the tables", "pick a color", "analyze the data", "the organization", "OLAP cubes"]
        self.assertEqual(sc("Normalise it to third normal form", "normalize it to third normal form", excerpts), [])
        self.assertEqual(sc("what colour is it", "what color is it", excerpts), [])
        self.assertEqual(sc("analyse the data", "analyze the data", excerpts), [])
        self.assertEqual(sc("why does an organisation need it", "why does an organization need it", excerpts), [])
        self.assertEqual(sc("what is olpa", "what is olap", excerpts), [("olpa", "olap")])
        # A British spelling next to a real typo: only the typo is hinted.
        self.assertEqual(sc("normalise the olpa cube", "normalize the olap cube", excerpts), [("olpa", "olap")])

    def test_without_corrections_the_prompt_is_exactly_the_old_one(self):
        old_user = ("Context excerpts from the course materials:\n\n[S05-0138] (source: DE Wiki; section: OLAP)\n"
                    "OLAP is a data model.\n\nStudent question: what is OLAP?\n\n" + settings.ANSWER_REMINDER)
        messages = llm.build_messages("what is OLAP?", [self.CHUNK])
        self.assertEqual(messages, [{"role": "system", "content": settings.SYSTEM_PROMPT}, {"role": "user", "content": old_user}])
        self.assertEqual(llm.build_messages("what is OLAP?", [self.CHUNK], corrections=[]), messages)

    def test_the_hint_asks_for_assuming_you_mean_and_never_mentions_the_search(self):
        user = llm.build_messages("what is olpa", [self.CHUNK], corrections=[("olpa", "olap")])[1]["content"]
        self.assertIn('Student question: what is olpa\nNote: "olpa" is probably a typo for "olap". If so, begin your answer with '
                      '"Assuming you mean olap," and answer that.\n\n' + settings.ANSWER_REMINDER, user)
        self.assertNotIn("search", llm.spelling_hint([("olpa", "olap")]).lower())
        self.assertNotIn("interpreted", llm.spelling_hint([("olpa", "olap")]).lower())
        self.assertEqual(llm.spelling_hint([("whst", "what"), ("olpa", "olap")]),
                         'Note: "whst" is probably a typo for "what"; "olpa" is probably a typo for "olap". If so, begin your '
                         'answer with "Assuming you mean what olap," and answer that.')

    def test_the_pipeline_passes_the_corrections_to_the_model(self):
        seen = {}
        pipeline = rag.RagPipeline.__new__(rag.RagPipeline)
        pipeline.speller = SimpleNamespace(correct=lambda text: text.replace("olpa", "olap").replace(" dat ", " at "))
        pipeline.retriever = SimpleNamespace(retrieve=lambda text, top_k=None: [
            SimpleNamespace(chunk_id="S05-0138", score=0.9, text="OLAP is a model. Load data at once.")])

        def fake_stream(question, chunks, model=None, corrections=None, transform=None):
            seen.update(question=question, corrections=corrections)
            yield "answer"
        with mock.patch.object(intent, "route", lambda q, top_score=None, clean=None: SimpleNamespace(kind="question", path="question_heuristic")), \
                mock.patch.object(llm, "stream_generate", fake_stream):
            for text, corrections in (("hey what is olpa", [("olpa", "olap")]), ("hi what is olap", []),
                                      ("how do I load dat into BigQuery?", [])):
                "".join(pipeline.answer(text, model="llama3.2:3b").answer_stream)
                self.assertEqual((seen["question"], seen["corrections"]), (text, corrections), text)


class Pipeline(unittest.TestCase):
    """The stripped text is embedded; the original question is what the model gets."""

    def test_search_uses_stripped_text_and_model_gets_original(self):
        seen = {}
        chunk = SimpleNamespace(chunk_id="S05-0138", score=0.9, text="OLAP")
        pipeline = rag.RagPipeline.__new__(rag.RagPipeline)
        pipeline.speller = SimpleNamespace(correct=lambda text: text.replace("olpa", "olap"))
        pipeline.retriever = SimpleNamespace(retrieve=lambda text, top_k=None: seen.setdefault("search", text) and [chunk])

        def fake_stream(question, chunks, model=None, **kwargs):
            seen["model_question"] = question
            yield "answer"
        with mock.patch.object(intent, "route", lambda q, top_score=None, clean=None: SimpleNamespace(kind="question", path="question_heuristic")), \
                mock.patch.object(llm, "stream_generate", fake_stream):
            result = pipeline.answer("Salam, what is olpa", model="llama3.2:3b")
            "".join(result.answer_stream)
        self.assertEqual(seen["search"], "what is olap")
        self.assertEqual(result.search_query, "what is olap")
        self.assertEqual(seen["model_question"], "Salam, what is olpa")

    def test_emotional_override_scores_the_cleaned_text(self):
        """The router's top_score gets the cleaned search text, the same as retrieval."""
        searched = []
        pipeline = rag.RagPipeline.__new__(rag.RagPipeline)
        pipeline.speller = SimpleNamespace(correct=lambda text: text)
        pipeline.retriever = SimpleNamespace(
            retrieve=lambda text, top_k=None: searched.append((text, top_k)) or [SimpleNamespace(chunk_id="S05-0138", score=0.708)])
        with mock.patch.object(intent, "classify", lambda message: "EMOTIONAL_OTHER"):
            routed = intent.route("hi what is olap", top_score=pipeline._top_score)
        self.assertEqual(searched, [("what is olap", 1)])
        self.assertEqual((routed.kind, routed.path, routed.top_score), ("question", "similarity_override", 0.708))

    def test_emotional_message_still_gets_the_empathy_reply(self):
        pipeline = rag.RagPipeline.__new__(rag.RagPipeline)
        pipeline.speller = SimpleNamespace(correct=lambda text: text)
        pipeline.retriever = SimpleNamespace(retrieve=lambda text, top_k=None: [SimpleNamespace(chunk_id="x", score=0.55)])
        with mock.patch.object(intent, "classify", lambda message: "EMOTIONAL_SICK"):
            routed = intent.route("hi im sick", top_score=pipeline._top_score)
        self.assertEqual((routed.kind, routed.reply), ("emotional", settings.EMOTIONAL_RESPONSES["EMOTIONAL_SICK"]))

    def test_greeting_then_question_word_skips_the_classifier(self):
        pipeline = rag.RagPipeline.__new__(rag.RagPipeline)
        pipeline.speller = SimpleNamespace(correct=lambda text: text.replace("olpa", "olap").replace("whst", "what"))
        with mock.patch.object(intent, "classify", side_effect=AssertionError("classifier must not be called")):
            for message in ("hey what is olpa", "hi how does Kafka work", "hello, whst is a data lake", "salam why use dbt"):
                routed = intent.route(message, top_score=None, clean=pipeline.search_text)
                self.assertEqual((routed.kind, routed.path), ("question", "question_heuristic"), message)

    def test_non_questions_still_reach_the_classifier_and_hi_stays_a_greeting(self):
        pipeline = rag.RagPipeline.__new__(rag.RagPipeline)
        pipeline.speller = None
        called = []
        with mock.patch.object(intent, "classify", lambda message: called.append(message) or "REAL_QUESTION"):
            intent.route("hey star schema", clean=pipeline.search_text)
            intent.route("hi im sick", clean=pipeline.search_text)
            greeting = intent.route("hi", clean=pipeline.search_text)
        self.assertEqual(called, ["hey star schema", "hi im sick"])      # the classifier gets the message as typed
        self.assertEqual(greeting.kind, "small_talk")

    def test_the_pipeline_passes_the_shared_function_to_the_router(self):
        seen = {}
        pipeline = rag.RagPipeline.__new__(rag.RagPipeline)
        pipeline.speller = None
        pipeline.retriever = SimpleNamespace(retrieve=lambda text, top_k=None: [])

        def fake_route(message, top_score=None, clean=None):
            seen["clean"] = clean
            return SimpleNamespace(kind="small_talk", reply="hi")
        with mock.patch.object(intent, "route", fake_route):
            pipeline.answer("hi")
        self.assertEqual(seen["clean"], pipeline.search_text)

    def test_freeze_contexts_uses_the_shared_function(self):
        sys.path.insert(0, str(ROOT / "evaluation" / "notebook04"))
        import freeze_contexts
        calls = []
        pipeline = SimpleNamespace(search_text=lambda text: calls.append(text) or "cleaned",
                                   speller=SimpleNamespace(correct=lambda text: self.fail("must not call the speller directly")),
                                   retriever=SimpleNamespace(retrieve=lambda text: calls.append(("retrieve", text)) or []))
        item = freeze_contexts.freeze_item(pipeline, {"id": "X", "kind": "question", "text": "hi what is olap"})
        self.assertEqual((calls, item["search_query"]), (["hi what is olap", ("retrieve", "cleaned")], "cleaned"))

    def test_switch_turns_it_off(self):
        pipeline = rag.RagPipeline.__new__(rag.RagPipeline)
        pipeline.speller = None
        with mock.patch.object(settings, "SEARCH_STRIP_LEADING_ENABLED", False):
            self.assertEqual(pipeline.search_text("hi what is olap"), "hi what is olap")
        self.assertEqual(pipeline.search_text("hi what is olap"), "what is olap")


if __name__ == "__main__":
    unittest.main(verbosity=1)
