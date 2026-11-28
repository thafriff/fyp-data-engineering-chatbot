"""
Offline tests of the comparison answer format (src/core/answer_format.py, settings.COMPARISON_*; author, 2026-11-26).

Comparison questions get an app-layer format instruction appended (one-sentence answer, a markdown table with one column per
item and 3-5 aspect rows with citations in the cells, then at most 2 sentences). Like coach mode it is NOT part of notebook 04:
build_messages() and the frozen prompts are untouched. No model, no network.

Run from the project root:  venv\\Scripts\\python tests\\test_comparison_format.py
"""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "evaluation" / "notebook04")]
sys.stdout.reconfigure(encoding="utf-8")

from config import settings  # noqa: E402
from src.core import answer_format, guardrail, intent, llm, rag  # noqa: E402
from src.core.retriever import RetrievedChunk  # noqa: E402

SEP = "\n\n"      # how llm.build_messages and the transforms join message parts
COMPARISONS = [
    "whts the difference between olap and oltp",          # the author's three
    "data lake vs data warehouse",
    "is a data warehouse the same as a database?",
    "what are the differences between ETL and ELT?",
    "how does a data lake differ from a data warehouse",
    "is a fact table different from a dimension table?",
    "how do OLAP and OLTP differ?",
    "Kafka vs. Spark Streaming",
    "OLAP versus OLTP",
    "compare star schema and snowflake schema",
    "comparing batch processing with stream processing",
    "star schema compared to snowflake schema",
    "give me a comparison of 2NF and 3NF",
    "is ELT the same thing as ETL",
    "which is faster, parquet or csv?",
    "is ETL or ELT better for cloud warehouses?",
    "should I use dbt or plain SQL scripts?",
]
NOT_COMPARISONS = [
    "what is a data warehouse",                            # the author's example
    "what is a data warehouse?",
    "what is ETL",
    "how do I load a CSV or JSON file into BigQuery?",     # "or" without a choice word
    "what happens if a pipeline fails or times out?",
    "what difference does partitioning make?",             # "difference" without "between"
    "explain the SQL comparison operators",                # "comparison" not followed by of/between
    "the rows are compared against the schema, why?",
    "how do I install vs code",                            # VS Code
    "What are the five Vs of big data?",                   # Q87: the plural of V, not "versus"
    "explain the 3 Vs",
    "do the same steps apply to BigQuery?",                # "same" without "as"
    "what is the primary key of a dimension table?",
    "best nasi lemak in Shah Alam",
    "write the full SQL query for my assignment",
]


class Detection(unittest.TestCase):
    def test_comparisons_are_detected(self):
        for text in COMPARISONS:
            self.assertIsNotNone(answer_format.detect_comparison(text), text)

    def test_non_comparisons_are_not(self):
        for text in NOT_COMPARISONS:
            self.assertIsNone(answer_format.detect_comparison(text), f"{text!r} matched {answer_format.detect_comparison(text)!r}")

    def test_the_cleaned_search_text_also_counts(self):
        self.assertIsNotNone(answer_format.detect_comparison("whts teh diference", "whats the difference between olap and oltp"))

    def test_switch_turns_it_off(self):
        with mock.patch.object(settings, "COMPARISON_FORMAT_ENABLED", False):
            self.assertIsNone(answer_format.detect_comparison("data lake vs data warehouse"))


class Instruction(unittest.TestCase):
    def test_instruction_asks_for_the_layout(self):
        """2026-11-26 v2: an "In simple terms:" line with one plain-English bullet per item replaces the one-sentence opening."""
        text = settings.COMPARISON_FORMAT_INSTRUCTION
        for phrase in ('Start with the line "In simple terms:" on its own', "one bullet per item being compared",
                       "each bullet on its own line", "the item's name in bold", "what it is used for",
                       "a beginner understands", "at most 20 words", "do not write anything else before the table",
                       "markdown table", "one column per item", "one row per aspect", "3 to 5 rows",
                       "chunk ID in square brackets inside every table cell", "at most 1 sentence"):
            self.assertIn(phrase, text)
        self.assertNotIn("one-sentence direct answer", text)

    def test_opening_rule_text(self):
        text = settings.ANSWER_OPENING_INSTRUCTION
        self.assertIn("Start directly with the answer itself", text)
        self.assertIn('"The course materials"', text)
        self.assertIn("or any sentence in it", text)

    def test_no_course_content_in_the_comparison_template(self):
        """The comparison layout shows only neutral placeholders, so the model cannot copy one example's content."""
        for term in ("OLTP", "OLAP", "ETL", "warehouse", "normalized", "S06-", "S05-", "S04-"):
            self.assertNotIn(term, settings.COMPARISON_FORMAT_INSTRUCTION)

    def test_comparison_layout_has_no_template_to_copy(self):
        """2026-11-28 (author): like the opening rule, the layout is in words only: the old template's angle-bracket
        placeholders ("- **<first item>** is for <...>") are gone, and so is the literal bold markup a model could copy."""
        text = settings.COMPARISON_FORMAT_INSTRUCTION
        self.assertNotRegex(text, r"<[^<>]*>")
        self.assertNotIn("**", text)
        self.assertNotIn("\n- ", text)

    def test_opening_rule_has_no_example_or_template_to_copy(self):
        """2026-11-26 (author): the concrete v2b example "OLTP databases are normalized [S06-0205]" appeared in 13 of 69
        multi-turn answers and 10 cited S06-0205 without retrieving it, so it became the placeholder "<fact> [<chunk ID>]".
        2026-11-27 (author): qwen3.5:2b copied that placeholder literally ("<fact>...</fact> [<S05-0097>]") in 2 of 105 answers,
        so the rule is now in words only: no course content, no chunk ID and no angle-bracket tag."""
        text = settings.ANSWER_OPENING_INSTRUCTION
        self.assertIn("put its chunk ID in square brackets after it", text)
        self.assertIn('instead of introducing it with "The course materials highlight that"', text)
        self.assertNotRegex(text, r"<[^<>]*>")
        for term in ("OLTP", "OLAP", "ETL", "warehouse", "normalized", "S06-", "S05-", "S04-"):
            self.assertNotIn(term, text)

    def test_messages_are_the_normal_ones_plus_the_instruction(self):
        chunk = SimpleNamespace(chunk_id="S06-0204", source_name="Zoomcamp", section="OLTP vs OLAP", text="OLTP ... OLAP ...")
        normal = llm.build_messages("data lake vs data warehouse", [chunk])
        formatted = answer_format.comparison_messages(normal)
        self.assertEqual(formatted[0], normal[0])                                  # the frozen system prompt, untouched
        self.assertEqual(formatted[1]["content"], normal[1]["content"] + "\n\n" + settings.COMPARISON_FORMAT_INSTRUCTION)
        self.assertLess(formatted[1]["content"].index(settings.ANSWER_REMINDER),
                        formatted[1]["content"].index(settings.COMPARISON_FORMAT_INSTRUCTION))


class Pipeline(unittest.TestCase):
    def run_answer(self, message, path="question_heuristic"):
        """Run RagPipeline.answer with a stand-in retriever and routing; return what stream_generate received."""
        seen = {}
        p = rag.RagPipeline.__new__(rag.RagPipeline)
        p.speller = None
        p.retriever = SimpleNamespace(retrieve=lambda text, top_k=None: [
            SimpleNamespace(chunk_id="S06-0204", score=0.85, text="OLTP and OLAP.", source_name="Zoomcamp", section="OLTP vs OLAP")])

        def fake_stream(question, chunks, model=None, corrections=None, transform=None):
            seen["transform"] = transform
            yield "answer"
        routed = SimpleNamespace(kind="question", path=path, task=None)
        with mock.patch.object(intent, "route", lambda q, top_score=None, clean=None: routed), \
                mock.patch.object(llm, "stream_generate", fake_stream):
            result = p.answer(message, model="qwen3.5:2b")
            "".join(result.answer_stream)
        return result, seen["transform"]

    BASE = llm.build_messages("q", [SimpleNamespace(chunk_id="S06-0204", source_name="Zoomcamp", section="OLTP vs OLAP",
                                                    text="OLTP and OLAP.")])

    def user_text(self, transform):
        return transform(self.BASE)[1]["content"]

    def test_comparison_gets_the_opening_rule_then_the_format(self):
        result, transform = self.run_answer("data lake vs data warehouse")
        self.assertTrue(result.comparison_format)
        self.assertEqual(self.user_text(transform), self.BASE[1]["content"] + SEP + settings.ANSWER_OPENING_INSTRUCTION
                         + SEP + settings.COMPARISON_FORMAT_INSTRUCTION)

    def test_other_questions_get_only_the_opening_rule(self):
        result, transform = self.run_answer("what is a data warehouse")
        self.assertFalse(result.comparison_format)
        self.assertEqual(self.user_text(transform), self.BASE[1]["content"] + SEP + settings.ANSWER_OPENING_INSTRUCTION)

    def test_coach_mode_wins(self):
        result, transform = self.run_answer("write the full code comparing ETL vs ELT for my assignment", path="task_request")
        self.assertTrue(result.coach_mode)
        self.assertFalse(result.comparison_format)
        self.assertEqual(self.user_text(transform), self.BASE[1]["content"] + SEP + settings.ANSWER_OPENING_INSTRUCTION
                         + SEP + settings.GUARDRAIL_COACH_INSTRUCTION)

    def test_system_prompt_is_never_changed(self):
        for message, path in (("data lake vs data warehouse", "question_heuristic"), ("what is ETL?", "question_heuristic"),
                              ("write the full SQL query for my assignment", "task_request")):
            _, transform = self.run_answer(message, path=path)
            self.assertEqual(transform(self.BASE)[0], {"role": "system", "content": settings.SYSTEM_PROMPT})

    def test_opening_rule_can_be_switched_off(self):
        with mock.patch.object(settings, "ANSWER_OPENING_RULE_ENABLED", False):
            _, transform = self.run_answer("what is a data warehouse")
            self.assertIsNone(transform)                                  # exactly the pre-2026-11-26 behaviour
            _, transform = self.run_answer("data lake vs data warehouse")
            self.assertEqual(self.user_text(transform), self.BASE[1]["content"] + SEP + settings.COMPARISON_FORMAT_INSTRUCTION)


class CheckScriptSavesPartialData(unittest.TestCase):
    """evaluation/comparison_format_check.py saves after every run, so a stopped run (e.g. a free-tier model that stops
    answering) keeps what it has. The pipeline is faked; the output goes to a temporary folder."""

    def test_a_stopped_run_keeps_its_records(self):
        import importlib.util
        import tempfile
        spec = importlib.util.spec_from_file_location("comparison_format_check", ROOT / "evaluation" / "comparison_format_check.py")
        check = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(check)
        calls = []

        class FakePipeline:
            def answer(self, question, model=None):
                calls.append(question)
                if len(calls) == 2:
                    raise KeyboardInterrupt          # the run is stopped during the second answer
                return SimpleNamespace(answer_stream=iter(["In simple terms:\n- **A** is a.\n- **B** is b."]), cloud_call=None,
                                       chunks=[], routing=SimpleNamespace(path="question_heuristic"), comparison_format=True,
                                       used_fallback=False, model="fake")
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "partial.json"
            with mock.patch.object(check, "RagPipeline", FakePipeline), \
                    mock.patch.object(sys, "argv", ["x", str(out), "--runs", "3", "--arms", "app", "--questions", "a vs b"]):
                with self.assertRaises(KeyboardInterrupt):
                    check.main()
            saved = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(len(saved["records"]), 1)
        self.assertFalse(saved["meta"]["finished"])
        self.assertTrue(saved["records"][0]["simple_terms_line"])


class Notebook04Untouched(unittest.TestCase):
    def test_runners_do_not_use_it(self):
        for name in ("run_generation.py", "freeze_contexts.py", "providers.py", "check_parity.py", "run_judging.py"):
            self.assertNotIn("answer_format", (ROOT / "evaluation" / "notebook04" / name).read_text(encoding="utf-8"), name)

    def test_all_110_frozen_prompts_rebuild_byte_identical(self):
        if not settings.NB04_CONTEXTS_FILE.exists():
            self.skipTest("contexts.json is Git ignored and absent in this checkout")
        import freeze_contexts
        items = json.loads(settings.NB04_CONTEXTS_FILE.read_text(encoding="utf-8"))["items"]
        for item in items:
            chunks = [RetrievedChunk(**c) for c in item["chunks"]]
            messages = llm.build_messages(item["text"], chunks, corrections=freeze_contexts.corrections_for(item, item["chunks"]))
            self.assertEqual(freeze_contexts.prompt_sha256(messages), item["prompt_sha256"], item["id"])
        self.assertEqual(len(items), 110)

    def test_how_many_test_questions_would_get_the_format(self):
        """Information for notebook 05, not a pass/fail rule: which of the 100 evaluation questions the chat app would format."""
        questions = json.loads((ROOT / "evaluation" / "test_questions.json").read_text(encoding="utf-8"))
        questions = questions["questions"] if isinstance(questions, dict) else questions
        hits = [q["id"] for q in questions if answer_format.detect_comparison(q["question"])]
        print(f"\n  [info] {len(hits)} of {len(questions)} evaluation questions match: {hits}")
        self.assertLess(len(hits), len(questions))


if __name__ == "__main__":
    unittest.main(verbosity=2)
