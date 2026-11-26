"""Offline tests of the code-level guardrail router (src/core/guardrail.py) and its place in the routing flow. No model, no
network. The held-out evaluation set is NOT used here (it is run by evaluation/guardrail_router_eval.py and reported as
measured); these are unit checks of the mechanism.

Run from the project root:  venv\\Scripts\\python tests\\test_guardrail_router.py
"""

import hashlib
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "evaluation" / "notebook04")]
sys.stdout.reconfigure(encoding="utf-8")
os.environ["HF_HUB_OFFLINE"] = "1"

from config import settings  # noqa: E402
from src.core import guardrail, intent, llm, rag  # noqa: E402
from src.core.retriever import RetrievedChunk  # noqa: E402

CHUNK = SimpleNamespace(chunk_id="S05-0138", source_name="DE Wiki", section="OLAP", text="OLAP is a data model.", score=0.9)


class Detection(unittest.TestCase):
    def test_the_three_signals_together_make_a_task(self):
        d = guardrail.detect_task_request("Write the SQL query for my assignment that joins the trips and zones tables")
        self.assertTrue(d.is_task)
        self.assertEqual((d.verb, d.deliverable, d.assessed), ("Write", "SQL", "my assignment"))

    def test_complete_alone_counts_as_the_c_signal(self):
        self.assertTrue(guardrail.detect_task_request("write the complete MapReduce word count program").is_task)

    def test_each_missing_signal_blocks(self):
        self.assertFalse(guardrail.detect_task_request("how do I write a query that joins two tables?").is_task)   # no C
        self.assertFalse(guardrail.detect_task_request("explain what my assignment means by a fact table").is_task)  # no verb
        self.assertFalse(guardrail.detect_task_request("write my full assignment").is_task)                        # no deliverable

    def test_one_word_cannot_be_both_verb_and_completeness_signal(self):
        d = guardrail.detect_task_request("is a complete schema better")
        self.assertFalse(d.is_task)
        self.assertIn("verb and completeness signal are the same word", d.reasons)
        self.assertTrue(guardrail.detect_task_request("complete the full schema for my report").is_task)

    def test_coach_instruction_withholds_the_final_artefact(self):
        self.assertIn("Withhold the final artefact", settings.GUARDRAIL_COACH_INSTRUCTION)
        self.assertIn("even if it is short", settings.GUARDRAIL_COACH_INSTRUCTION)

    def test_coach_instruction_overrides_the_direct_answer_rule(self):
        """2026-11-26: the answer-style prompt asks for a direct answer first; in coach mode the appended instruction says not to,
        and forbids listing the final tables, schema or design in prose too. It comes after the reminder, so it is read last."""
        self.assertIn("do not open with a direct answer", settings.GUARDRAIL_COACH_INSTRUCTION)
        self.assertIn("Never list the final tables, schema or design, in code or in prose.", settings.GUARDRAIL_COACH_INSTRUCTION)
        chunk = SimpleNamespace(chunk_id="S01-0287", source_name="BCcampus", section="3NF", text="Third normal form ...")
        user = guardrail.coach_messages(llm.build_messages("Normalise my orders table to 3NF and give me the final tables",
                                                           [chunk]))[1]["content"]
        self.assertTrue(user.endswith(settings.GUARDRAIL_COACH_INSTRUCTION))
        self.assertLess(user.index(settings.ANSWER_REMINDER), user.index("do not open with a direct answer"))

    def test_example_exception(self):
        self.assertFalse(guardrail.detect_task_request("give me a full example query of a JOIN").is_task)
        self.assertTrue(guardrail.detect_task_request("give me a full example query for my assignment").is_task)

    def test_reasons_explain_a_pass_through(self):
        d = guardrail.detect_task_request("show me an example of a JOIN")
        self.assertFalse(d.is_task)
        self.assertIn("no deliverable", d.reasons)


class CoachMessages(unittest.TestCase):
    def test_instruction_is_appended_and_the_system_prompt_untouched(self):
        normal = llm.build_messages("write the full code for my assignment", [CHUNK])
        coach = guardrail.coach_messages(normal)
        self.assertEqual(coach[0], normal[0])
        self.assertTrue(coach[1]["content"].startswith(normal[1]["content"]))
        self.assertTrue(coach[1]["content"].endswith(settings.GUARDRAIL_COACH_INSTRUCTION))
        self.assertEqual(normal, llm.build_messages("write the full code for my assignment", [CHUNK]))  # not mutated


class CodeCap(unittest.TestCase):
    def test_long_blocks_are_cut_short_ones_kept(self):
        short = "```sql\nSELECT 1;\nSELECT 2;\n```"
        long_block = "```python\n" + "\n".join(f"line{i}" for i in range(20)) + "\n```"
        text, cut = guardrail.cap_code_blocks(f"Intro\n{short}\nMiddle\n{long_block}\nEnd", max_lines=8)
        self.assertEqual(cut, 1)
        self.assertIn(short, text)
        self.assertIn("line7\n" + settings.GUARDRAIL_CODE_CUT_NOTE + "\n```", text)
        self.assertNotIn("line8", text)
        self.assertTrue(text.endswith("\nEnd"))

    def test_unclosed_block_is_capped_too(self):
        text, cut = guardrail.cap_code_blocks("```\n" + "\n".join("x" for _ in range(12)), max_lines=8)
        self.assertEqual(cut, 1)
        self.assertEqual(text.count("x"), 8)
        self.assertIn(settings.GUARDRAIL_CODE_CUT_NOTE, text)

    def test_default_limit_is_the_setting(self):
        block = "```\n" + "\n".join("y" for _ in range(settings.GUARDRAIL_CODE_MAX_LINES + 1)) + "\n```"
        self.assertEqual(guardrail.cap_code_blocks(block)[1], 1)
        block = "```\n" + "\n".join("y" for _ in range(settings.GUARDRAIL_CODE_MAX_LINES)) + "\n```"
        self.assertEqual(guardrail.cap_code_blocks(block)[1], 0)


class Routing(unittest.TestCase):
    def pipeline(self):
        p = rag.RagPipeline.__new__(rag.RagPipeline)
        p.speller = None
        p.retriever = SimpleNamespace(retrieve=lambda text, top_k=None: [CHUNK])
        return p

    def test_task_request_is_routed_before_the_obvious_question_check(self):
        p = self.pipeline()
        with mock.patch.object(intent, "classify", side_effect=AssertionError("classifier must not be called")):
            routed = intent.route("Give me the full PySpark code for my assignment that groups the taxi trips", clean=p.search_text)
        self.assertEqual((routed.kind, routed.path), ("question", "task_request"))
        self.assertTrue(routed.task.is_task)

    def test_greeting_before_a_task_request_is_removed_first(self):
        p = self.pipeline()
        routed = intent.route("hi, write the complete dbt schema.yml for my project", clean=p.search_text)
        self.assertEqual(routed.path, "task_request")

    def test_ordinary_requests_keep_their_old_path(self):
        p = self.pipeline()
        with mock.patch.object(intent, "classify", side_effect=AssertionError("classifier must not be called")):
            routed = intent.route("show me an example of a JOIN", clean=p.search_text)
        self.assertEqual(routed.path, "question_heuristic")
        self.assertEqual(intent.route("hi", clean=p.search_text).kind, "small_talk")

    def test_pipeline_uses_coach_messages_only_for_a_task_request(self):
        p = self.pipeline()
        seen = {}

        def fake_stream(question, chunks, model=None, corrections=None, transform=None):
            seen["transform"] = transform
            yield "answer"
        chunk = SimpleNamespace(chunk_id="S05-0138", source_name="DE Wiki", section="OLAP", text="OLAP is a data model.")
        base = llm.build_messages("q", [chunk])
        with mock.patch.object(llm, "stream_generate", fake_stream):
            task = p.answer("write the complete SQL query for my assignment", model="llama3.2:3b")
            "".join(task.answer_stream)
            self.assertTrue(task.coach_mode)
            # 2026-11-26: every app answer also gets the opening rule first (answer_format.app_transform), then the coach line.
            self.assertTrue(seen["transform"](base)[1]["content"].endswith(settings.GUARDRAIL_COACH_INSTRUCTION))
            normal = p.answer("what is OLAP?", model="llama3.2:3b")
            "".join(normal.answer_stream)
            self.assertFalse(normal.coach_mode)
            self.assertNotIn(settings.GUARDRAIL_COACH_INSTRUCTION, seen["transform"](base)[1]["content"])

    def test_router_can_be_switched_off(self):
        p = self.pipeline()
        with mock.patch.object(settings, "GUARDRAIL_ROUTER_ENABLED", False):
            routed = intent.route("Give me the full PySpark code for my assignment", clean=p.search_text)
        self.assertEqual(routed.path, "question_heuristic")


class FreezeUntouched(unittest.TestCase):
    """The router is an app-layer feature: the candidate prompt is unchanged."""

    def test_system_prompt_sha(self):
        self.assertTrue(hashlib.sha256(settings.SYSTEM_PROMPT.encode("utf-8")).hexdigest().startswith("677e8b3b"))   # re-registered 2026-11-24 (was fe8912d1)

    def test_all_110_frozen_prompts_rebuild_byte_identical(self):
        import freeze_contexts
        items = json.loads(settings.NB04_CONTEXTS_FILE.read_text(encoding="utf-8"))["items"]
        for item in items:
            chunks = [RetrievedChunk(**c) for c in item["chunks"]]
            messages = llm.build_messages(item["text"], chunks, corrections=freeze_contexts.corrections_for(item, item["chunks"]))
            self.assertEqual(freeze_contexts.prompt_sha256(messages), item["prompt_sha256"], item["id"])
        self.assertEqual(len(items), 110)

    def test_the_held_out_set_is_unchanged(self):
        recorded = (ROOT / "evaluation" / "guardrail_router_cases.sha256").read_text(encoding="utf-8").split()[0]
        self.assertEqual(hashlib.sha256((ROOT / "evaluation" / "guardrail_router_cases.json").read_bytes()).hexdigest(), recorded)


if __name__ == "__main__":
    unittest.main(verbosity=1)
