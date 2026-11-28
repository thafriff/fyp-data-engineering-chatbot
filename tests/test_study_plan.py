"""
Offline tests for the study plan (feature A; docs/feature_design_2026-11-28.md). NOTHING HERE CALLS A MODEL: the goal/question writer's
model calls, the quiz writer and the chat answer are faked, and the intent classifier is replaced. The routing cases come from the hashed
design set written before the code (evaluation/study_plan_cases.json), whose hash is checked first. The structural checks run on the
real knowledge base (the retriever loads once).

Run from the project root:  venv\\Scripts\\python tests\\test_study_plan.py
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
sys.path[:0] = [str(ROOT)]
sys.stdout.reconfigure(encoding="utf-8")
os.environ["HF_HUB_OFFLINE"] = "1"

from streamlit.testing.v1 import AppTest  # noqa: E402

from config import settings  # noqa: E402
from src.core import intent, llm, quiz, study_plan  # noqa: E402
from src.core.rag import RagPipeline  # noqa: E402

EVAL = ROOT / "evaluation"
CASES = json.loads((EVAL / "study_plan_cases.json").read_text(encoding="utf-8"))
APP = str(ROOT / "src" / "app" / "app.py")
PLAN_PAGE = str(ROOT / "src" / "app" / "pages" / "1_Study_plan.py")


def no_classifier():
    return mock.patch.object(intent, "classify", lambda message: "REAL_QUESTION")


def chunk(chunk_id, section, score=0.8, source_id="S04", text="text"):
    return SimpleNamespace(chunk_id=chunk_id, section=section, score=score, source_id=source_id, source_name=f"Src {source_id}",
                           text=text, source_type="textbook", page=None, file="f", topic="t")


class HashedSet(unittest.TestCase):
    def test_set_is_the_one_hashed_before_the_code(self):
        recorded = (EVAL / "study_plan_cases.sha256").read_text(encoding="utf-8").split()[0]
        self.assertEqual(hashlib.sha256((EVAL / "study_plan_cases.json").read_bytes()).hexdigest(), recorded)


class Routing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = RagPipeline(retriever=SimpleNamespace())

    def route(self, text):
        with no_classifier():
            return intent.route(text, top_score=lambda m: 0.0, clean=self.pipeline.search_text)

    def test_in_scope_and_malay_parse_the_budget_and_topic(self):
        for case in CASES["in_scope"] + CASES["malay"]:
            decision = self.route(case["text"])
            self.assertEqual(decision.kind, "study_plan", case["id"])
            self.assertEqual(decision.plan_request.minutes, case["budget_minutes"], case["id"])
            self.assertEqual(decision.plan_request.topic.lower(), case["topic"].lower(), case["id"])

    def test_fixed_replies(self):
        expect = {"SB01": "broad", "SS01": "too_short", "SM01": "ask_time", "SM02": "ask_topic"}
        for case in CASES["too_broad"] + CASES["too_short"] + CASES["missing"]:
            decision = self.route(case["text"])
            self.assertEqual(decision.kind, "study_plan", case["id"])
            self.assertEqual(study_plan.problem(decision.plan_request)[0], expect[case["id"]], case["id"])

    def test_not_plan_cases_do_not_route_to_the_plan(self):
        for case in CASES["not_plan"]:
            self.assertNotEqual(self.route(case["text"]).kind, "study_plan", case["id"])

    def test_no_evaluation_question_probe_router_or_quiz_case_reaches_the_plan(self):
        texts = [q["question"] for q in json.loads((EVAL / "test_questions.json").read_text(encoding="utf-8"))["questions"]]
        texts += [p["request"] for p in json.loads((EVAL / "notebook04" / "guardrail_probes.json").read_text(encoding="utf-8"))["probes"]]
        router = json.loads((EVAL / "guardrail_router_cases.json").read_text(encoding="utf-8"))
        texts += [c["text"] for c in router["recall"] + router["near_miss"]]
        for name in ("quiz_cases", "quiz_cases_holdout"):
            sets = json.loads((EVAL / f"{name}.json").read_text(encoding="utf-8"))
            texts += [c["text"] for group, cases in sets.items() if group != "meta" for c in cases if "text" in c]
        self.assertEqual([t for t in texts if self.route(t).kind == "study_plan"], [])

    def test_limits_and_hours_a_day(self):
        self.assertEqual(study_plan.problem(study_plan.detect("plan 30 days for SQL"))[0], "too_long")
        self.assertEqual(study_plan.problem(study_plan.detect("plan 41 hours for SQL"))[0], "too_long")
        request = study_plan.detect("plan 3 days for SQL, 3 hours a day")
        self.assertEqual((request.minutes, request.days, request.minutes_per_day, request.per_day_given), (540, 3, 180, True))
        self.assertEqual(study_plan.detect("study plan for SQL in half an hour").minutes, 30)


class Time(unittest.TestCase):
    def test_session_lengths_use_the_whole_budget(self):
        expected = {20: [20], 45: [45], 60: [60], 90: [45, 35], 120: [55, 55], 180: [54, 53, 53], 240: [45, 45, 45, 45, 20]}
        for budget, lengths in expected.items():
            self.assertEqual(study_plan.session_lengths(budget), lengths, budget)
            self.assertEqual(sum(lengths) + settings.PLAN_BREAK_MINUTES * (len(lengths) - 1), budget, budget)
        for budget in range(20, 600, 7):
            lengths = study_plan.session_lengths(budget)
            self.assertEqual(sum(lengths) + 10 * (len(lengths) - 1), budget, budget)
            self.assertTrue(all(settings.PLAN_MIN_SESSION_MINUTES <= n <= 74 for n in lengths), (budget, lengths))

    def test_allocation_caps_and_relevance(self):
        groups = [{"title": f"T{i}", "chunk_ids": ["x"] * n, "best": b} for i, (n, b) in enumerate([(24, 0.9), (1, 0.7), (2, 0.8)])]
        placed, more, review = study_plan.allocate(groups, 60)
        self.assertEqual([(g["title"], g["minutes"]) for g in placed], [("T0", 45), ("T2", 15)])
        self.assertEqual([g["title"] for g in more], ["T1"])
        placed, more, review = study_plan.allocate(groups[1:], 100)
        self.assertEqual((sum(g["minutes"] for g in placed), more, review), (26, [], 74))

    def test_pack_never_leaves_a_tiny_piece(self):
        items = [{"title": "A", "minutes": 45, "chunk_ids": []}, {"title": "B", "minutes": 13, "chunk_ids": []},
                 {"title": "C", "minutes": 20, "chunk_ids": []}]
        sessions = study_plan.pack(items, [54, 53])
        self.assertEqual([[(i["title"], i["minutes"]) for i in s] for s in sessions], [[("A", 54)], [("B", 13), ("C", 20)]])


class Structure(unittest.TestCase):
    """The deterministic checks of the hashed set's scoring, on the real knowledge base (no model)."""

    @classmethod
    def setUpClass(cls):
        cls.pipeline = RagPipeline()
        cls.chunks = cls.pipeline.retriever.chunks
        cls.titles = {study_plan.section_title(c["section"])[0] for c in cls.chunks.values()}

    def build(self, text):
        request = study_plan.detect(text)
        chunks = study_plan.gather(request.topic, self.pipeline.plan_search)
        return request, chunks, study_plan.build(request, chunks, lambda cid: self.chunks[cid]["difficulty"])

    def test_in_scope_plans_meet_every_deterministic_check(self):
        for case in CASES["in_scope"] + CASES["malay"]:
            request, chunks, plan = self.build(case["text"])
            self.assertTrue(study_plan.in_scope(chunks), case["id"])
            self.assertLessEqual(abs(study_plan.total_minutes(plan) - case["budget_minutes"]), 0.1 * case["budget_minutes"], case["id"])
            retrieved = {c.chunk_id for c in chunks}
            last = -1
            for s in plan["sessions"]:
                items = [i for i in s["items"] if not i.get("review")]
                for item in items:
                    self.assertIn(item["title"], self.titles, case["id"])
                    for cid in item["chunk_ids"]:
                        self.assertIn(cid, retrieved, case["id"])
                        self.assertEqual(study_plan.section_title(self.chunks[cid]["section"])[0].lower().split(),
                                         item["title"].lower().split(), case["id"])
                if items:
                    level = max(study_plan._DIFFICULTY[i["difficulty"]] for i in items)
                    self.assertGreaterEqual(level, last, case["id"])
                    last = level
            self.assertTrue(plan["sessions"][-1]["checkpoint"], case["id"])

    def test_off_topic_is_declined(self):
        for case in CASES["off_topic"]:
            result = self.pipeline.answer(case["text"], model="deepseek-flash")
            self.assertEqual((result.intent, result.plan_call), ("plan_declined", None), case["id"])
            self.assertIn("could not find enough", "".join(result.answer_stream))

    def test_days_mode_and_checkpoints(self):
        _, _, plan = self.build("plan 3 days for SQL")
        self.assertEqual([s["day"] for s in plan["sessions"]], [1, 1, 2, 2, 3, 3])
        self.assertEqual([s["n"] for s in plan["sessions"] if s["checkpoint"]], [2, 4, 6])
        self.assertEqual([s["break_after"] for s in plan["sessions"]], [10, 0, 10, 0, 10, 0])


def fake_writer(questions_for):
    """A fake cloud call returning goals and questions; questions_for(session) gives each session's questions."""
    def complete(model, messages, max_tokens, json_output=False, on_status=None):
        numbers = [int(line.split(":")[0].split()[1]) for line in messages[1]["content"].splitlines() if line.startswith("Session ")]
        body = {"sessions": [{"session": n, "goal": f"Learn session {n} [S04-0001].", "questions": questions_for(n)} for n in numbers]}
        return llm.CloudAnswer(json.dumps(body), 1500, 400, 0, 2.0, 0, 0, 0.0009)
    return complete


class Writing(unittest.TestCase):
    """write_session_text: the grounding check, the template replacement and the failure fallback, on a small fixed plan."""

    def plan(self):
        request = study_plan.PlanRequest(topic="star schema", minutes=60)
        chunks = [chunk("S04-0001", "8.5 Data Warehousing"), chunk("S04-0002", "8.5 Data Warehousing"),
                  chunk("S06-0003", "03-data-warehouse > Star schema", 0.75, "S06")]
        return study_plan.build(request, chunks, lambda cid: "beginner")

    def run_write(self, complete, grounded):
        search = lambda q: [chunk("S04-0001", "x", 0.8)] if grounded(q) else [chunk("S09-9999", "x", 0.9)]
        with mock.patch.object(llm, "cloud_complete", complete):
            return study_plan.write_session_text(self.plan(), "deepseek-flash", search, lambda cid: "chunk text")

    def test_grounded_questions_kept_others_replaced_or_dropped(self):
        plan = self.run_write(fake_writer(lambda n: ["What is a star schema?", "Tell me something unrelated"]),
                              grounded=lambda q: "star schema" in q or q.startswith("What is 8.5"))
        session = plan["sessions"][0]
        self.assertEqual(session["questions"][0], "What is a star schema?")
        self.assertEqual(session["goal"], "Learn session 1.")
        self.assertEqual(plan["question_stats"]["generated"], 2)
        self.assertEqual(plan["question_stats"]["grounded_before_replacement"], 1)
        self.assertEqual(plan["question_stats"]["replaced"] + plan["question_stats"]["dropped"], 1)

    def test_a_failed_call_still_gives_a_plan_with_template_questions(self):
        def broken(*args, **kwargs):
            raise llm.CloudError("DeepSeek is overloaded")
        plan = self.run_write(broken, grounded=lambda q: True)
        self.assertTrue(plan["write_errors"])
        self.assertTrue(plan["sessions"][0]["goal"].startswith("Learn "))
        self.assertTrue(all(q.startswith("What is ") for s in plan["sessions"] for q in s["questions"]))

    def test_parser_accepts_one_session_object_lists_and_session_labels(self):
        """Round 1 of the evaluation: a one-session plan came back as a single object and got only template questions."""
        self.assertEqual(list(study_plan._parse('{"session": 1, "goal": "g", "questions": ["q"]}', [1])), [1])
        rows = study_plan._parse('[{"session": "Session 2", "goal": "a"}, {"goal": "b"}]', [2, 3])
        self.assertEqual((rows[2]["goal"], rows[3]["goal"]), ("a", "b"))
        self.assertEqual(study_plan._parse("not json", [1]), {})
        self.assertEqual(study_plan._plain_title("8.5 Data Warehousing"), "Data Warehousing")
        self.assertEqual(study_plan._plain_title("Chapter 15 SQL Structured Query Language"), "SQL Structured Query Language")

    def test_an_unusable_reply_is_asked_for_once_more(self):
        """Round 2: DeepSeek's JSON mode sometimes closed a list with "}" (invalid JSON); the call is repeated once."""
        good = fake_writer(lambda n: ["What is a star schema?"])
        replies = iter(["Sorry, I can only reply in prose.", None])

        def flaky(model, messages, max_tokens, json_output=False, on_status=None):
            text = next(replies)
            return good(model, messages, max_tokens) if text is None else llm.CloudAnswer(text, 1, 1, 0, 1.0, 0, 0, 0.0)
        plan = self.run_write(flaky, grounded=lambda q: True)
        self.assertEqual(plan["unusable_reply_retries"], 1)
        self.assertEqual(plan["question_stats"]["generated"], 1)
        self.assertTrue(plan["write_errors"][0].startswith("unusable reply"))

    def test_a_stray_closing_brace_is_repaired(self):
        """Rounds 2-3: the same fault twice in a row for some prompts, ..."q3"}]}]} for ..."q3"]}]}; the parser repairs it."""
        text = '{"sessions":[{"session":1,"goal":"g","questions":["q1","q2","q3"}]}]}'
        self.assertEqual(study_plan._parse(text, [1])[1]["questions"], ["q1", "q2", "q3"])
        self.assertEqual(study_plan._repair_brackets('{"q":"a } and ] in a string"}'), '{"q":"a } and ] in a string"}')

    def test_markdown_export(self):
        plan = self.run_write(fake_writer(lambda n: ["What is a star schema?"]), grounded=lambda q: True)
        text = study_plan.to_markdown(plan)
        self.assertIn("# Study plan: star schema", text)
        self.assertIn("What is a star schema?", text)

    def test_runners_never_import_the_planner(self):
        for path in (EVAL / "notebook04").glob("*.py"):
            self.assertNotIn("study_plan", path.read_text(encoding="utf-8"), path.name)


def fake_quiz(topic, chunks, model, search, on_status=None):
    ids = [c.chunk_id for c in chunks]
    item = {"type": "mcq", "question": "What is at the centre of a star schema?", "options": {"A": "Fact table", "B": "Queue",
            "C": "Lake", "D": "Server"}, "correct": "A", "explanation": f"Fact table [{ids[0]}].", "chunk_ids": ids[:1]}
    return {"topic": topic, "model": model, "chunk_ids": ids, "rounds": [], "seconds": 0.1, "usage": None,
            "items": [item, {**item, "question": "What does a fact table hold?"}]}


class Interface(unittest.TestCase):
    """The chat page and the Study plan page with every model call faked."""

    def setUp(self):
        for p in (mock.patch.object(llm, "cloud_complete", fake_writer(lambda n: ["What is a star schema?"])),
                  mock.patch.object(quiz, "make_quiz", fake_quiz), mock.patch.object(llm, "_api_key", lambda provider: "fake"),
                  mock.patch.object(llm, "cloud_generate", lambda *a, **k: SimpleNamespace(
                      text="A star schema has a fact table.", input_tokens=1, output_tokens=1, reasoning_tokens=0, cost_usd_list=0.0,
                      latency_seconds=1.0, retries=0, retry_wait_seconds=0)), no_classifier()):
            p.start()
            self.addCleanup(p.stop)

    def test_chat_plan_card_question_and_checkpoint(self):
        at = AppTest.from_file(APP, default_timeout=180).run()
        at.chat_input[0].set_value("I have 3 hours to study data warehousing").run()
        self.assertFalse(at.exception, at.exception)
        headings = [m.value for m in at.markdown if m.value.startswith("#### Session")]
        self.assertEqual(len(headings), 3)
        self.assertEqual(at.session_state["conversation"].exchanges, [], "a plan is not added to the conversation history")
        next(b for b in at.button if b.label == "Quiz checkpoint").click().run()
        self.assertFalse(at.exception, at.exception)
        self.assertEqual(len(at.radio), 2)
        next(b for b in at.button if b.label == "What is a star schema?").click().run()
        self.assertFalse(at.exception, at.exception)
        self.assertTrue(any(m.value == "What is a star schema?" for m in at.markdown), "the suggested question was sent to the chat")

    def test_plan_page(self):
        at = AppTest.from_file(PLAN_PAGE, default_timeout=180).run()
        self.assertFalse(at.exception, at.exception)
        at.text_input[0].input("data quality")
        at.number_input[0].set_value(45.0)
        at.selectbox[0].set_value("minutes")
        next(b for b in at.button if b.label == "Make a plan").click().run()
        self.assertFalse(at.exception, at.exception)
        self.assertEqual(len([m for m in at.markdown if m.value.startswith("#### Session")]), 1)
        at.text_input[0].input("cooking pasta")
        next(b for b in at.button if b.label == "Make a plan").click().run()
        self.assertTrue(any("could not find enough" in i.value for i in at.info))


if __name__ == "__main__":
    unittest.main(verbosity=1)
