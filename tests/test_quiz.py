"""
Offline tests for "Quiz me" (feature B; docs/feature_design_2026-11-28.md). NOTHING HERE CALLS A MODEL: the quiz writer, the cloud
and the Ollama calls are faked, and the intent classifier is replaced, so no network is used. The routing cases come from the hashed
test sets written before the code (evaluation/quiz_cases.json and quiz_cases_holdout.json), whose hashes are checked first.

Run from the project root:  venv\\Scripts\\python tests\\test_quiz.py
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
from src.core import intent, llm, quiz  # noqa: E402
from src.core.conversation import Conversation  # noqa: E402
from src.core.rag import RagPipeline  # noqa: E402

EVAL = ROOT / "evaluation"
CASES = json.loads((EVAL / "quiz_cases.json").read_text(encoding="utf-8"))
HOLDOUT = json.loads((EVAL / "quiz_cases_holdout.json").read_text(encoding="utf-8"))
APP = str(ROOT / "src" / "app" / "app.py")
QUIZ_PAGE = str(ROOT / "src" / "app" / "pages" / "2_Quiz.py")


def chunk(chunk_id, text, score=0.85):
    return SimpleNamespace(chunk_id=chunk_id, text=text, score=score, source_name="Src", section="Sec", page=None, file="f.md",
                           source_type="course_notes", topic="Data Warehousing", source_id=chunk_id[:3])


CHUNKS = [chunk("S04-0210", "A star schema has a central fact table joined to dimension tables. Fact tables hold measures."),
          chunk("S05-0138", "OLAP supports complex analytical queries over historical data in a data warehouse.")]
RETRIEVED = {c.chunk_id: c for c in CHUNKS}


def found(*ids, score=0.8):
    """A fake search that finds these chunk IDs."""
    return lambda question: [chunk(i, "", score) for i in ids]


MCQ = {"type": "mcq", "question": "What sits at the centre of a star schema?",
       "options": {"A": "A fact table", "B": "A message queue", "C": "A data lake", "D": "A web server"}, "correct": "A",
       "explanation": "The centre is a fact table [S04-0210].", "chunk_ids": ["S04-0210"]}
MCQ2 = {**MCQ, "question": "What do fact tables in a star schema hold?",
        "options": {"A": "Measures", "B": "Passwords", "C": "Images", "D": "Emails"}, "correct": "A",
        "explanation": "Fact tables hold measures [S04-0210]."}
SHORT = {"type": "short", "question": "What kind of queries does OLAP support?",
         "answer": "Complex analytical queries over historical data.", "key_points": ["analytical queries", "historical data"],
         "explanation": "OLAP supports analytical queries [S05-0138].", "chunk_ids": ["S05-0138"]}


def reply(*items):
    return json.dumps({"questions": list(items)})


def no_classifier():
    """Routing tests never call the intent classifier (it would reach Ollama)."""
    return mock.patch.object(intent, "classify", lambda message: "REAL_QUESTION")


class HashedSets(unittest.TestCase):
    def test_sets_are_the_ones_hashed_before_the_code(self):
        for name in ("quiz_cases", "quiz_cases_holdout"):
            recorded = (EVAL / f"{name}.sha256").read_text(encoding="utf-8").split()[0]
            self.assertEqual(hashlib.sha256((EVAL / f"{name}.json").read_bytes()).hexdigest(), recorded, name)


class Routing(unittest.TestCase):
    """The router alone (no retriever): RagPipeline.search_text gives the same cleaned text the app uses."""

    @classmethod
    def setUpClass(cls):
        cls.pipeline = RagPipeline(retriever=SimpleNamespace())

    def route(self, text):
        with no_classifier():
            return intent.route(text, top_score=lambda m: 0.0, clean=self.pipeline.search_text)

    def test_in_scope_cases_route_to_the_quiz_with_a_topic(self):
        for case in CASES["in_scope"] + HOLDOUT["in_scope"]:
            decision = self.route(case["text"])
            self.assertEqual((decision.kind, decision.path), ("quiz", "quiz"), case["id"])
            self.assertTrue(decision.quiz_topic, case["id"])

    def test_off_topic_and_near_domain_route_to_the_quiz_then_retrieval_decides(self):
        for case in CASES["off_topic"] + CASES["near_domain"] + HOLDOUT["off_topic"]:
            self.assertEqual(self.route(case["text"]).kind, "quiz", case["id"])

    def test_assessed_tasks_never_become_a_quiz(self):
        """The quiz's own duty: a task request is never turned into a quiz. Whether the router then uses coach mode is the
        guardrail router's behaviour, reported by evaluation/quiz_eval.py against the case file's "coach" expectation (QT02, "the
        complete answers to my homework", is missed by the router because its deliverable list has "answer" but not "answers")."""
        for case in CASES["task"]:
            self.assertNotEqual(self.route(case["text"]).kind, "quiz", case["id"])
        self.assertEqual(self.route(CASES["task"][0]["text"]).path, "task_request")

    def test_not_quiz_cases_do_not_route_to_the_quiz(self):
        for case in CASES["not_quiz"] + HOLDOUT["not_quiz"]:
            self.assertNotEqual(self.route(case["text"]).kind, "quiz", case["id"])

    def test_no_evaluation_question_probe_or_router_case_reaches_the_quiz(self):
        texts = [q["question"] for q in json.loads((EVAL / "test_questions.json").read_text(encoding="utf-8"))["questions"]]
        texts += [p["request"] for p in json.loads((EVAL / "notebook04" / "guardrail_probes.json").read_text(encoding="utf-8"))["probes"]]
        router = json.loads((EVAL / "guardrail_router_cases.json").read_text(encoding="utf-8"))
        texts += [c["text"] for c in router["recall"] + router["near_miss"]]
        self.assertEqual(len(texts), 100 + 10 + len(router["recall"]) + len(router["near_miss"]))
        caught = [t for t in texts if self.route(t).kind == "quiz"]
        self.assertEqual(caught, [])

    def test_quiz_me_alone_names_no_topic(self):
        self.assertEqual(self.route("quiz me").quiz_topic, "")
        self.assertEqual(quiz.detect("quiz me on star schema please?"), "star schema")


class Checks(unittest.TestCase):
    def test_good_items_pass(self):
        self.assertIsNone(quiz.check_item(MCQ, RETRIEVED, found("S04-0210")))
        self.assertIsNone(quiz.check_item(SHORT, RETRIEVED, found("S05-0138")))

    def test_each_failure_has_its_reason(self):
        cases = [
            ({**MCQ, "chunk_ids": ["S06-0322"]}, "outside the retrieved set"),
            ({**MCQ, "options": {**MCQ["options"], "B": "A fact table"}}, "four distinct"),
            ({**MCQ, "options": {"A": "x", "B": "y", "C": "z"}}, "A, B, C and D"),
            ({**MCQ, "correct": "E"}, "correct letter"),
            ({**MCQ, "options": {**MCQ["options"], "A": "A quantum router"}}, "not supported"),
            ({**SHORT, "key_points": []}, "key points"),
            ({**SHORT, "answer": ""}, "no answer"),
            ({**MCQ, "chunk_ids": []}, "no chunk IDs"),
            ({**MCQ, "type": "essay"}, "unknown type"),
            ("not a dict", "not an object"),
        ]
        for item, reason in cases:
            self.assertIn(reason, quiz.check_item(item, RETRIEVED, found("S04-0210", "S05-0138")) or "", item)

    def test_questions_about_the_excerpts_and_negative_questions_are_rejected(self):
        """v2, after round 1 on the design cases: a student never sees the excerpts, and a NOT question cannot be supported."""
        for question in ("According to the excerpts, what is a fact table?", "What does the text say about star schemas?",
                         "What is one benefit mentioned in the recap?"):
            self.assertIn("refers to the excerpts", quiz.check_item({**MCQ, "question": question}, RETRIEVED, found("S04-0210")))
        for question in ("Which of the following is NOT part of a star schema?", "Which one is not a property of a fact table?"):
            self.assertEqual(quiz.check_item({**MCQ, "question": question}, RETRIEVED, found("S04-0210")), "negative question")
        ordinary = {**MCQ, "question": "Why is a fact table not stored as a dimension in a star schema?"}
        self.assertIsNone(quiz.check_item(ordinary, RETRIEVED, found("S04-0210")))

    def test_question_must_retrieve_its_cited_chunk(self):
        self.assertIn("retrieve", quiz.check_item(MCQ, RETRIEVED, found("S05-0138")))
        self.assertIn("retrieve", quiz.check_item(MCQ, RETRIEVED, found("S04-0210", score=0.65)))

    def test_support_share(self):
        self.assertEqual(quiz.support_share("fact tables", "a fact table"), 1.0)
        self.assertEqual(quiz.support_share("the of and", "anything"), 0.0)

    def test_parse_handles_fences_and_extra_text(self):
        self.assertEqual(len(quiz.parse("```json\n" + reply(MCQ) + "\n```")), 1)
        self.assertEqual(len(quiz.parse("Here you go: " + reply(MCQ, SHORT) + " Enjoy!")), 2)
        self.assertIsNone(quiz.parse("no json here"))

    def test_mark(self):
        made = {"items": [quiz.clean_item(MCQ), quiz.clean_item(MCQ2), quiz.clean_item(SHORT)]}
        self.assertEqual(quiz.mark(made, {0: "A", 1: "B", 2: "anything"}), {0: True, 1: False})


class Making(unittest.TestCase):
    """make_quiz with the model call faked: the first try, one retry for what is missing, and the minimum."""

    def run_make(self, *replies, model="deepseek-flash"):
        sent = []
        queue = list(replies)

        def fake_generate(messages, model_name, on_status):
            sent.append(messages)
            return queue.pop(0), None
        with mock.patch.object(quiz, "_generate", fake_generate):
            try:
                result = quiz.make_quiz("star schema", CHUNKS, model, found("S04-0210", "S05-0138"))
            except quiz.QuizError as error:
                result = error
        return result, sent

    def test_three_good_items_in_one_call(self):
        made, sent = self.run_make(reply(MCQ, MCQ2, SHORT))
        self.assertEqual([i["type"] for i in made["items"]], ["mcq", "mcq", "short"])
        self.assertEqual(len(sent), 1)
        self.assertEqual(made["chunk_ids"], ["S04-0210", "S05-0138"])

    def test_one_retry_asks_only_for_what_is_missing(self):
        bad_short = {**SHORT, "chunk_ids": ["S06-0322"]}
        made, sent = self.run_make(reply(MCQ, MCQ2, bad_short), reply(SHORT))
        self.assertEqual(len(made["items"]), 3)
        retry = sent[1][1]["content"]
        self.assertIn("0 multiple-choice question(s) and 1 short-answer", retry)
        self.assertIn(MCQ["question"], retry)
        self.assertEqual([r["attempt"] for r in made["rounds"]], [1, 2])
        self.assertFalse(made["rounds"][0]["items"][2]["passed"])

    def test_two_items_are_enough_but_one_is_not(self):
        made, _ = self.run_make(reply(MCQ, MCQ2, {**SHORT, "answer": ""}), reply({**SHORT, "answer": ""}))
        self.assertEqual(len(made["items"]), 2)
        error, _ = self.run_make(reply(MCQ), reply({**MCQ, "chunk_ids": ["S06-0322"]}))
        self.assertIsInstance(error, quiz.QuizError)
        self.assertEqual(str(error), settings.QUIZ_FAILED_MESSAGE)
        self.assertEqual(len(error.rounds), 2)

    def test_unreadable_json_is_retried_and_local_failure_says_offline(self):
        made, _ = self.run_make("garbage", reply(MCQ, MCQ2, SHORT))
        self.assertEqual(len(made["items"]), 3)
        self.assertFalse(made["rounds"][0]["parsed"])
        error, _ = self.run_make("garbage", "garbage", model=settings.DEV_LLM_MODEL)
        self.assertEqual(str(error), settings.QUIZ_FAILED_LOCAL_MESSAGE)

    def test_a_repeated_question_is_not_kept_twice(self):
        made, _ = self.run_make(reply(MCQ, MCQ, SHORT), reply(MCQ2))
        self.assertEqual([i["question"] for i in made["items"]], [MCQ["question"], MCQ2["question"], SHORT["question"]])

    def test_prompt_is_the_quiz_prompt_with_the_chat_excerpts(self):
        messages = quiz.build_messages("star schema", CHUNKS, 2, 1)
        self.assertEqual(messages[0]["content"], settings.QUIZ_SYSTEM_PROMPT)
        self.assertNotEqual(messages[0]["content"], settings.SYSTEM_PROMPT)
        self.assertIn("[S04-0210] (source: Src; section: Sec)", messages[1]["content"])
        self.assertIn("2 multiple-choice question(s) and 1 short-answer", messages[1]["content"])
        self.assertNotIn("<", settings.QUIZ_FORMAT_INSTRUCTION)


class ModelCalls(unittest.TestCase):
    def test_cloud_json_requests(self):
        messages = quiz.build_messages("star schema", CHUNKS, 2, 1)
        deepseek = llm._cloud_request("deepseek-flash", messages, max_tokens=settings.QUIZ_MAX_OUTPUT_TOKENS, json_output=True)
        self.assertEqual(deepseek["response_format"], {"type": "json_object"})
        self.assertEqual(deepseek["max_tokens"], settings.QUIZ_MAX_OUTPUT_TOKENS)
        self.assertEqual(deepseek["extra_body"], {"thinking": {"type": "disabled"}})
        gemini = llm._cloud_request("gemini-3.8-flash", messages, json_output=True)
        self.assertEqual(gemini["config"]["response_mime_type"], "application/json")
        chat = llm._cloud_request("deepseek-flash", messages)
        self.assertNotIn("response_format", chat)
        self.assertEqual(chat["max_tokens"], settings.DEV_LLM_MAX_ANSWER_TOKENS)

    def test_local_json_body(self):
        posted = {}

        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                return {"message": {"content": reply(MCQ)}}
        with mock.patch.object(llm.requests, "post", lambda url, json, timeout: posted.update(json) or Response()):
            text = llm.local_complete_json([{"role": "user", "content": "x"}], "qwen3.5:2b", 1500, settings.QUIZ_JSON_SCHEMA)
        self.assertEqual(text, reply(MCQ))
        self.assertEqual((posted["format"], posted["think"], posted["options"]["num_predict"]),
                         (settings.QUIZ_JSON_SCHEMA, False, 1500))

    def test_cloud_complete_uses_json_mode_and_no_answer_guards(self):
        captured = {}

        def fake_send(model, key, request):
            captured.update(request)
            return llm._Reply(reply(MCQ), 900, 400, 0, "STOP")
        with mock.patch.object(llm, "_api_key", lambda provider: "fake"), mock.patch.object(llm, "_send_openai", fake_send):
            answer = llm.cloud_complete("deepseek-flash", [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
                                        1500, json_output=True)
        self.assertEqual(answer.text, reply(MCQ))
        self.assertEqual(captured["response_format"], {"type": "json_object"})

    def test_runners_never_import_the_quiz(self):
        for path in (EVAL / "notebook04").glob("*.py"):
            self.assertNotIn("quiz", path.read_text(encoding="utf-8").lower(), path.name)


def fake_make_quiz(topic, chunks, model, search, on_status=None):
    """A fixed quiz citing the real retrieved chunks (the UI tests never call a model)."""
    ids = [c.chunk_id for c in chunks]
    return {"topic": topic, "model": model, "chunk_ids": ids, "rounds": [], "seconds": 0.1,
            "usage": {"input_tokens": 1000, "output_tokens": 300, "reasoning_tokens": 0, "cost_usd_list": 0.00066,
                      "latency_seconds": 2.0, "retries": 0, "retry_wait_seconds": 0},
            "items": [{**quiz.clean_item(MCQ), "chunk_ids": ids[:1], "explanation": f"A fact table [{ids[0]}]."},
                      {**quiz.clean_item(MCQ2), "chunk_ids": ids[:1], "explanation": f"Measures [{ids[0]}]."},
                      {**quiz.clean_item(SHORT), "chunk_ids": ids[1:2], "explanation": f"Analytical [{ids[1]}]."}]}


class Pipeline(unittest.TestCase):
    """The real retriever (loads the embedding model once); the quiz writer is faked."""

    @classmethod
    def setUpClass(cls):
        cls.pipeline = RagPipeline()

    def ask(self, text, conversation=None):
        with no_classifier():
            return self.pipeline.answer(text, model="deepseek-flash", conversation=conversation)

    def test_off_topic_gets_the_fallback_and_no_quiz(self):
        for case in CASES["off_topic"] + HOLDOUT["off_topic"]:
            result = self.ask(case["text"])
            self.assertEqual((result.intent, result.used_fallback, result.quiz_call), ("quiz", True, None), case["id"])
            self.assertEqual("".join(result.answer_stream), settings.FALLBACK_MESSAGE)

    def test_an_in_scope_request_gets_a_quiz_call_on_the_retrieved_chunks(self):
        result = self.ask("quiz me on star schema")
        self.assertIsNotNone(result.quiz_call)
        self.assertEqual(len(result.chunks), settings.TOP_K)
        self.assertGreaterEqual(result.chunks[0].score, settings.SIMILARITY_THRESHOLD)
        with mock.patch.object(quiz, "make_quiz", fake_make_quiz):
            made = result.quiz_call()
        self.assertEqual(made["topic"], "star schema")

    def test_follow_up_quizzes_the_earlier_topic_and_no_topic_asks(self):
        convo = Conversation()
        with mock.patch.object(llm, "cloud_generate", lambda *a, **k: SimpleNamespace(text="A star schema [S04-0210].")):
            first = self.ask(CASES["follow_up"][0]["turns"][0], convo)
            convo.record(first, first.cloud_call().text)
        result = self.ask(CASES["follow_up"][0]["turns"][1], convo)
        self.assertEqual(result.intent, "quiz")
        self.assertIn("star schema", result.quiz_topic)
        ask = self.ask(CASES["follow_up"][1]["turns"][0], Conversation())
        self.assertEqual((ask.intent, "".join(ask.answer_stream)), ("quiz_ask", settings.QUIZ_ASK_TOPIC))

    def test_quiz_for_topic_is_the_same_path(self):
        result = self.pipeline.quiz_for_topic("star schema", model="deepseek-flash")
        self.assertIsNotNone(result.quiz_call)
        self.assertEqual(self.ask("quiz me on star schema").search_query, result.search_query)


class Interface(unittest.TestCase):
    """The real chat page and the Quiz page, with the quiz writer faked and the cloud key faked."""

    def setUp(self):
        for p in (mock.patch.object(quiz, "make_quiz", fake_make_quiz), mock.patch.object(llm, "_api_key", lambda provider: "fake"),
                  mock.patch.object(llm, "_send_openai", side_effect=AssertionError("no model call")), no_classifier()):
            p.start()
            self.addCleanup(p.stop)

    def check_button(self, at):
        return next(b for b in at.button if b.label == settings.QUIZ_CHECK_LABEL)

    def test_chat_quiz_card_answer_and_reveal(self):
        at = AppTest.from_file(APP, default_timeout=180).run()
        at.chat_input[0].set_value("quiz me on star schema").run()
        self.assertFalse(at.exception, at.exception)
        self.assertEqual(len(at.radio), 2)
        self.assertEqual(len(at.text_area), 1)
        self.assertEqual(at.session_state["conversation"].exchanges, [], "a quiz is not added to the conversation history")
        at.radio[0].set_value("A")
        at.radio[1].set_value("B")
        at.text_area[0].input("measures over time")
        self.check_button(at).click().run()
        self.assertFalse(at.exception, at.exception)
        self.assertTrue(any(s.value.startswith("Correct: A.") for s in at.success))
        self.assertTrue(any("Correct answer: A." in e.value for e in at.error))
        self.assertTrue(any("Model answer:" in i.value for i in at.info))
        self.assertTrue(any("Multiple choice: 1 of 2 correct." in m.value for m in at.markdown))
        next(b for b in at.button if b.label == settings.QUIZ_SELF_MARK_LABELS[0]).click().run()
        self.assertTrue(any("You marked this: I had this" in c.value for c in at.caption))
        self.assertTrue(any(m.value == "**Sources**" for m in at.markdown))

    def test_quiz_page(self):
        at = AppTest.from_file(QUIZ_PAGE, default_timeout=180).run()
        self.assertFalse(at.exception, at.exception)
        at.text_input[0].input("star schema")
        next(b for b in at.button if b.label == "Make a quiz").click().run()
        self.assertFalse(at.exception, at.exception)
        self.assertEqual(len(at.radio), 2)
        at.text_input[0].input("the French revolution")
        next(b for b in at.button if b.label == "Make a quiz").click().run()
        self.assertTrue(any(settings.FALLBACK_MESSAGE in i.value for i in at.info))


if __name__ == "__main__":
    unittest.main(verbosity=1)
