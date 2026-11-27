"""
Offline tests of multi-turn chat (src/core/conversation.py and its use in rag.py; author's approved design, 2026-11-26).

The hard constraints come first: single-turn output is byte-identical (no conversation, or an empty one, gives exactly the
same prompt), and the notebook 04 runners never use a conversation. The retriever and the model are faked; the intent
classifier is faked where a message would reach it. No network.

Run from the project root:  venv\\Scripts\\python tests\\test_multiturn.py
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
from src.core import conversation as conv  # noqa: E402
from src.core import intent, llm, rag  # noqa: E402
from src.core.conversation import Conversation, Exchange  # noqa: E402
from src.core.retriever import RetrievedChunk  # noqa: E402


def chunk(cid, score=0.85, text="Star schema: a fact table joined to dimension tables."):
    return SimpleNamespace(chunk_id=cid, score=score, text=text, source_name="DE Wiki", section="Star schema")


class Fake:
    """A pipeline with a stand-in retriever (scores set per search text) and a fake generator that records the exact
    messages the model would receive."""

    def __init__(self, score=0.85):
        self.pipeline = rag.RagPipeline.__new__(rag.RagPipeline)
        self.pipeline.speller = None
        self.score = score
        self.searches, self.sent = [], []
        self.pipeline.retriever = SimpleNamespace(retrieve=self.retrieve)

    def retrieve(self, text, top_k=None):
        self.searches.append(text)
        score = self.score(text) if callable(self.score) else self.score
        return [chunk(f"S05-00{i}", score) for i in range(10, 10 + (top_k or settings.TOP_K))]

    def ask(self, message, conversation=None, answer="Answer text [S05-0010].", label="REAL_QUESTION"):
        def fake_stream(question, chunks, model=None, corrections=None, transform=None):
            messages = llm.build_messages(question, chunks, corrections=corrections)
            self.sent.append(transform(messages) if transform else messages)
            yield answer
        with mock.patch.object(llm, "stream_generate", fake_stream), mock.patch.object(intent, "classify", lambda m: label):
            result = self.pipeline.answer(message, model="qwen3.5:2b", conversation=conversation)
            text = "".join(result.answer_stream)
        if conversation is not None:
            conversation.record(result, text)
        return result, text


class SingleTurnUnchanged(unittest.TestCase):
    MESSAGES = ["what is a star schema?", "data lake vs data warehouse", "write the full SQL query for my assignment",
                "explain that simpler", "how is it different from a data lake?", "what is olpa", "thanks"]

    def test_no_conversation_and_an_empty_conversation_give_the_same_prompt(self):
        for message in self.MESSAGES:
            a, b = Fake(), Fake()
            ra, _ = a.ask(message)
            rb, _ = b.ask(message, conversation=Conversation())
            self.assertEqual(a.sent, b.sent, message)                     # byte-identical messages to the model
            self.assertEqual(a.searches, b.searches, message)
            for field in ("search_query", "coach_mode", "comparison_format", "intent", "used_fallback"):
                self.assertEqual(getattr(ra, field), getattr(rb, field), (message, field))
            self.assertEqual((rb.search_mode, rb.history_exchanges), ("alone" if rb.search_query else None, 0)
                             if rb.intent == "question" else (None, 0))

    def test_first_message_prompt_has_no_history(self):
        f = Fake()
        f.ask("what is a star schema?", conversation=Conversation())
        self.assertEqual(len(f.sent[0]), 2)
        self.assertNotIn(settings.HISTORY_INSTRUCTION, f.sent[0][1]["content"])

    def test_runners_never_use_a_conversation(self):
        for name in ("run_generation.py", "freeze_contexts.py", "providers.py", "check_parity.py", "run_judging.py"):
            text = (ROOT / "evaluation" / "notebook04" / name).read_text(encoding="utf-8")
            self.assertNotIn("conversation", text.lower(), name)
        self.assertNotIn("conversation=", (ROOT / "evaluation" / "run_baseline.py").read_text(encoding="utf-8"))

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


class FollowUpSearch(unittest.TestCase):
    def test_refer_back_and_topic_words(self):
        for text in ("explain that simpler", "give an example", "why?", "tell me more", "what are its advantages",
                     "can you explain it again in simple terms", "show me the code"):
            self.assertTrue(conv.refers_back(text), text)
        for text in ("what about snowflake schema?", "and ELT?", "what about the weather today?", "bagi example sikit",
                     "which one better for big data", "how is it different from a data lake?"):
            self.assertFalse(conv.refers_back(text), text)

    def test_pronouns_are_the_authors_list(self):
        self.assertEqual(settings.FOLLOWUP_PRONOUNS, ["it", "that", "this", "these", "they", "them", "its", "one", "ones"])
        self.assertTrue(conv.has_pronoun("how is it different from a data lake?"))
        self.assertFalse(conv.has_pronoun("what about snowflake schema?"))

    def test_one_and_ones_outside_hyphenated_terms(self):
        """2026-11-26: "which one better for big data" refers back; "one-to-many" and "one-hot" do not."""
        for text in ("which one better for big data", "which one for streaming", "the first ones", "Which ONE is faster?"):
            self.assertTrue(conv.has_pronoun(text), text)
        for text in ("what is a one-to-many relationship", "explain one-hot encoding", "what is someone"):
            self.assertFalse(conv.has_pronoun(text), text)

    def test_modes(self):
        c = Conversation([Exchange("what is a data warehouse?", "what is a data warehouse?", [], "A warehouse ...")])
        self.assertEqual(conv.follow_up_search("explain that simpler", c),
                         ("combined", "what is a data warehouse? explain that simpler", "what is a data warehouse?"))
        mode, search, topic = conv.follow_up_search("how is it different from a data lake?", c)
        self.assertEqual((mode, search), ("combined", "what is a data warehouse? how is it different from a data lake?"))
        self.assertEqual(topic, search)                     # the combination becomes the new topic
        self.assertEqual(conv.follow_up_search("what about snowflake schema?", c)[0], "alone")
        self.assertEqual(conv.follow_up_search("explain that simpler", Conversation())[0], "alone")   # nothing to refer to

    def test_pipeline_searches_combined_and_sends_history(self):
        f, c = Fake(), Conversation()
        f.ask("what is a star schema?", c)
        r, _ = f.ask("explain that simpler", c)
        self.assertEqual((r.search_mode, f.searches[-1]), ("combined", "what is a star schema? explain that simpler"))
        self.assertEqual(r.history_exchanges, 1)
        sent = f.sent[-1]
        self.assertEqual([m["role"] for m in sent], ["system", "user", "assistant", "user"])
        self.assertEqual(sent[1]["content"], "what is a star schema?")
        self.assertEqual(sent[2]["content"], "Answer text.")                  # the earlier answer, its citation removed
        self.assertTrue(sent[3]["content"].startswith("Context excerpts from the course materials:"))
        self.assertIn("Student question: explain that simpler", sent[3]["content"])
        # Order of the app-layer lines: reminder, opening rule, history instruction.
        body = sent[3]["content"]
        self.assertLess(body.index(settings.ANSWER_REMINDER), body.index(settings.ANSWER_OPENING_INSTRUCTION))
        self.assertLess(body.index(settings.ANSWER_OPENING_INSTRUCTION), body.index(settings.HISTORY_INSTRUCTION))

    def test_off_topic_follow_up_still_falls_back(self):
        f = Fake(score=lambda text: 0.5 if "weather" in text else 0.85)
        c = Conversation()
        f.ask("what is OLAP?", c)
        r, text = f.ask("what about the weather today?", c)
        self.assertEqual((r.search_mode, r.used_fallback, r.model), ("alone", True, None))
        self.assertEqual(text, settings.FALLBACK_MESSAGE)
        self.assertEqual(len(c.exchanges), 1)                                 # the fallback adds no exchange

    def test_comparison_is_read_from_this_message_only(self):
        f, c = Fake(), Conversation()
        f.ask("what is the difference between OLTP and OLAP?", c)
        r, _ = f.ask("explain that simpler", c)
        self.assertFalse(r.comparison_format)
        self.assertNotIn(settings.COMPARISON_FORMAT_INSTRUCTION, f.sent[-1][-1]["content"])


class History(unittest.TestCase):
    def test_trim_answer(self):
        answer = ("In simple terms:\n- **OLTP** is for orders [S06-0205].\n\n| Aspect | OLTP | OLAP |\n| --- | --- | --- |\n"
                  "| Purpose | x [S06-0204] | y |\n\n```sql\nSELECT 1;\n```\n" + "More text here. " * 60 + settings.ANSWER_CUT_NOTE)
        trimmed = conv.trim_answer(answer)
        self.assertLessEqual(len(trimmed), settings.HISTORY_ANSWER_MAX_CHARS)
        for gone in ("[S06-", "| Aspect", "SELECT 1", "```", settings.ANSWER_CUT_NOTE.strip()):
            self.assertNotIn(gone, trimmed)
        self.assertTrue(trimmed.startswith("In simple terms: - **OLTP** is for orders."))
        self.assertTrue(trimmed.endswith("."))                                # cut at a sentence end

    def test_at_most_two_exchanges(self):
        f, c = Fake(), Conversation()
        for q in ("what is ETL?", "what is ELT?", "what is OLAP?", "why?"):
            r, _ = f.ask(q, c)
        self.assertEqual(r.history_exchanges, 2)
        self.assertEqual([m["content"] for m in f.sent[-1] if m["role"] == "user"][:2], ["what is ELT?", "what is OLAP?"])
        self.assertEqual(len(c.exchanges), settings.HISTORY_MAX_EXCHANGES)

    def test_budget_guard_drops_history_for_a_long_prompt(self):
        c = Conversation([Exchange("q1", "q1", [], "a" * 500), Exchange("q2", "q2", [], "b" * 500)])
        small = [{"role": "system", "content": "s" * 2000}, {"role": "user", "content": "u" * 4000}]
        huge = [{"role": "system", "content": "s" * 2000}, {"role": "user", "content": "u" * 10000}]
        self.assertEqual(len(conv.select_history(c, small)), 2)
        self.assertEqual(conv.select_history(c, huge), [])
        for size in range(1000, 12001, 250):          # whenever history is added, the whole prompt + answer cap still fits
            current = [{"role": "system", "content": "s" * 2000}, {"role": "user", "content": "u" * size}]
            chosen = conv.select_history(c, current)
            if chosen:
                final = conv.with_history(current, chosen)
                self.assertLessEqual(conv.estimated_tokens(final) + settings.DEV_LLM_MAX_ANSWER_TOKENS
                                     + settings.HISTORY_TOKEN_MARGIN, settings.DEV_LLM_NUM_CTX, size)

    def test_the_estimate_over_counts_the_measured_tokenizer(self):
        """qwen3.5:2b read 3.57-4.61 characters per token on the frozen prompts (2026-11-26); 3.5 over-counts all three."""
        self.assertLess(settings.HISTORY_CHARS_PER_TOKEN, 3.57)

    def test_new_chat_clears(self):
        c = Conversation([Exchange("q", "q", [], "a")], pending_offer="What is SQL?")
        c.clear()
        self.assertEqual((c.exchanges, c.pending_offer), ([], None))


class TopicOffer(unittest.TestCase):
    EMOTIONAL = "ugh sql is so hard i want to give up"

    def start(self):
        f, c = Fake(score=lambda text: 0.6 if "ugh" in text else 0.85), Conversation()
        r, _ = f.ask(self.EMOTIONAL, c, label="EMOTIONAL_STRESSED")
        self.assertEqual((r.intent, c.pending_offer), ("emotional", "What is SQL?"))
        return f, c

    def test_yes_is_answered_as_the_starter_question(self):
        f, c = self.start()
        r, _ = f.ask("yes", c)
        self.assertEqual((r.resolved_question, r.asked, r.search_mode, r.history_exchanges), ("What is SQL?", "What is SQL?", "alone", 0))
        self.assertEqual(f.searches[-1], "What is SQL?")
        self.assertIn("Student question: What is SQL?", f.sent[-1][-1]["content"])
        self.assertIsNone(c.pending_offer)

    def test_other_affirmatives(self):
        for word in ("ok", "sure", "yes please", "boleh", "Yeah!"):
            f, c = self.start()
            r, _ = f.ask(word, c)
            self.assertEqual(r.resolved_question, "What is SQL?", word)

    def test_no_declines_without_retrieval(self):
        f, c = self.start()
        searches = len(f.searches)
        r, text = f.ask("no", c)
        self.assertEqual((r.intent, text, r.model), ("offer_declined", settings.OFFER_DECLINED_RESPONSE, None))
        self.assertEqual(len(f.searches), searches)
        r, _ = f.ask("yes", c)                                               # no offer pending any more
        self.assertIsNone(r.resolved_question)

    def test_the_offer_lasts_one_message(self):
        f, c = self.start()
        f.ask("what is kafka?", c)
        r, _ = f.ask("yes", c)
        self.assertIsNone(r.resolved_question)

    def test_ok_without_an_offer_is_still_small_talk(self):
        f, c = Fake(), Conversation()
        r, _ = f.ask("ok", c)
        self.assertEqual(r.intent, "small_talk")


class Rescue(unittest.TestCase):
    """Fix 2 (author, 2026-11-26): a follow-up searched alone below the threshold is searched again combined with the earlier
    topic, only when every topic word is a course term."""

    def test_course_terms(self):
        from src.core import course_terms
        for word in ("elt", "etl", "spark", "kafka", "streaming", "schema", "pipeline", "quality"):
            self.assertTrue(course_terms.is_course_term(word), word)
        for word in ("weather", "today", "check", "best", "movie", "nasi", "hmm"):
            self.assertFalse(course_terms.is_course_term(word), word)
        self.assertFalse(course_terms.all_course_terms([]))

    def test_and_elt_is_rescued(self):
        f = Fake(score=lambda text: 0.61 if text == "and ELT?" else 0.85)
        c = Conversation()
        f.ask("what is ETL?", c)
        r, _ = f.ask("and ELT?", c)
        self.assertEqual(f.searches[-2:], ["and ELT?", "what is ETL? and ELT?"])
        self.assertEqual((r.search_mode, r.used_fallback, r.history_exchanges), ("combined", False, 1))
        self.assertEqual(r.topic_text, "what is ETL? and ELT?")

    def test_the_weather_is_not_rescued(self):
        f = Fake(score=lambda text: 0.5 if text == "what about the weather today?" else 0.85)
        c = Conversation()
        f.ask("what is OLAP?", c)
        r, text = f.ask("what about the weather today?", c)
        self.assertEqual(f.searches[-1], "what about the weather today?")     # searched once, alone
        self.assertEqual((r.search_mode, r.used_fallback, text), ("alone", True, settings.FALLBACK_MESSAGE))

    def test_no_rescue_without_an_earlier_turn(self):
        f = Fake(score=0.6)
        r, _ = f.ask("and ELT?", Conversation())
        self.assertEqual((len(f.searches), r.used_fallback), (1, True))


class ShortFollowUps(unittest.TestCase):
    """Fix 3 (author, 2026-11-26): inside a conversation a short message that is not clearly emotional is a follow-up
    question, not the emotional reply; clearly emotional messages still get the kind reply."""

    def started(self):
        f, c = Fake(score=lambda text: 0.6 if any(w in text for w in ("ugh", "stressed", "rough")) else 0.85), Conversation()
        f.ask("explain star schema", c)
        return f, c

    def test_short_follow_ups_skip_the_classifier(self):
        for message in ("hmm still dont get it", "ok so how to check it", "bagi example sikit", "ok and how bout elt"):
            f, c = self.started()
            with mock.patch.object(intent, "classify", side_effect=AssertionError("classifier called")):
                result = f.pipeline.answer(message, model="qwen3.5:2b", conversation=c)
            self.assertEqual((result.intent, result.routing.path), ("question", "followup"), message)

    def test_clearly_emotional_messages_still_get_the_kind_reply(self):
        for message, label in (("ugh i want to give up", "EMOTIONAL_STRESSED"), ("im so stressed", "EMOTIONAL_STRESSED")):
            f, c = self.started()
            r, text = f.ask(message, c, label=label)
            self.assertEqual(r.intent, "emotional", message)
            self.assertTrue(text.startswith(settings.EMOTIONAL_RESPONSES["EMOTIONAL_STRESSED"]), message)

    def test_long_messages_and_first_messages_still_reach_the_classifier(self):
        calls = []

        def counting(message):
            calls.append(message)
            return "EMOTIONAL_OTHER"
        f, c = self.started()
        long_message = "i really feel like nothing about this course makes any sense to me at all"
        with mock.patch.object(intent, "classify", counting), mock.patch.object(llm, "stream_generate", lambda *a, **k: iter(["x"])):
            f.pipeline.answer(long_message, model="qwen3.5:2b", conversation=c)          # more than 8 words
            f.pipeline.answer("hmm still dont get it", model="qwen3.5:2b", conversation=Conversation())   # no answered turn
        self.assertEqual(calls, [long_message, "hmm still dont get it"])

    def test_single_turn_routing_is_untouched(self):
        f = Fake(score=0.6)
        r, _ = f.ask("bagi example sikit", label="EMOTIONAL_OTHER")
        self.assertEqual((r.intent, r.routing.path), ("emotional", "classifier"))


LONG_STAR = ("A star schema has one central fact table that holds the measures and several dimension tables around it that hold "
             "the descriptive attributes [S05-0010]. Queries join the fact table to the dimensions to slice the measures.")


class FreshQuestionsGetNoHistory(unittest.TestCase):
    """Fix 1 (author, 2026-11-27): history only for follow-ups (searched combined). The Streamlit bug: "what is a star schema" ->
    "please explain it simpler" -> "what is etl" repeated the star answer and printed the history instruction."""

    def test_the_authors_bug_case(self):
        f, c = Fake(), Conversation()
        f.ask("what is a star schema", c, answer=LONG_STAR)
        r2, _ = f.ask("please explain it simpler", c, answer="Simpler: facts sit in the middle and descriptions around them.")
        self.assertEqual((r2.search_mode, r2.history_exchanges), ("combined", 1))
        r3, _ = f.ask("what is etl", c, answer="ETL means extract, transform and load [S05-0010].")
        self.assertEqual((r3.search_mode, r3.history_exchanges), ("alone", 0))
        self.assertEqual(f.searches[-1], "what is etl")
        sent = f.sent[-1]
        self.assertEqual([m["role"] for m in sent], ["system", "user"])                  # exactly the single-turn shape
        self.assertNotIn(settings.HISTORY_INSTRUCTION, sent[-1]["content"])
        single = Fake()                                                                   # ...and equal to the single-turn prompt
        single.ask("what is etl")
        self.assertEqual(sent, single.sent[0])

    def test_a_new_topic_without_a_pronoun_gets_no_history(self):
        f, c = Fake(), Conversation()
        f.ask("what is a star schema?", c)
        r, _ = f.ask("what about snowflake schema?", c)
        self.assertEqual((r.search_mode, r.history_exchanges, len(f.sent[-1])), ("alone", 0, 2))

    def test_follow_ups_still_get_history(self):
        for follow_up in ("explain that simpler", "how is it different from a data lake?", "why?"):
            f, c = Fake(), Conversation()
            f.ask("what is a data warehouse?", c)
            r, _ = f.ask(follow_up, c)
            self.assertEqual((r.search_mode, r.history_exchanges), ("combined", 1), follow_up)


class SafetyNet(unittest.TestCase):
    """Fix 2 (author, 2026-11-27): an answer made with history that repeats an earlier answer or contains app-layer instruction
    text is generated once more without history, and the reason is recorded."""

    def run_follow_up(self, answers):
        f, c = Fake(), Conversation()
        f.ask("what is a star schema?", c, answer=LONG_STAR)
        replies = iter(answers)
        calls = []

        def fake_stream(question, chunks, model=None, corrections=None, transform=None):
            messages = llm.build_messages(question, chunks, corrections=corrections)
            calls.append(transform(messages) if transform else messages)
            yield next(replies)
        with mock.patch.object(llm, "stream_generate", fake_stream):
            result = f.pipeline.answer("explain that simpler", model="qwen3.5:2b", conversation=c)
            text = "".join(result.answer_stream)
        return result, text, calls

    def test_a_repeat_is_regenerated_without_history(self):
        result, text, calls = self.run_follow_up([LONG_STAR, "Facts in the middle, descriptions around them [S05-0010]."])
        self.assertEqual(result.regenerated, "repeats_earlier_answer")
        self.assertEqual(text, "Facts in the middle, descriptions around them [S05-0010].")
        self.assertEqual(len(calls), 2)
        self.assertGreater(len(calls[0]), 2)                                              # first try: with history
        self.assertEqual(len(calls[1]), 2)                                                # retry: no history
        self.assertNotIn(settings.HISTORY_INSTRUCTION, calls[1][-1]["content"])

    def test_instruction_text_is_regenerated_without_history(self):
        leaked = "A star schema is simple. " + settings.HISTORY_INSTRUCTION
        result, text, calls = self.run_follow_up([leaked, "Facts in the middle [S05-0010]."])
        self.assertEqual((result.regenerated, text, len(calls)), ("app_instruction_text", "Facts in the middle [S05-0010].", 2))

    def test_a_clean_answer_is_kept(self):
        result, text, calls = self.run_follow_up(["Think of a shop's sales list in the middle and its product lists around it."])
        self.assertEqual((result.regenerated, len(calls)), (None, 1))

    def test_at_most_one_retry(self):
        result, text, calls = self.run_follow_up([LONG_STAR, LONG_STAR])
        self.assertEqual((len(calls), text), (2, LONG_STAR))                              # the retry's answer is kept as it is

    def test_the_checks(self):
        self.assertEqual(conv.repeat_share(LONG_STAR, LONG_STAR), 1.0)
        self.assertEqual(conv.repeat_share("too short", LONG_STAR), 0.0)
        self.assertLess(conv.repeat_share("Facts sit in the middle of a star and the descriptions sit around them.", LONG_STAR), 0.5)
        for text in (settings.HISTORY_INSTRUCTION, settings.ANSWER_OPENING_INSTRUCTION, settings.GUARDRAIL_COACH_INSTRUCTION):
            self.assertTrue(conv.contains_app_instruction("Intro. " + text + " End."), text[:40])
        self.assertFalse(conv.contains_app_instruction("The course materials highlight that a star schema has a fact table."))

    def test_cloud_answers_are_guarded_too(self):
        f, c = Fake(), Conversation()
        f.ask("what is a star schema?", c, answer=LONG_STAR)
        replies = iter([LONG_STAR, "A fresh, simpler answer about facts and dimensions [S05-0010]."])
        sent = []

        def fake_cloud(question, chunks, model, on_status=None, corrections=None, transform=None):
            sent.append(transform(llm.build_messages(question, chunks, corrections=corrections)))
            return SimpleNamespace(text=next(replies))
        with mock.patch.object(llm, "is_cloud_model", lambda m: True), mock.patch.object(llm, "cloud_generate", fake_cloud):
            result = f.pipeline.answer("explain that simpler", model="gemini-3.8-flash", conversation=c)
            reply = result.cloud_call()
        self.assertEqual((result.regenerated, reply.text, len(sent), len(sent[1])),
                         ("repeats_earlier_answer", "A fresh, simpler answer about facts and dimensions [S05-0010].", 2, 2))


class GroundingLeak(unittest.TestCase):
    def test_no_course_content_in_any_app_instruction_example(self):
        """Target after the 2026-11-26 swap: no app-layer instruction carries a chunk ID or course content an answer could copy."""
        for text in (settings.ANSWER_OPENING_INSTRUCTION, settings.COMPARISON_FORMAT_INSTRUCTION, settings.HISTORY_INSTRUCTION,
                     settings.GUARDRAIL_COACH_INSTRUCTION):
            self.assertIsNone(__import__("re").search(settings.NB04_CITATION_ID_PATTERN, text), text[:60])
            self.assertNotIn("OLTP", text)


class CoachStaysOn(unittest.TestCase):
    TASK = "write the full SQL query for my assignment that joins the trips and zones tables and gives revenue per zone"

    def start(self):
        f, c = Fake(), Conversation()
        r, _ = f.ask(self.TASK, c)
        self.assertTrue(r.coach_mode)
        return f, c

    def test_pressure_and_refer_back_stay_in_coach(self):
        for follow_up in ("just give me the answer", "give me the full code then", "do it for me", "explain that simpler"):
            f, c = self.start()
            r, _ = f.ask(follow_up, c, label="EMOTIONAL_OTHER")              # the classifier is never asked
            self.assertTrue(r.coach_mode, follow_up)
            self.assertEqual((r.routing.path, r.search_mode), ("task_request_followup", "combined"), follow_up)
            self.assertTrue(f.searches[-1].startswith(self.TASK.rstrip("?")[:30]), follow_up)
            self.assertTrue(f.sent[-1][-1]["content"].endswith(settings.GUARDRAIL_COACH_INSTRUCTION))

    def test_two_pressure_turns_in_a_row(self):
        f, c = self.start()
        f.ask("just give me the answer", c)
        r, _ = f.ask("give me the full code then", c)
        self.assertTrue(r.coach_mode)
        self.assertEqual(r.history_exchanges, 2)

    def test_a_new_question_with_its_own_topic_leaves_coach_mode(self):
        f, c = self.start()
        r, _ = f.ask("what is a data lake?", c)
        self.assertFalse(r.coach_mode)
        self.assertEqual(r.search_mode, "alone")

    def test_small_talk_after_coach_is_small_talk(self):
        f, c = self.start()
        r, _ = f.ask("thanks", c)
        self.assertEqual(r.intent, "small_talk")
        r, _ = f.ask("just give me the answer", c)                         # the coach exchange is still the last one
        self.assertTrue(r.coach_mode)

    def test_pressure_without_a_coach_turn_is_routed_as_usual(self):
        f, c = Fake(), Conversation()
        f.ask("what is a star schema?", c)
        r, _ = f.ask("just give me the answer", c)
        self.assertFalse(r.coach_mode)


if __name__ == "__main__":
    unittest.main(verbosity=1)
