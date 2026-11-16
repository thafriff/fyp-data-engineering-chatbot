"""Offline tests of readable citations in the prototype, and of their agreement with the notebook 04 judge's citation check.
Fake answers only: no model, no network. Writes nothing to the project.

Run from the project root:  venv\\Scripts\\python tests\\test_citations.py
"""
import hashlib
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "evaluation" / "notebook04")]
sys.stdout.reconfigure(encoding="utf-8")
os.environ["HF_HUB_OFFLINE"] = "1"

from streamlit.testing.v1 import AppTest   # noqa: E402

import run_judging   # noqa: E402  (read only: its check_citations is the reference the app must agree with)
from config import settings   # noqa: E402
from src.core import citations, llm   # noqa: E402

RETRIEVED = ["S05-0138", "S04-0210", "S06-0204", "S05-0128", "S06-0205"]
APP = str(ROOT / "src" / "app" / "app.py")

FAKE = {
    "single": "OLAP supports analysis [S05-0138].",
    "list": "Two views [S05-0138, S04-0210]. Three more [S06-0204; S05-0128 and S06-0205].",
    "twice": "First [S04-0210]. Second [S05-0138]. First again [S04-0210].",
    "code": "Example:\n```python\nrows = data[i]          # [S06-0205]\npairs = [1, 2]\nbad = cfg[S060322]\n```\n"
            "Inline `arr[S07x]` too. Prose [S05-0138].",
    "outside": "Flink windows [S06-0951] and dimensions [S06-0204].",
    "malformed": "See [S060322] and [S06-0204, S06 0205].",
    "none": "OLAP means online analytical processing.",
    "link_and_note": "A [link](https://example.org) and [S05-0138].\n\n*[Answer cut off at the length limit.]*",
}


class Parsing(unittest.TestCase):
    def check(self, name):
        return citations.check(FAKE[name], RETRIEVED)

    def test_single(self):
        c = self.check("single")
        self.assertEqual((c.text, c.cited_ids), ("OLAP supports analysis [1].", ["S05-0138"]))

    def test_list_is_split_id_by_id(self):
        c = self.check("list")
        self.assertEqual(c.text, "Two views [1, 2]. Three more [3; 4 and 5].")
        self.assertEqual(c.cited_ids, RETRIEVED)

    def test_same_chunk_keeps_its_number(self):
        c = self.check("twice")
        self.assertEqual(c.text, "First [1]. Second [2]. First again [1].")
        self.assertEqual(c.cited_ids, ["S04-0210", "S05-0138"])

    def test_code_brackets_are_ignored_but_a_real_id_in_code_counts(self):
        c = self.check("code")
        self.assertIn("# [1]", c.text)
        self.assertIn("data[i]", c.text)
        self.assertIn("[1, 2]", c.text)               # an ordinary list in code is untouched
        self.assertIn("cfg[S060322]", c.text)
        self.assertEqual(c.cited_ids, ["S06-0205", "S05-0138"])
        self.assertEqual(c.malformed, [])             # brackets in code are not judged as malformed

    def test_outside_set_is_flagged_not_numbered(self):
        c = self.check("outside")
        self.assertEqual(c.text, "Flink windows [S06-0951] and dimensions [1].")
        self.assertEqual((c.outside_set, c.cited_ids), (["S06-0951"], ["S06-0204"]))

    def test_malformed_is_flagged(self):
        c = self.check("malformed")
        self.assertEqual(c.malformed, ["S060322", "S06 0205"])
        self.assertEqual(c.text, "See [S060322] and [1, S06 0205].")

    def test_no_citation(self):
        c = self.check("none")
        self.assertFalse(c.cites_course_material)
        self.assertEqual(c.text, FAKE["none"])

    def test_links_and_guard_notes_are_not_citations(self):
        c = self.check("link_and_note")
        self.assertEqual((c.cited_ids, c.malformed, c.outside_set), (["S05-0138"], [], []))
        self.assertIn("[link](https://example.org)", c.text)
        self.assertIn("*[Answer cut off at the length limit.]*", c.text)

    def test_counts_agree_with_the_judge_runner(self):
        for name, answer in FAKE.items():
            mine = citations.check(answer, RETRIEVED).counts
            reference = run_judging.check_citations(answer, set(RETRIEVED))
            self.assertEqual({k: mine[k] for k in reference}, reference, name)

    def test_the_id_pattern_is_the_shared_one(self):
        self.assertEqual(citations._CITATION_ID.pattern, run_judging._CITATION_ID.pattern)
        for a, b in [(citations._BRACKET, run_judging._BRACKET), (citations._CITATION_LIKE, run_judging._CITATION_LIKE),
                     (citations._TOKEN_SPLIT, run_judging._TOKEN_SPLIT), (citations._FENCE, run_judging._FENCE),
                     (citations._INLINE_CODE, run_judging._INLINE_CODE)]:
            self.assertEqual((a.pattern, a.flags), (b.pattern, b.flags))

    def test_session_counts(self):
        checks = [citations.check(FAKE[n], RETRIEVED).to_dict() for n in ("single", "outside", "malformed", "none")]
        self.assertEqual(citations.session_counts(checks),
                         {"answers_checked": 4, "outside_set": 1, "malformed": 1, "no_citation": 1})


class Interface(unittest.TestCase):
    """The real page and the real retrieval, with the local model's stream faked. The fake cites the actual retrieved IDs."""

    def setUp(self):
        self.template = ""

        def fake_stream(question, chunks, model=None, **kwargs):
            ids = [c.chunk_id for c in chunks]
            text = self.template.format(*ids)
            for i in range(0, len(text), 7):
                yield text[i:i + 7]
        for p in (mock.patch.object(llm, "stream_generate", fake_stream),
                  mock.patch.object(llm, "check_backend", lambda model: (True, "fake backend"))):
            p.start()
            self.addCleanup(p.stop)
        self.at = AppTest.from_file(APP, default_timeout=180).run()

    def ask(self, question, template=""):
        self.template = template
        self.at.chat_input[0].set_value(question).run()
        self.assertFalse(self.at.exception, self.at.exception)
        return self.at.chat_message[-1]

    @staticmethod
    def texts(message):
        return [m.value for m in message.markdown], [c.value for c in message.caption], [e.label for e in message.expander]

    def test_list_citation_renders_numbers_sources_and_full_excerpt(self):
        message = self.ask("What is OLAP?", "OLAP views [{0}, {1}]. Again [{0}].")
        markdown, captions, labels = self.texts(message)
        self.assertEqual(markdown[0], "OLAP views [1, 2]. Again [1].")
        self.assertIn("**Sources**", markdown)
        self.assertTrue(labels[0].startswith("[1] ") and labels[1].startswith("[2] "), labels)
        self.assertTrue(labels[-1].startswith("All retrieved material given to the model · 5 chunks"), labels)
        self.assertEqual(len(labels), 3)
        self.assertFalse(any("does not cite" in c for c in captions))

    def test_problems_are_noted_and_counted_in_the_sidebar(self):
        self.ask("What is OLAP?", "Flink [S06-0951]. Odd [S060322]. Fine [{0}].")
        _, captions, labels = self.texts(self.at.chat_message[-1])
        self.assertTrue(any("Not from the material given to the model: `S06-0951`" in c for c in captions), captions)
        self.assertTrue(any("Malformed citation, not a chunk ID: `S060322`" in c for c in captions), captions)
        self.assertEqual(sum(label.startswith("[") for label in labels), 1)      # S06-0951 is not a source
        self.ask("What is a star schema?", "No citation here.")
        _, captions, _ = self.texts(self.at.chat_message[-1])
        self.assertTrue(any(settings.DEV_CITATION_MSG_NONE in c for c in captions), captions)
        sidebar = " ".join(c.value for c in self.at.sidebar.caption)
        self.assertIn("2 model answers in this chat: 1 cited material not given to the model · 1 had a malformed "
                      "citation · 1 cited no course material", sidebar)

    def test_citations_survive_a_rerun(self):
        self.ask("What is OLAP?", "Answer [{2}].")
        before = self.texts(self.at.chat_message[-1])
        self.at.run()
        self.assertEqual(self.texts(self.at.chat_message[-1]), before)
        self.assertEqual(before[0][0], "Answer [1].")

    def test_fixed_reply_has_no_sources(self):
        markdown, captions, labels = self.texts(self.ask("hi"))
        self.assertEqual(labels, [])
        self.assertNotIn("**Sources**", markdown)
        self.assertFalse(any("cite" in c for c in captions))

    def test_fallback_has_no_sources_section(self):
        markdown, captions, labels = self.texts(self.ask("What is the capital of France?"))
        self.assertEqual(markdown[0], settings.FALLBACK_MESSAGE)
        self.assertNotIn("**Sources**", markdown)
        self.assertFalse(any("cite" in c for c in captions))
        self.assertEqual(len(labels), 1)
        self.assertIn("not used: below the relevance threshold", labels[0])


class Frozen(unittest.TestCase):
    def test_prompt_unchanged(self):
        self.assertTrue(hashlib.sha256(settings.SYSTEM_PROMPT.encode("utf-8")).hexdigest().startswith("fe8912d1"))


if __name__ == "__main__":
    unittest.main(verbosity=1)
