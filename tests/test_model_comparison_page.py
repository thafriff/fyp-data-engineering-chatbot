"""
Offline tests for the read-only "Model comparison" page (src/app/pages/1_Model_comparison.py), driven by streamlit.testing (AppTest).

What must hold: the page renders from results_summary.json; its only button is "Recompute scores from saved results"; loading it and
pressing the button never call a model or a judge (providers.send fails the test if anything does); the button runs
results_summary.write_summary, which reads saved files only; one fixed colour per model, the same in both charts.

Usage (from the project root):  venv\\Scripts\\python tests\\test_model_comparison_page.py
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK04 = PROJECT_ROOT / "evaluation" / "notebook04"
for extra in (PROJECT_ROOT, NOTEBOOK04):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from streamlit.testing.v1 import AppTest  # noqa: E402

import providers  # noqa: E402
import results_summary  # noqa: E402
from config import settings  # noqa: E402

PAGE = PROJECT_ROOT / "src" / "app" / "pages" / "1_Model_comparison.py"
HAVE_SUMMARY = settings.NB04_RESULTS_SUMMARY_FILE.exists()


def load_page_module():
    spec = importlib.util.spec_from_file_location("model_comparison_page", PAGE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)            # main() is behind the __main__ guard, so nothing renders here
    return module


class NoModelCalls(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.object(providers, "send", side_effect=AssertionError("the page must never call a model or judge"))
        patch.start()
        self.addCleanup(patch.stop)

    @unittest.skipUnless(HAVE_SUMMARY, "needs evaluation/notebook04/results_summary.json")
    def test_renders_with_one_button_and_every_section(self):
        at = AppTest.from_file(str(PAGE), default_timeout=120).run()
        self.assertEqual(list(at.exception), [])
        self.assertEqual([b.label for b in at.button], [settings.MODEL_COMPARISON_RECOMPUTE_LABEL])
        self.assertEqual([t.value for t in at.title], [settings.MODEL_COMPARISON_PAGE_TITLE])
        headers = [s.value for s in at.subheader]
        for header in ("Final weighted ranking", "The eight criterion scores", "Accuracy: primary and secondary", "Latency and cost",
                       "Pre-registered sensitivity rankings", "Guardrail compliance (not in the weighted score)", "Notes"):
            self.assertIn(header, headers)
        self.assertTrue(any("Winner under the decision rule" in m.value for m in at.markdown))

    @unittest.skipUnless(HAVE_SUMMARY, "needs evaluation/notebook04/results_summary.json")
    def test_the_button_recomputes_from_saved_results_only(self):
        with mock.patch.object(results_summary, "write_summary") as write:
            at = AppTest.from_file(str(PAGE), default_timeout=120).run()
            write.assert_not_called()
            at.button[0].click().run()
        write.assert_called_once_with()
        self.assertEqual(list(at.exception), [])
        self.assertTrue(any("recomputed" in s.value for s in at.success))

    def test_without_a_summary_the_page_says_so(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(settings, "NB04_RESULTS_SUMMARY_FILE", Path(tmp) / "results_summary.json"):
            at = AppTest.from_file(str(PAGE), default_timeout=120).run()
        self.assertEqual(list(at.exception), [])
        self.assertTrue(any("No results yet" in i.value for i in at.info))

    def test_the_page_imports_no_api_code(self):
        source = PAGE.read_text(encoding="utf-8")
        for forbidden in ("import providers", "batch_judging", "run_generation", "OpenAI(", "Anthropic(", "genai.Client"):
            self.assertNotIn(forbidden, source)


@unittest.skipUnless(HAVE_SUMMARY, "needs evaluation/notebook04/results_summary.json")
class Charts(unittest.TestCase):
    def setUp(self):
        self.page = load_page_module()
        self.summary = json.loads(settings.NB04_RESULTS_SUMMARY_FILE.read_text(encoding="utf-8"))
        self.models = [m["model"] for m in self.summary["models"]]

    def colours(self, chart):
        spec = chart.to_dict()
        layers = spec.get("layer", [spec])
        return [(layer["encoding"]["color"]["scale"]["domain"], layer["encoding"]["color"]["scale"]["range"]) for layer in layers]

    def test_one_fixed_colour_per_model_in_line_up_order(self):
        expected = (list(settings.NB04_CANDIDATES), [settings.NB04_MODEL_COLOURS["light"][m] for m in settings.NB04_CANDIDATES])
        for chart in (self.page.ranking_chart(self.summary, self.models), self.page.criteria_chart(self.summary, self.models)):
            for domain, colours in self.colours(chart):
                self.assertEqual((domain, colours), expected)

    def test_colour_sets_cover_every_candidate_in_both_themes(self):
        for mode in ("light", "dark"):
            self.assertEqual(set(settings.NB04_MODEL_COLOURS[mode]), set(settings.NB04_CANDIDATES))
            self.assertEqual(len(set(settings.NB04_MODEL_COLOURS[mode].values())), len(settings.NB04_CANDIDATES))


if __name__ == "__main__":
    unittest.main(verbosity=1)
