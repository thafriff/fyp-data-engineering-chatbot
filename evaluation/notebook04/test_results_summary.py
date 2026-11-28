"""
Offline tests for results_summary.py (STEP 4). Nothing here calls a model or a judge.

The bootstrap tests use small synthetic tables. The real-data tests run only where the Git-ignored caches exist; they check that the
summary's point estimates are run_judging.build_report's own, that the saved results_summary.json is what a fresh build gives (same
seed, same numbers), and that it holds no answer text, judge quotes or keys.

Usage (from the project root):  venv\\Scripts\\python evaluation\\notebook04\\test_results_summary.py
"""

import json
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import results_summary as rs  # noqa: E402
import run_judging as rj  # noqa: E402
from config import settings  # noqa: E402

N = 100


def table(base: float = 4.0, faith: float = 0.9, latency: float = 4.0, cost: float = 0.002, seed: int = 0, reachable=range(94)):
    """A synthetic per-question table: 1-5 scores around base, NaN primary Accuracy outside the reachable questions."""
    rng = np.random.default_rng(seed)
    scores = np.clip(np.round(base + rng.normal(0, 0.7, N)), 1, 5)
    accuracy = np.full(N, np.nan)
    accuracy[list(reachable)] = scores[list(reachable)]
    return {"accuracy": accuracy, "accuracy_all": scores, "relevance": scores.copy(), "clarity": scores.copy(),
            "alignment": scores.copy(), "faithfulness": np.clip(faith + rng.normal(0, 0.05, N), 0, 1),
            "latency": np.abs(latency + rng.normal(0, 1, N)), "cost": np.full(N, cost)}


class Bootstrap(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.object(settings, "NB04_BOOTSTRAP_RESAMPLES", 300)
        patch.start()
        self.addCleanup(patch.stop)
        self.qids = [f"Q{n:02d}" for n in range(1, N + 1)]
        self.reachable = list(range(94))

    def boot(self, tables, ease=None):
        ease = ease or {m: 4 for m in tables}
        return rs.run_bootstrap(self.qids, tables, ease, self.reachable)

    def test_same_seed_same_numbers(self):
        tables = {"a": table(seed=1), "b": table(seed=2)}
        first, second = self.boot(tables), self.boot(tables)
        for model in tables:
            np.testing.assert_array_equal(first["replicates"][model]["total"], second["replicates"][model]["total"])
            np.testing.assert_array_equal(first["accuracy94"][model]["raw"], second["accuracy94"][model]["raw"])

    def test_each_resample_is_scored_by_run_judgings_own_functions(self):
        tables = {"a": table(seed=1)}
        with mock.patch.object(rs.rj, "weighted_total", wraps=rj.weighted_total) as weighted, \
                mock.patch.object(rs.rj, "normalise", wraps=rj.normalise) as normalise:
            result = self.boot(tables)
        self.assertEqual(normalise.call_count, settings.NB04_BOOTSTRAP_RESAMPLES)
        self.assertEqual(weighted.call_count, 4 * settings.NB04_BOOTSTRAP_RESAMPLES)   # total, two sensitivity, all-points
        # Resample 0 by hand, from the same generator.
        rng = np.random.default_rng(settings.NB04_BOOTSTRAP_SEED)
        index = rng.integers(0, N, size=(settings.NB04_BOOTSTRAP_RESAMPLES, N))[0]
        t = tables["a"]
        metrics = {"accuracy": float(np.nanmean(t["accuracy"][index])), "relevance": float(t["relevance"][index].mean()),
                   "clarity": float(t["clarity"][index].mean()), "alignment": float(t["alignment"][index].mean()),
                   "faithfulness": float(t["faithfulness"][index].mean()), "mean_seconds": float(t["latency"][index].mean()),
                   "usd_per_100": float(t["cost"][index].mean()) * 100, "ease": 4}
        self.assertAlmostEqual(result["replicates"]["a"]["total"][0], rj.weighted_total(rj.normalise(metrics)), places=12)

    def test_identical_models_differ_by_nothing(self):
        tables = {"a": table(seed=1), "b": table(seed=1)}
        result = self.boot(tables)
        rep = result["replicates"]
        pairs = rs.pairwise(["a", "b"], {m: rep[m]["total"] for m in tables}, {"a": 0.5, "b": 0.5})
        self.assertEqual(pairs[0]["ci95"], [0.0, 0.0])
        self.assertFalse(pairs[0]["significant"])

    def test_a_model_better_on_every_question_differs_significantly(self):
        tables = {"good": table(base=4.6, faith=0.95, seed=1), "weak": table(base=3.0, faith=0.6, seed=1)}
        rep = self.boot(tables)["replicates"]
        pair = rs.pairwise(["good", "weak"], {m: rep[m]["total"] for m in tables}, {"good": 1, "weak": 0})[0]
        self.assertTrue(pair["significant"])
        self.assertGreater(pair["ci95"][0], 0)

    def test_primary_accuracy_interval_uses_the_reachable_questions_only(self):
        t = table(seed=3)
        t["accuracy_all"][94:] = 1.0          # the six unreachable questions are bad on all points only
        result = self.boot({"a": t})
        self.assertFalse(np.isnan(result["accuracy94"]["a"]["raw"]).any())
        self.assertEqual(result["reachable_questions"], 94)

    def test_response_time_from_a_subset_of_questions(self):
        t = table(latency=2.0, seed=4)
        t["latency"][20:] = np.nan            # pre-registration (a): only Q01-Q20 re-measured
        rep = self.boot({"a": t})["replicates"]["a"]
        self.assertFalse(np.isnan(rep["total"]).any())
        self.assertAlmostEqual(float(np.nanmean(t["latency"])), float(np.nanmean(t["latency"][:20])))

    def test_ease_has_no_interval_and_enters_every_resample_unchanged(self):
        rep = self.boot({"a": table(seed=5)}, ease={"a": 3})["replicates"]["a"]
        self.assertTrue(np.all(rep["ease"] == 0.5))

    def test_interval_is_the_percentile_interval(self):
        values = np.arange(1, 1001, dtype=float)
        low, high = rs.interval(values)
        self.assertAlmostEqual(low, float(np.percentile(values, 2.5)))
        self.assertAlmostEqual(high, float(np.percentile(values, 97.5)))
        self.assertIsNone(rs.interval(np.array([math.nan])))



class FrozenValues(unittest.TestCase):
    """Plan section 3 (weights, anchors, tie margin; frozen 2026-11-01) and section 6a(c) (bootstrap; fixed 2026-11-27)."""

    def test_bootstrap_as_fixed(self):
        self.assertEqual((settings.NB04_BOOTSTRAP_RESAMPLES, settings.NB04_BOOTSTRAP_SEED, settings.NB04_BOOTSTRAP_INTERVAL),
                         (10_000, 42, 0.95))

    def test_weights_anchors_and_tie_margin_unchanged(self):
        self.assertEqual(settings.NB04_WEIGHTS, {"accuracy": 0.25, "alignment": 0.15, "faithfulness": 0.15, "relevance": 0.10,
                                                 "clarity": 0.10, "response_time": 0.10, "cost": 0.10, "ease": 0.05})
        self.assertEqual((settings.NB04_RESPONSE_TIME_BEST_SECONDS, settings.NB04_RESPONSE_TIME_WORST_SECONDS,
                          settings.NB04_COST_WORST_USD, settings.NB04_TIE_MARGIN), (3, 15, 5.00, 0.02))
        self.assertEqual(tuple(settings.NB04_SENSITIVITY_EXCLUSIONS), ("response_time", "cost"))


HAVE_DATA = all((settings.NB04_RAW_DIR / f"{m}.json").exists() for m in settings.NB04_CANDIDATES) and \
    (settings.NB04_JUDGE_DIR / "rubric_gpt-5.4.json").exists() and settings.NB04_CONTEXTS_FILE.exists()


@unittest.skipUnless(HAVE_DATA, "needs the Git-ignored answer and judge caches")
class RealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = rs.build_summary()
        cls.report = rj.build_report()

    def test_point_estimates_are_build_reports(self):
        rows = {r["model"]: r for r in self.report["rows"]}
        for m in self.summary["models"]:
            self.assertAlmostEqual(m["total"], rows[m["model"]]["total"], places=12)
            for criterion, c in m["criteria"].items():
                self.assertAlmostEqual(c["normalised"], rows[m["model"]]["normalised"][criterion], places=12)
        self.assertEqual(self.summary["decision"]["winner"], self.report["main"]["winner"])
        for ours, theirs in zip(self.summary["decision"]["sensitivity"], self.report["sensitivity"]):
            self.assertEqual((ours["excluded"], ours["winner"]), (theirs["excluded"], theirs["winner"]))

    def test_every_interval_contains_its_point_estimate(self):
        for m in self.summary["models"]:
            low, high = m["total_ci95"]
            self.assertLessEqual(low, m["total"] + 1e-9)
            self.assertGreaterEqual(high, m["total"] - 1e-9)

    def test_saved_file_is_a_fresh_build(self):
        saved = json.loads(settings.NB04_RESULTS_SUMMARY_FILE.read_text(encoding="utf-8"))
        fresh = json.loads(json.dumps(self.summary))
        for doc in (saved, fresh):
            doc["meta"].pop("generated_at")
        self.assertEqual(saved, fresh)

    def test_no_answer_text_quotes_or_keys(self):
        text = settings.NB04_RESULTS_SUMMARY_FILE.read_text(encoding="utf-8")
        for name in ("\"answer\"", "solution_quote", "contradiction_quote", "\"reasons\"", "last_output", "sk-", "AIza"):
            self.assertNotIn(name, text)
        for cache in rj.load_answer_caches().values():
            for record in cache["results"].values():
                answer = record["answer"] or ""
                if len(answer) > 80:
                    self.assertNotIn(answer[20:80], text)


if __name__ == "__main__":
    unittest.main(verbosity=1)
