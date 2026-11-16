"""Offline tests of evaluation/notebook04/ease_note.py. The real ease_log.json is never written.

Run from the project root:  venv\\Scripts\\python tests\\test_ease_note.py
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "evaluation" / "notebook04")]
sys.stdout.reconfigure(encoding="utf-8")

import ease_note  # noqa: E402
from config import settings  # noqa: E402

NOW = "2026-11-16T10:00:00+00:00"


class EaseNote(unittest.TestCase):
    def log(self):
        return {"meta": {}, "models": {"gemma-4-31b-it": {"score": None, "trial_errors": []}}}

    def test_plus_key_appends_and_now_is_stamped(self):
        log = self.log()
        touched = ease_note.apply_note(log, "gemma-4-31b-it", {"+trial_reruns": {"started_at": "now", "n": [1, "now"]}}, NOW)
        ease_note.apply_note(log, "gemma-4-31b-it", {"+trial_reruns": {"completed_at": "now"}}, NOW)
        self.assertEqual(touched, ["+trial_reruns"])
        self.assertEqual(log["models"]["gemma-4-31b-it"]["trial_reruns"],
                         [{"started_at": NOW, "n": [1, NOW]}, {"completed_at": NOW}])

    def test_plain_key_sets_the_value(self):
        log = self.log()
        ease_note.apply_note(log, "gemma-4-31b-it", {"trial_outcome": "saved"}, NOW)
        self.assertEqual(log["models"]["gemma-4-31b-it"]["trial_outcome"], "saved")

    def test_scores_are_refused_and_unknown_models_too(self):
        with self.assertRaises(ValueError):
            ease_note.apply_note(self.log(), "gemma-4-31b-it", {"score": 4}, NOW)
        with self.assertRaises(KeyError):
            ease_note.apply_note(self.log(), "no-such-model", {"x": 1}, NOW)

    def test_command_line_writes_only_the_given_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path, patch_path = Path(tmp) / "ease_log.json", Path(tmp) / "patch.json"
            log_path.write_text(json.dumps(self.log()), encoding="utf-8")
            patch_path.write_text(json.dumps({"+trial_reruns": {"started_at": "now"}}), encoding="utf-8")
            with mock.patch.object(settings, "NB04_EASE_LOG_FILE", log_path), \
                    mock.patch.object(sys, "argv", ["ease_note.py", "gemma-4-31b-it", str(patch_path)]):
                ease_note.main()
            saved = json.loads(log_path.read_text(encoding="utf-8"))
        self.assertEqual(len(saved["models"]["gemma-4-31b-it"]["trial_reruns"]), 1)
        self.assertRegex(saved["models"]["gemma-4-31b-it"]["trial_reruns"][0]["started_at"], r"\+00:00$")


if __name__ == "__main__":
    unittest.main(verbosity=1)
