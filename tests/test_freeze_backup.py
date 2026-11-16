"""Offline tests of the automatic freeze backup in evaluation/notebook04/freeze_contexts.py. Works only in temp folders.

Run from the project root:  venv\\Scripts\\python tests\\test_freeze_backup.py
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "evaluation" / "notebook04")]
sys.stdout.reconfigure(encoding="utf-8")

import freeze_contexts  # noqa: E402
from config import settings  # noqa: E402


class FreezeBackup(unittest.TestCase):
    def test_previous_freeze_is_copied_before_it_is_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            contexts, manifest, backups = tmp / "contexts.json", tmp / "contexts_manifest.json", tmp / "backups"
            contexts.write_text('{"items": "old"}', encoding="utf-8")
            manifest.write_text('{"items": "old manifest"}', encoding="utf-8")
            with mock.patch.multiple(settings, NB04_CONTEXTS_FILE=contexts, NB04_MANIFEST_FILE=manifest, NB04_BACKUP_DIR=backups):
                folder = freeze_contexts.back_up_previous_freeze()
            self.assertEqual(folder.parent, backups)
            self.assertEqual((folder / "contexts.json").read_text(encoding="utf-8"), '{"items": "old"}')
            self.assertEqual((folder / "contexts_manifest.json").read_text(encoding="utf-8"), '{"items": "old manifest"}')

    def test_nothing_to_back_up_on_a_first_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with mock.patch.multiple(settings, NB04_CONTEXTS_FILE=tmp / "contexts.json",
                                     NB04_MANIFEST_FILE=tmp / "contexts_manifest.json", NB04_BACKUP_DIR=tmp / "backups"):
                self.assertIsNone(freeze_contexts.back_up_previous_freeze())
            self.assertFalse((tmp / "backups").exists())

    def test_backup_folder_is_inside_the_project_and_git_ignored(self):
        self.assertEqual(settings.NB04_BACKUP_DIR.resolve().parent, settings.NB04_DIR.resolve())
        ignored = subprocess.run(["git", "check-ignore", "-q", str(settings.NB04_BACKUP_DIR / "x" / "contexts.json")],
                                 cwd=ROOT).returncode == 0
        self.assertTrue(ignored)


if __name__ == "__main__":
    unittest.main(verbosity=1)
