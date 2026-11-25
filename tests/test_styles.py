"""
Offline tests of the chat page's dark-theme bold styling (src/app/styles.py, settings.DEV_UI_DARK_BOLD_*).

Run from the project root:  venv\\Scripts\\python tests\\test_styles.py
"""

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
os.environ["HF_HUB_OFFLINE"] = "1"

from config import settings  # noqa: E402
from src.app import styles  # noqa: E402

RULE = (f'[data-testid="stMarkdownContainer"] strong, [data-testid="stMarkdownContainer"] b '
        f"{{ font-weight: {settings.DEV_UI_DARK_BOLD_WEIGHT}; color: {settings.DEV_UI_DARK_BOLD_COLOR}; }}")


class DarkBold(unittest.TestCase):
    def test_dark_theme_gets_the_rule(self):
        self.assertEqual(styles.dark_bold_css("dark"), f"<style>{RULE}</style>")

    def test_light_theme_is_unchanged(self):
        self.assertEqual(styles.dark_bold_css("light"), "")

    def test_unknown_theme_follows_the_browser(self):
        self.assertEqual(styles.dark_bold_css(None), f"<style>@media (prefers-color-scheme: dark) {{ {RULE} }}</style>")

    def test_settings(self):
        self.assertRegex(settings.DEV_UI_DARK_BOLD_COLOR, r"^#[0-9A-Fa-f]{6}$")
        self.assertGreaterEqual(settings.DEV_UI_DARK_BOLD_WEIGHT, 700)

    def test_contrast_on_the_dark_background(self):
        """WCAG contrast of the bold colour on Streamlit's dark background (#0E1117) must be at least 4.5:1 (AA, body text)."""
        def luminance(hex_colour: str) -> float:
            channels = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
            linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
            return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
        ratio = (luminance(settings.DEV_UI_DARK_BOLD_COLOR) + 0.05) / (luminance("#0E1117") + 0.05)
        self.assertGreaterEqual(ratio, 4.5, f"{ratio:.1f}:1")

    def test_the_page_runs_with_the_style(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(ROOT / "src" / "app" / "app.py"), default_timeout=180).run()
        self.assertFalse(at.exception, at.exception)


if __name__ == "__main__":
    unittest.main(verbosity=2)
