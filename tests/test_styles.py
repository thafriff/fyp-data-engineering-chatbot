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
        self.assertGreaterEqual(settings.DEV_UI_DARK_BOLD_WEIGHT, 800)      # heavier than Streamlit's own bold (600-700)

    def test_bold_is_near_white_not_a_warning_colour(self):
        """2026-11-26 (author): the red tint #FF8F8F looked like warning text. Bold must be near-white: every channel at least
        0xF0 and the channels within 0x10 of each other, so it has no red (or other) cast."""
        channels = [int(settings.DEV_UI_DARK_BOLD_COLOR[i:i + 2], 16) for i in (1, 3, 5)]
        self.assertTrue(all(c >= 0xF0 for c in channels), channels)
        self.assertLessEqual(max(channels) - min(channels), 0x10, channels)

    def test_sources_heading_uses_the_same_bold_rule(self):
        """The Sources heading and the chunk IDs in the Sources list are markdown bold, which is what the rule styles, so they
        match the answer's bold text (no separate colour)."""
        # The Sources helpers moved to src/app/shared.py on 2026-11-28 (shared by the chat, Quiz and Study plan pages).
        shared = (ROOT / "src" / "app" / "shared.py").read_text(encoding="utf-8")
        self.assertIn('st.markdown("**Sources**")', shared)
        self.assertIn("[data-testid=\"stMarkdownContainer\"] strong", styles.dark_bold_css("dark"))
        self.assertNotIn("#FF", styles.dark_bold_css("dark").upper().replace(settings.DEV_UI_DARK_BOLD_COLOR.upper(), ""))

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
