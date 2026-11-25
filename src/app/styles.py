"""
Page styling for the Streamlit chat interface (DEV PROTOTYPE). Display only: nothing here touches retrieval or generation.
Values live in config/settings.py (DEV_UI_*).
"""

from config import settings


def dark_bold_css(theme_type: str | None) -> str:
    """CSS that makes bold text stand out in the dark theme (settings.DEV_UI_DARK_BOLD_*), or "" in the light theme.
    theme_type is st.context.theme.type: "dark" or "light" when Streamlit knows the theme, None when it does not; then the
    rule applies only when the browser prefers a dark colour scheme (which Streamlit's default "system" theme follows)."""
    if theme_type == "light":
        return ""
    rule = (f'[data-testid="stMarkdownContainer"] strong, [data-testid="stMarkdownContainer"] b '
            f"{{ font-weight: {settings.DEV_UI_DARK_BOLD_WEIGHT}; color: {settings.DEV_UI_DARK_BOLD_COLOR}; }}")
    if theme_type == "dark":
        return f"<style>{rule}</style>"
    return f"<style>@media (prefers-color-scheme: dark) {{ {rule} }}</style>"
