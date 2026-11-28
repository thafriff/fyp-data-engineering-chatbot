"""
The quiz card (feature B), shown in the chat and on the Quiz page. Display only: the quiz is written and checked in src/core/quiz.py.

Before the student answers: the questions in a form (a radio button per multiple-choice item, a text box for the short answer) and a
"Check answers" button. After: each multiple-choice answer marked by code, the short answer's model answer and key points with two
self-mark buttons, every explanation with its citations renumbered as in chat, and the Sources list. Sources stay hidden until the
student has answered, because they would give the answers away. The card's state lives in st.session_state under its own key, so a
rerun keeps the answers.
"""

import streamlit as st

from config import settings
from src.app.shared import show_cited_sources, show_retrieved
from src.core import citations
from src.core import quiz as quiz_rules

_SEPARATOR = "\n␞\n"      # joins the explanations for one renumbering pass, then splits them again


def renumbered_explanations(quiz: dict) -> tuple[list[str], citations.CitationCheck]:
    """The explanations with their chunk IDs numbered [1], [2] ... across the whole quiz, and the citation check behind it."""
    check = citations.check(_SEPARATOR.join(item["explanation"] for item in quiz["items"]), quiz["chunk_ids"])
    return check.text.split(_SEPARATOR), check


def _state(key: str) -> dict:
    return st.session_state.setdefault(key, {"submitted": False, "answers": {}, "self_marks": {}})


def _ask(quiz: dict, key: str) -> None:
    """The unanswered card: one form, submitted once."""
    state = _state(key)
    with st.form(f"{key}_form"):
        for n, item in enumerate(quiz["items"]):
            st.markdown(f"**{n + 1}. {item['question']}**")
            if item["type"] == "mcq":
                st.radio("Choose one", list(quiz_rules.LETTERS), index=None, key=f"{key}_q{n}", label_visibility="collapsed",
                         format_func=lambda letter, options=item["options"]: f"{letter}. {options[letter]}")
            else:
                st.text_area("Your answer", key=f"{key}_q{n}", label_visibility="collapsed",
                             placeholder="Write your answer, then compare it with the model answer.")
        if st.form_submit_button(settings.QUIZ_CHECK_LABEL):
            state["answers"] = {n: st.session_state.get(f"{key}_q{n}") for n in range(len(quiz["items"]))}
            state["submitted"] = True
            st.rerun()


def _reveal(quiz: dict, sources: list[dict], key: str) -> None:
    """The answered card: marks, model answers, explanations and sources."""
    state = _state(key)
    explanations, check = renumbered_explanations(quiz)
    marks = quiz_rules.mark(quiz, state["answers"])
    yes, no = settings.QUIZ_SELF_MARK_LABELS
    for n, item in enumerate(quiz["items"]):
        st.markdown(f"**{n + 1}. {item['question']}**")
        given = state["answers"].get(n)
        if item["type"] == "mcq":
            right = f"{item['correct']}. {item['options'][item['correct']]}"
            if given is None:
                st.warning(f"No answer chosen. Correct answer: {right}")
            elif marks[n]:
                st.success(f"Correct: {right}")
            else:
                st.error(f"Your answer: {given}. {item['options'][given]} · Correct answer: {right}")
        else:
            st.caption("Your answer")
            st.text(given or "(no answer)")
            st.info(f"Model answer: {item['answer']}")
            st.markdown("Key points: " + "; ".join(item["key_points"]))
            mark = state["self_marks"].get(n)
            if mark is None:
                left, right_col = st.columns(2)
                if left.button(yes, key=f"{key}_self{n}_yes"):
                    state["self_marks"][n] = True
                    st.rerun()
                if right_col.button(no, key=f"{key}_self{n}_no"):
                    state["self_marks"][n] = False
                    st.rerun()
            else:
                st.caption(f"You marked this: {yes if mark else no}")
        st.caption(explanations[n])
    summary = f"**Multiple choice: {sum(marks.values())} of {len(marks)} correct.**"
    shorts = [n for n, item in enumerate(quiz["items"]) if item["type"] == "short"]
    if shorts:
        marked = [state["self_marks"].get(n) for n in shorts]
        summary += (" Short answer: mark it yourself above." if None in marked
                    else f" Short answer: {sum(marked)} of {len(marked)} self-marked as right.")
    st.markdown(summary)
    show_cited_sources(sources, check.to_dict())
    show_retrieved(sources, False)


def render(quiz: dict, sources: list[dict], key: str) -> None:
    """The whole card for one quiz; key keeps its state apart from every other quiz on the page."""
    st.markdown(settings.QUIZ_INTRO.format(topic=quiz["topic"], n=len(quiz["items"])))
    if len(quiz["items"]) < settings.QUIZ_MCQ_COUNT + settings.QUIZ_SHORT_COUNT:
        st.caption(settings.QUIZ_FEWER_NOTE.format(n=len(quiz["items"])))
    if _state(key)["submitted"]:
        _reveal(quiz, sources, key)
    else:
        _ask(quiz, key)
