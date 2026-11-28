"""
The study plan card (feature A), shown in the chat and on the Study plan page. Display only: the plan is built in src/core/study_plan.py.

Each session: its time, goal, sub-topics (real section titles) with numbered citations, the suggested questions as buttons (a click
sends the question to the chat), a quiz checkpoint (feature B, on the chunks of the sessions since the last checkpoint) and the break.
Then "If you have more time", a Markdown download and the Sources list. The card's state lives in st.session_state under its key.
"""

import re
from collections.abc import Callable

import streamlit as st

from config import settings
from src.app import quiz_view
from src.app.shared import show_cited_sources
from src.core import study_plan

MORE_TIME_SHOWN = 10


def _budget(plan: dict) -> str:
    if plan["mode"] == "days":
        hours = plan["minutes_per_day"] / 60
        assumed = "" if plan["per_day_given"] else " (assumed; say \"3 hours a day\" to change it)"
        return f"{plan['days']} day(s) × {hours:g} h a day{assumed}"
    return f"{plan['budget_minutes']} minutes"


def numbering(plan: dict) -> dict:
    """The plan's citations numbered in order of first appearance, in the shape show_cited_sources expects."""
    ordered = list(dict.fromkeys(i for s in plan["sessions"] for item in s["items"] for i in item["chunk_ids"]))
    return {"cited_ids": ordered, "outside_set": [], "malformed": []}


def checkpoint_chunks(plan: dict, n: int) -> tuple[str, list[str]]:
    """The sub-topic titles and chunk IDs of the sessions since the previous checkpoint, up to session n (most relevant first)."""
    sessions = plan["sessions"]
    start = max((s["n"] for s in sessions if s["checkpoint"] and s["n"] < n), default=0)
    covered = [s for s in sessions if start < s["n"] <= n]
    items = [i for s in covered for i in s["items"] if not i.get("review")]
    if not items:                                      # a review-only checkpoint quizzes the whole plan
        items = [i for s in sessions for i in s["items"] if not i.get("review")]
    titles = ", ".join(dict.fromkeys(i["title"] for i in items))
    return titles, list(dict.fromkeys(c for i in items for c in i["chunk_ids"]))


def render(plan: dict, sources: list[dict], key: str, make_quiz: Callable[[str, list[str]], dict],
           ask: Callable[[str], None]) -> None:
    """The whole card. make_quiz(titles, chunk_ids) returns {"quiz": ..., "sources": ...} or {"error": ...}; ask(question) sends a
    suggested question to the chat."""
    numbers = {cid: n for n, cid in enumerate(numbering(plan)["cited_ids"], 1)}
    st.markdown(f"**Study plan: {plan['topic']}** · {_budget(plan)} · {len(plan['sessions'])} session(s) of up to "
                f"{settings.PLAN_SESSION_MINUTES} minutes")
    if plan.get("note"):
        st.caption(plan["note"])
    for s in plan["sessions"]:
        day = f"Day {s['day']} · " if s["day"] else ""
        st.markdown(f"#### {day}Session {s['n']} · {s['minutes']} min")
        if s["goal"]:
            st.markdown(s["goal"])
        for item in s["items"]:
            cite = (" [" + ", ".join(str(numbers[c]) for c in item["chunk_ids"]) + "]") if item["chunk_ids"] else ""
            cont = " (continued)" if item.get("continued") else ""
            where = f" · {item['source_name']}" if item.get("source_name") else ""
            st.markdown(f"- {item['title']}{cont} · {item['minutes']} min{where}{cite}")
        if s["questions"]:
            st.caption("Ask the chatbot:")
            for i, question in enumerate(s["questions"]):
                st.button(question, key=f"{key}_s{s['n']}_q{i}", on_click=ask, args=(question,))
        if s["checkpoint"]:
            _checkpoint(plan, s["n"], key, make_quiz)
        if s["break_after"]:
            st.caption(f"Break · {s['break_after']} min")
    if plan["more_time"]:
        with st.expander(f"If you have more time · {len(plan['more_time'])} more section(s)"):
            for m in plan["more_time"][:MORE_TIME_SHOWN]:
                st.markdown(f"- {m['title']}")
            if len(plan["more_time"]) > MORE_TIME_SHOWN:
                st.caption(f"... and {len(plan['more_time']) - MORE_TIME_SHOWN} more.")
    slug = re.sub(r"[^a-z0-9]+", "_", plan["topic"].lower()).strip("_") or "topic"
    st.download_button("Download plan (.md)", study_plan.to_markdown(plan), file_name=f"study_plan_{slug}.md",
                       mime="text/markdown", key=f"{key}_download")
    show_cited_sources(sources, numbering(plan))


def _checkpoint(plan: dict, n: int, key: str, make_quiz: Callable[[str, list[str]], dict]) -> None:
    """A quiz checkpoint: a button, then the quiz card (made once and kept in session state)."""
    state_key = f"{key}_cp{n}"
    made = st.session_state.get(state_key)
    if made is None:
        if st.button("Quiz checkpoint", key=f"{key}_cpbtn{n}"):
            titles, chunk_ids = checkpoint_chunks(plan, n)
            with st.spinner("Writing a checkpoint quiz from these sessions' material..."):
                st.session_state[state_key] = make_quiz(titles, chunk_ids)
            st.rerun()
        return
    if made.get("error"):
        st.error(made["error"])
    else:
        quiz_view.render(made["quiz"], made["sources"], f"{state_key}_quiz")
