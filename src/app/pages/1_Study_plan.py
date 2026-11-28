"""
"Study plan" page (feature A): a form around the same planner as the chat ("I have 3 hours to study data warehousing";
RagPipeline.plan_for and src/core/study_plan.py), so both behave identically. It uses the answer model picked in the chat sidebar
(default deepseek-flash). A suggested question opens the chat and is sent there. UI only.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from config import settings  # noqa: E402
from src.app import plan_view  # noqa: E402
from src.app.shared import (answer_model, ask_in_chat, checkpoint_quiz_maker, load_pipeline, reply_caption,  # noqa: E402
                            sources_of)
from src.core import llm  # noqa: E402
from src.core.study_plan import PlanRequest  # noqa: E402

UNITS = {"minutes": 1, "hours": 60, "days": None}


def request_from_form(topic: str, amount: float, unit: str, hours_per_day: float) -> PlanRequest:
    """The same PlanRequest the chat parser builds: days mode uses the hours a day given (default PLAN_HOURS_PER_DAY)."""
    if unit == "days":
        per_day = round(hours_per_day * 60)
        days = max(1, round(amount))
        return PlanRequest(topic=topic, minutes=days * per_day, mode="days", days=days, minutes_per_day=per_day,
                           per_day_given=hours_per_day != settings.PLAN_HOURS_PER_DAY)
    return PlanRequest(topic=topic, minutes=round(amount * UNITS[unit]), mode="hours")


def main() -> None:
    st.set_page_config(page_title=settings.PLAN_PAGE_TITLE, page_icon="🗓️")
    st.title(settings.PLAN_PAGE_TITLE)
    model = answer_model()
    st.caption(f"A time-boxed plan from the course materials: sessions of up to {settings.PLAN_SESSION_MINUTES} minutes, sub-topics "
               f"named after the real sections, questions to ask the chatbot and quiz checkpoints. Goals and questions by {model}.")
    if llm.is_cloud_model(model):
        st.caption(settings.DEV_CLOUD_PRIVACY_NOTE.format(host=llm.cloud_host(model)))
    with st.form("plan_form"):
        topic = st.text_input("Topic", placeholder="e.g. data warehousing, SQL, ETL and ELT")
        left, middle, right = st.columns(3)
        amount = left.number_input("Time", min_value=0.0, value=3.0, step=0.5)
        unit = middle.selectbox("Unit", list(UNITS), index=1)
        hours_per_day = right.number_input("Hours a day (days only)", min_value=0.5, max_value=12.0,
                                           value=float(settings.PLAN_HOURS_PER_DAY), step=0.5)
        asked = st.form_submit_button("Make a plan")
    if asked and topic.strip():
        result = load_pipeline().plan_for(request_from_form(topic.strip(), amount, unit, hours_per_day), model=model)
        slot = st.empty()
        if result.plan_call is None:                   # missing input, limits, the whole field or not in the course
            st.session_state.plan_page = {"message": "".join(result.answer_stream)}
        else:
            if llm.is_cloud_model(model):
                plan = result.plan_call(slot.info)
            else:
                with st.spinner(settings.PLAN_WAITING_LOCAL):
                    plan = result.plan_call()
            count = st.session_state.get("plan_page_count", 0) + 1
            st.session_state.plan_page_count = count
            st.session_state.plan_page = {"plan": plan, "sources": sources_of(result.chunks), "key": f"plan_page_{count}",
                                          "model": model}
        slot.empty()
    shown = st.session_state.get("plan_page")
    if shown:
        if shown.get("message"):
            st.info(shown["message"])
        else:
            plan_view.render(shown["plan"], shown["sources"], shown["key"], checkpoint_quiz_maker(shown["model"]), ask_in_chat)
            st.caption(reply_caption(shown["model"], False))
    if st.session_state.get("pending_question"):       # a suggested question was clicked: answer it in the chat
        st.switch_page("app.py")


if __name__ == "__main__":       # Streamlit runs a page as __main__
    main()
