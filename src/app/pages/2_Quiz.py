"""
"Quiz" page (feature B): a form around the same quiz function as the chat command "quiz me on <topic>" (RagPipeline.quiz_for_topic
and src/core/quiz.py), so both behave identically. It uses the answer model picked in the chat sidebar (default deepseek-flash).
UI only: no retrieval or prompt logic here.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from config import settings  # noqa: E402
from src.app import quiz_view  # noqa: E402
from src.app.shared import answer_model, load_pipeline, reply_caption, sources_of, usage_caption  # noqa: E402
from src.core import llm, quiz  # noqa: E402


def main() -> None:
    st.set_page_config(page_title=settings.QUIZ_PAGE_TITLE, page_icon="📝")
    st.title(settings.QUIZ_PAGE_TITLE)
    model = answer_model()
    st.caption(f"Three questions from the course materials on any course topic, written by {model} and checked against the "
               "material before they are shown.")
    if llm.is_cloud_model(model):
        st.caption(settings.DEV_CLOUD_PRIVACY_NOTE.format(host=llm.cloud_host(model)))
    with st.form("quiz_topic_form"):
        topic = st.text_input("Topic", placeholder="e.g. star schema, SQL joins, Kafka topics")
        asked = st.form_submit_button("Make a quiz")
    if asked and topic.strip():
        pipeline = load_pipeline()
        result = pipeline.quiz_for_topic(topic.strip(), model=model)
        slot = st.empty()
        if result.quiz_call is None:          # below the relevance threshold: the course fallback, no quiz
            st.session_state.quiz_page = {"message": settings.FALLBACK_MESSAGE}
        else:
            try:
                if llm.is_cloud_model(model):
                    made = result.quiz_call(slot.info)
                else:
                    with st.spinner(settings.QUIZ_WAITING_LOCAL):
                        made = result.quiz_call()
            except (llm.CloudError, quiz.QuizError) as error:
                st.session_state.quiz_page = {"error": str(error)}
            else:
                count = st.session_state.get("quiz_page_count", 0) + 1
                st.session_state.quiz_page_count = count
                st.session_state.quiz_page = {"quiz": made, "sources": sources_of(result.chunks), "key": f"quiz_page_{count}",
                                              "model": model}
        slot.empty()
    shown = st.session_state.get("quiz_page")
    if not shown:
        return
    if shown.get("message"):
        st.info(shown["message"])
    elif shown.get("error"):
        st.error(shown["error"])
    else:
        quiz_view.render(shown["quiz"], shown["sources"], shown["key"])
        st.caption(reply_caption(shown["model"], False))
        if shown["quiz"].get("usage"):
            st.caption(usage_caption(shown["quiz"]["usage"]))


if __name__ == "__main__":       # Streamlit runs a page as __main__
    main()
