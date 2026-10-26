"""
Streamlit chat interface (DEV PROTOTYPE).

A minimal page to talk to the knowledge base: one text box, the chat history, and the sources behind each
answer. All retrieval and generation happens in src/core (rag.py); this file only displays things.

Launch from the project root:  venv\\Scripts\\streamlit run src\\app\\app.py
"""

import sys
from pathlib import Path

# Make the project root importable (config/ and src/) when Streamlit runs this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from config import settings
from src.core import llm
from src.core.rag import RagPipeline

st.set_page_config(page_title="Data Engineering Course Assistant", page_icon="💬")


@st.cache_resource(show_spinner="Loading the knowledge base (embedding model + search index)...")
def load_pipeline() -> RagPipeline:
    """Load the RAG pipeline once per server session (the embedding model takes a few seconds)."""
    return RagPipeline()


def show_sources(sources: list[dict], used_fallback: bool) -> None:
    """Show the retrieved chunks behind an answer in a collapsible box."""
    label = "Closest course material found (not used: below the relevance threshold)" if used_fallback else "Sources"
    with st.expander(f"{label} · {len(sources)} chunks"):
        for s in sources:
            where = f"page {s['page']}" if s["page"] else (s["file"] or "")
            st.markdown(f"**[{s['chunk_id']}]** {s['source_name']} · {s['section']} · {where} · "
                        f"similarity {s['score']:.2f}")
            st.caption(s["text"][:300] + ("..." if len(s["text"]) > 300 else ""))


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.subheader("Dev prototype")
    backend_ok, backend_message = llm.check_backend()
    (st.success if backend_ok else st.error)(backend_message)
    st.caption(f"Local model: {settings.DEV_LLM_MODEL} (Ollama) · retrieval: {settings.EMBEDDING_MODEL} · "
               f"top {settings.TOP_K} chunks · fallback below similarity {settings.SIMILARITY_THRESHOLD}")
    st.caption("Each question is answered on its own; earlier messages are not sent to the model.")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

st.title("Data Engineering Course Assistant")
st.caption("Ask about SQL, ETL/ELT, pipelines, data cleaning, processing, quality, warehousing or big data. "
           "Answers come only from the course knowledge base and cite the chunks they use.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------- history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"], message["used_fallback"])

# ---------------------------------------------------------------- new question
question = st.chat_input("Type your question...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not backend_ok:
            st.error(backend_message)
        else:
            pipeline = load_pipeline()
            with st.spinner("Searching the course materials..."):
                result = pipeline.answer(question)
            answer = st.write_stream(result.answer_stream)
            sources = [{"chunk_id": c.chunk_id, "source_name": c.source_name, "section": c.section,
                        "page": c.page, "file": c.file, "score": c.score, "text": c.text} for c in result.chunks]
            if sources:                       # small talk / unclear input skips retrieval: nothing to show
                show_sources(sources, result.used_fallback)
            st.session_state.messages.append({"role": "assistant", "content": answer,
                                              "sources": sources, "used_fallback": result.used_fallback})
