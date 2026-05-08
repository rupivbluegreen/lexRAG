from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from rag import OLLAMA_MODEL, TOP_K, Source, ask_stream, retrieve

load_dotenv()

st.set_page_config(page_title="lexRAG", page_icon=None, layout="wide")

st.title("lexRAG")
st.caption(
    "Local-only chat over DORA, NIS2, and the AI Act. "
    "Retrieval: BGE-small + ChromaDB. Generation: Ollama."
)

with st.sidebar:
    st.header("Settings")
    st.text_input("Ollama model", value=OLLAMA_MODEL, disabled=True)
    top_k = st.slider("Top-K excerpts", min_value=1, max_value=15, value=TOP_K, step=1)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources: list[Source]) -> None:
    with st.expander(f"Sources ({len(sources)})"):
        for i, s in enumerate(sources, start=1):
            st.markdown(f"**[{i}] {s.source} — p.{s.page}** · score={s.score:.3f}")
            st.text(s.text)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg["sources"])


query = st.chat_input("Ask about DORA, NIS2, or the AI Act…")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        sources = retrieve(query, k=top_k)
        placeholder = st.empty()
        accumulated = ""
        for token in ask_stream(query, k=top_k):
            accumulated += token
            placeholder.markdown(accumulated)
        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": accumulated, "sources": sources}
    )
