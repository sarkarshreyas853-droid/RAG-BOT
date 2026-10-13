"""Streamlit front end. Run with: streamlit run app.py"""
import os

import streamlit as st

from ask import RAG
from ingest import build

DB = "chroma_500"

st.set_page_config(page_title="Document Q&A", page_icon="📄")
st.title("📄 Document Q&A")
st.caption("Ask a question. Answers come only from the loaded documents, with page citations.")


@st.cache_resource(show_spinner="Loading documents (first run builds the index)...")
def load_rag():
    if not os.path.isdir(DB):
        build("data", DB, 500, 50)
    return RAG(DB, k=5)


rag = load_rag()
question = st.text_input("Your question")

if question:
    with st.spinner("Thinking..."):
        answer, chunks = rag.answer(question)
    st.markdown(answer)
    with st.expander("Sources"):
        for c in chunks:
            st.markdown(f"**{c['file']}, page {c['page']}**")
            st.caption(c["text"][:500] + ("..." if len(c["text"]) > 500 else ""))
