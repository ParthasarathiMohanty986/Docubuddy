import hashlib
import os

import requests
import streamlit as st

from rag import OllamaEmbeddings, build_index, generate_answer, retrieve

st.set_page_config(page_title="PDF Chat", page_icon="📖")
st.title("📖 Chat with your PDF")
st.caption("Upload a PDF, ask a question, and check the supporting pages.")


def embedding_model():
    return OllamaEmbeddings(base_url)


def clear_document():
    index = st.session_state.pop("index", None)
    if index:
        index[0].delete_collection(index[1].name)
    st.session_state.pop("document_id", None)
    st.session_state["messages"] = []


with st.sidebar:
    st.header("Setup")
    mode = st.selectbox("Answer mode", ["Local Ollama answers", "Search passages only"])
    base_url = st.text_input("Ollama URL", os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    model_name = st.text_input("Chat model", os.getenv("LLM_MODEL", "qwen3:1.7b"))
    st.caption("Embeddings: embeddinggemma. Your PDF text and questions go to this Ollama server. Keep the localhost URL for local processing.")
    uploaded = st.file_uploader("Choose one PDF (max 20 MB)", type=["pdf"])
    st.caption("Text PDFs only • up to 200 pages. Ollama must be running with both models installed.")
    if st.button("Clear chat"):
        st.session_state["messages"] = []

st.session_state.setdefault("messages", [])
if uploaded is None:
    if "index" in st.session_state:
        clear_document()
    st.info("Upload a PDF in the sidebar to get started.")
    st.stop()

data = uploaded.getvalue()
document_id = hashlib.sha256(data + base_url.encode()).hexdigest()
if document_id != st.session_state.get("document_id"):
    clear_document()
    if len(data) > 20 * 1024 * 1024:
        st.error("Please upload a PDF smaller than 20 MB.")
        st.stop()
    try:
        with st.spinner("Loading the embedding model and indexing your PDF…"):
            st.session_state["index"] = build_index(data, embedding_model())
            st.session_state["document_id"] = document_id
    except Exception as exc:
        st.error(f"Could not index this PDF: {exc}")
        st.stop()

_, collection, count, skipped = st.session_state["index"]
st.success(f"{uploaded.name} · {count} searchable passages")
if skipped:
    st.warning(f"{skipped} page(s) had no extractable text and were skipped.")
st.caption("Ask each question in full; previous messages are displayed but are not used for retrieval.")


def show_sources(sources):
    with st.expander("Retrieved passages — check the answer here"):
        for source in sources:
            st.markdown(f"**Page {source['page']}**")
            st.text(source["text"])


for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"])

if question := st.chat_input("Ask about your PDF", max_chars=2000):
    if not base_url.strip() or not model_name.strip():
        st.warning("Enter the Ollama URL and model name first.")
        st.stop()
    with st.chat_message("user"):
        st.write(question)
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("assistant"):
        try:
            with st.spinner("Finding relevant passages…"):
                passages = retrieve(collection, embedding_model(), question)
            if mode == "Search passages only":
                answer = "Here are the closest matching passages. Select Local Ollama answers to generate an answer. Matches may not contain an answer."
            else:
                with st.spinner("Writing an answer…"):
                    answer = generate_answer(question, passages, base_url.strip(), model_name.strip())
            st.markdown(answer)
            show_sources(passages)
            st.session_state["messages"].append({"role": "assistant", "content": answer, "sources": passages})
        except requests.RequestException:
            st.error("Could not reach Ollama or the request failed. Check the server URL, installed model name, and whether Ollama is running.")
        except Exception:
            st.error("Could not complete this question. Check your model configuration or try reloading the PDF.")
