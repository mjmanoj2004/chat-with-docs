"""Streamlit UI: upload documents, chat with RAG (calls FastAPI)."""

import os
import streamlit as st
import httpx

# Backend URL (env or default)
BACKEND_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
API = f"{BACKEND_URL.rstrip('/')}/api"

st.set_page_config(page_title="Chat With Your Docs", page_icon="📄", layout="centered")
st.title("Chat With Your Docs")
st.caption("Upload PDF/TXT/DOC, then ask questions. Hybrid search (semantic + BM25), Ollama LLM.")

# Session state: messages (conversation memory per session)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources" not in st.session_state:
    st.session_state.sources = []

# Sidebar: upload
with st.sidebar:
    st.header("Upload")
    uploaded = st.file_uploader(
        "Choose one or more files",
        type=["pdf", "txt", "md", "doc", "docx"],
        accept_multiple_files=True,
    )
    if uploaded:
        with st.spinner("Indexing..."):
            try:
                # Send all files in one request (backend accepts list "files")
                file_parts = [
                    ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
                    for f in uploaded
                ]
                with httpx.Client(timeout=120.0) as client:
                    r = client.post(f"{API}/upload", files=file_parts)
                r.raise_for_status()
                data = r.json()
                if "results" in data:
                    total = data.get("total_chunks", 0)
                    st.success(f"Indexed **{len(data['results'])}** file(s), **{total}** chunks total.")
                else:
                    st.success(f"Indexed **{data['filename']}** ({data['chunks_added']} chunks).")
            except Exception as e:
                st.error(f"Upload failed: {e}")

    st.divider()
    show_sources = st.checkbox("Show source references", value=True)

# Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources") and show_sources:
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.write(f"**{s.get('filename', '')}** (page: {s.get('page', '')})")
                    st.caption(s.get("snippet", "")[:300])

if prompt := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with httpx.Client(timeout=120.0) as client:
                r = client.post(f"{API}/ask", json={"question": prompt})
            r.raise_for_status()
            data = r.json()
            answer = data.get("answer", "")
            sources = data.get("sources", [])
            st.markdown(answer)
            if show_sources and sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.write(f"**{s.get('filename', '')}** (page: {s.get('page', '')})")
                        st.caption((s.get("snippet") or "")[:300])
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })
        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.messages.append({"role": "assistant", "content": str(e), "sources": []})
