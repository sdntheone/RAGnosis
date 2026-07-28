"""
streamlit_app_v2.py

New UI, separate from streamlit_app.py (which is untouched and keeps
talking to the original /chat endpoint against the global book index).

This app talks only to the api_v2 routes:
  - upload_routes.py    (create session, upload, status polling)
  - document_routes.py  (list, delete, clear, rebuild)
  - chat_stream_routes.py (streaming chat with guardrails + sources)
  - observability routes (added in the next file, observability_routes.py --
    this UI's dashboard tab is written against that endpoint's expected
    shape now, ready to work once that file exists)

Run with: streamlit run streamlit_app_v2.py
Requires the FastAPI app (main.py) running separately, with api_v2 routers
included (final wiring step, after all api_v2 files exist).
"""

import json
import time

import requests
import streamlit as st

API_BASE = "http://localhost:8000/api/v2"

st.set_page_config(page_title="RAGnosis v2", layout="wide")


# ----------------------------------------------------------------------
# Session bootstrap
# ----------------------------------------------------------------------

def _init_session():
    if "session_id" not in st.session_state:
        resp = requests.post(f"{API_BASE}/sessions")
        resp.raise_for_status()
        st.session_state.session_id = resp.json()["session_id"]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of {"role", "content", "sources", "confidence"}


_init_session()
session_id = st.session_state.session_id


# ----------------------------------------------------------------------
# Sidebar: upload + document management + filters
# ----------------------------------------------------------------------

with st.sidebar:
    st.header("Documents")

    uploaded_files = st.file_uploader(
        "Drag & drop files",
        type=["pdf", "docx", "pptx", "xlsx", "xlsm", "csv", "tsv", "txt", "md",
              "png", "jpg", "jpeg", "webp", "bmp", "tiff"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Upload & Index"):
        files_payload = [
            ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
            for f in uploaded_files
        ]
        resp = requests.post(f"{API_BASE}/sessions/{session_id}/upload", files=files_payload)
        if resp.ok:
            queued = resp.json()["files"]
            progress_placeholder = st.empty()
            doc_ids = [f["doc_id"] for f in queued]

            with progress_placeholder.container():
                progress_bar = st.progress(0)
                status_text = st.empty()

                done = set()
                while len(done) < len(doc_ids):
                    completed = 0
                    lines = []
                    for doc_id, meta in zip(doc_ids, queued):
                        status_resp = requests.get(
                            f"{API_BASE}/sessions/{session_id}/documents/{doc_id}/status"
                        )
                        if not status_resp.ok:
                            continue
                        status = status_resp.json()
                        if status["status"] in ("ready", "failed"):
                            done.add(doc_id)
                            completed += 1
                        lines.append(f"{status['filename']}: {status['status']}")
                    progress_bar.progress(completed / len(doc_ids) if doc_ids else 1.0)
                    status_text.write("\n".join(lines))
                    if len(done) < len(doc_ids):
                        time.sleep(1)

            st.success("Indexing complete.")
            st.rerun()
        else:
            st.error(f"Upload failed: {resp.text}")

    st.divider()

    docs_resp = requests.get(f"{API_BASE}/sessions/{session_id}/documents")
    documents = docs_resp.json()["documents"] if docs_resp.ok else []

    if documents:
        for doc in documents:
            with st.expander(f"{doc['filename']} ({doc['status']})"):
                st.write(f"Type: {doc['file_type']}")
                st.write(f"Chunks: {doc['chunk_count']}")
                if doc["block_counts"]:
                    st.write("Content:", doc["block_counts"])
                if doc["warnings"]:
                    st.warning("\n".join(doc["warnings"]))
                if doc["error_message"]:
                    st.error(doc["error_message"])
                if st.button("Delete", key=f"del_{doc['doc_id']}"):
                    requests.delete(f"{API_BASE}/sessions/{session_id}/documents/{doc['doc_id']}")
                    st.rerun()
    else:
        st.caption("No documents uploaded yet.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear session"):
            requests.delete(f"{API_BASE}/sessions/{session_id}")
            st.session_state.chat_history = []
            st.rerun()
    with col2:
        if st.button("Rebuild index"):
            requests.post(f"{API_BASE}/sessions/{session_id}/rebuild")
            st.rerun()

    st.divider()
    st.subheader("Filters")
    filter_file_types = st.multiselect(
        "File types", options=sorted({d["file_type"] for d in documents}) if documents else []
    )
    filter_chunk_types = st.multiselect(
        "Content types", options=["text", "table", "image", "ocr_text", "image_caption_only"]
    )
    filter_has_media = st.selectbox("Media only", options=["any", "media only", "text only"])

    st.divider()
    mode = st.selectbox("Answer mode", options=["default", "interview", "beginner"])


# ----------------------------------------------------------------------
# Main area: tabs for Chat and Observability
# ----------------------------------------------------------------------

chat_tab, dashboard_tab = st.tabs(["Chat", "Observability"])

with chat_tab:
    st.title("RAGnosis — Document Q&A")

    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("sources"):
                with st.expander(f"Sources (confidence: {turn.get('confidence', 'n/a')})"):
                    for src in turn["sources"]:
                        page = f", page {src['page_number']}" if src.get("page_number") is not None else ""
                        st.caption(f"{src['source']}{page} — {src['chunk_type']}")

    query = st.chat_input("Ask a question about your documents...")

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        payload = {
            "query": query,
            "mode": mode,
            "file_types": filter_file_types or None,
            "chunk_types": filter_chunk_types or None,
            "has_media": (
                True if filter_has_media == "media only"
                else False if filter_has_media == "text only"
                else None
            ),
        }

        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            full_answer = ""
            sources = []
            confidence = None

            try:
                with requests.post(
                    f"{API_BASE}/sessions/{session_id}/chat/stream",
                    json=payload,
                    stream=True,
                    timeout=120,
                ) as resp:
                    for line in resp.iter_lines(decode_unicode=True):
                        if not line or not line.startswith("data: "):
                            continue
                        event = json.loads(line[len("data: "):])

                        if event["type"] == "token":
                            full_answer += event["text"]
                            answer_placeholder.markdown(full_answer + "▌")
                        elif event["type"] == "done":
                            sources = event.get("sources", [])
                            confidence = event.get("confidence")
                        elif event["type"] in ("blocked", "no_info"):
                            full_answer = event["message"]
                        elif event["type"] == "error":
                            full_answer = f"⚠️ {event['message']}"

                answer_placeholder.markdown(full_answer)

                if sources:
                    with st.expander(f"Sources (confidence: {confidence or 'n/a'})"):
                        for src in sources:
                            page = f", page {src['page_number']}" if src.get("page_number") is not None else ""
                            st.caption(f"{src['source']}{page} — {src['chunk_type']}")

            except requests.RequestException as e:
                full_answer = f"⚠️ Request failed: {e}"
                answer_placeholder.markdown(full_answer)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": full_answer,
            "sources": sources,
            "confidence": confidence,
        })


with dashboard_tab:
    st.title("Observability")

    if st.button("Refresh"):
        st.rerun()

    stats_resp = requests.get(f"{API_BASE}/sessions/{session_id}/observability/summary")
    if stats_resp.ok:
        stats = stats_resp.json()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Requests", stats.get("request_count", 0))
        c2.metric("Avg latency (ms)", stats.get("avg_total_latency_ms"))
        c3.metric("Avg tokens", stats.get("avg_total_tokens"))
        c4.metric("Guardrail pass rate", stats.get("guardrail_pass_rate"))

        c5, c6 = st.columns(2)
        c5.metric("Avg similarity score", stats.get("avg_similarity_score"))
        c6.metric("Avg groundedness score", stats.get("avg_groundedness_score"))

        if stats.get("avg_stage_latencies_ms"):
            st.subheader("Latency by stage (ms)")
            st.bar_chart(stats["avg_stage_latencies_ms"])
    else:
        st.info("No observability data yet — send a chat message first.")

    st.subheader("Recent requests")
    traces_resp = requests.get(f"{API_BASE}/sessions/{session_id}/observability/traces")
    if traces_resp.ok:
        traces = traces_resp.json().get("traces", [])
        for t in traces:
            with st.expander(f"{t['query'][:80]} — {t.get('confidence', 'n/a')} confidence"):
                st.write(f"Total latency: {t['total_latency_ms']} ms")
                st.write("Stage latencies:", t["stage_latencies_ms"])
                st.write(f"Tokens: {t.get('total_tokens')}")
                st.write("Guardrails:", t["guardrail_outcomes"])
                st.write("Sources:", [c["source"] for c in t["retrieved_chunks"]])
                if t.get("langsmith_url"):
                    st.markdown(f"[View in LangSmith]({t['langsmith_url']})")
    else:
        st.caption("No recent requests.")