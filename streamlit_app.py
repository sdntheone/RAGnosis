import streamlit as st
import requests

# API_URL = "http://127.0.0.1:8000/chat"
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/chat")

st.set_page_config(
    page_title="RAGnosis",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAGnosis - ML Knowledge Assistant")

# Sidebar
st.sidebar.header("Settings")

mode = st.sidebar.selectbox(
    "Response Mode",
    ["default", "interview", "beginner"]
)

k = st.sidebar.slider("Top-K Retrieval", 1, 10, 3)

st.sidebar.markdown("---")
st.sidebar.info("Ask questions about Machine Learning concepts 📚")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
query = st.chat_input("Ask something about ML...")

if query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    API_URL,
                    json={
                        "query": query,
                        "mode": mode,
                        "k": k
                    }
                )

                result = response.json()

                answer = result.get("answer", "No response")

                st.markdown(answer)

                st.caption(f"⏱ Response time: {result.get('latency', 0)}s")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error("Error connecting to API")