"""capstone_streamlit.py
Run: streamlit run capstone_streamlit.py
"""

import os
import uuid

import streamlit as st
from dotenv import load_dotenv

from agent import DOCUMENTS, DOMAIN_DESCRIPTION, DOMAIN_NAME, build_agent

load_dotenv()

# Prefer Streamlit secrets in deployment, fall back to .env locally.
try:
    if "GROQ_API_KEY" in st.secrets and not os.getenv("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

if not os.getenv("GROQ_API_KEY"):
    st.warning(
        "GROQ_API_KEY is missing. Set it in Streamlit Secrets for cloud deployment or in .env for local runs."
    )

st.set_page_config(page_title=DOMAIN_NAME, page_icon="🤖", layout="centered")
st.title(f"🤖 {DOMAIN_NAME}")
st.caption(DOMAIN_DESCRIPTION)


@st.cache_resource
def load_agent():
    return build_agent()


try:
    agent_app, kb_count, active_mode = load_agent()
    if active_mode == "chroma":
        st.success(f"Knowledge base loaded: {kb_count} documents (ChromaDB)")
    else:
        st.warning(f"Knowledge base loaded: {kb_count} documents (in-memory fallback)")
except Exception as e:
    st.error(f"Failed to load agent: {e}")
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]


with st.sidebar:
    st.header("About")
    st.write(DOMAIN_DESCRIPTION)
    st.write(f"Session: {st.session_state.thread_id}")
    st.divider()
    st.write("Topics covered:")
    for topic in [d["topic"] for d in DOCUMENTS]:
        st.write(f"- {topic}")
    if st.button("New conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


if prompt := st.chat_input("Ask something..."):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = {}
            try:
                result = agent_app.invoke({"question": prompt}, config=config)
                answer = result.get("answer", "Sorry, I could not generate an answer.")
            except Exception as e:
                err_text = f"{e.__class__.__name__}: {e}"
                if "authentication" in err_text.lower() or "api_key" in err_text.lower():
                    st.error(
                        "Groq authentication failed. Update GROQ_API_KEY in Streamlit Secrets, then redeploy/restart the app."
                    )
                    answer = (
                        "I cannot answer right now because the LLM key is invalid or missing. "
                        "Please update GROQ_API_KEY in deployment secrets."
                    )
                else:
                    st.error(f"Agent runtime error: {err_text}")
                    answer = "I encountered a runtime error while processing your request. Please try again."
        st.write(answer)
        faith = result.get("faithfulness", 0.0)
        if faith > 0:
            st.caption(f"Faithfulness: {faith:.2f} | Sources: {result.get('sources', [])}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
