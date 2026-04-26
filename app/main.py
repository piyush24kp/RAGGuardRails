"""
Streamlit frontend for the RAG + RBAC + Guardrails chatbot.
Run with:  streamlit run app/main.py  (from the project root)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import streamlit as st

from retrieval.rbac import USERS, get_allowed_departments
from retrieval.rag_chain import answer
from guardrails.scope_check import check_scope
from guardrails.pii_filter import redact

_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|good\s*(morning|afternoon|evening)|howdy|greetings|what'?s up|how are you)\W*\s*$",
    re.IGNORECASE,
)


def _greeting_response(display_name: str) -> str:
    return f"Hello, {display_name}! How can I help you today? You can ask me about HR policies, financial reports, company guidelines, or anything else within your access."

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSolve Internal Assistant",
    page_icon="🏦",
    layout="wide",
)

# ── Session state defaults ────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user" not in st.session_state:
    st.session_state.user = None

# ── Sidebar: login ────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏦 FinSolve AI Assistant")
    st.divider()

    selected = st.selectbox(
        "Login as",
        options=list(USERS.keys()),
        format_func=lambda u: USERS[u]["display_name"],
    )

    if st.button("Login / Switch User", use_container_width=True):
        st.session_state.user = selected
        st.session_state.messages = []  # clear chat on login
        st.rerun()

    if st.session_state.user:
        u = USERS[st.session_state.user]
        st.success(f"Logged in as **{u['display_name']}**")
        allowed = get_allowed_departments(u["role"])
        st.caption(f"Access: `{'`, `'.join(allowed)}`")

    st.divider()
    st.caption("Demo project — RAG + RBAC + Guardrails")

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("Internal Company Chatbot")

if not st.session_state.user:
    st.info("Select a user in the sidebar and click **Login** to start.")
    st.stop()

user_info = USERS[st.session_state.user]
role = user_info["role"]

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources", expanded=False):
                for src in msg["sources"]:
                    st.markdown(f"- `{src}`")
        if msg.get("guardrail"):
            st.warning(msg["guardrail"])

# ── Chat input ────────────────────────────────────────────────────────────────
query = st.chat_input("Ask a question about company data...")

if query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        # 1. Greeting check — respond directly, skip RAG and scope check entirely
        if _GREETING_RE.match(query):
            greeting_reply = _greeting_response(user_info["display_name"])
            st.markdown(greeting_reply)
            st.session_state.messages.append({"role": "assistant", "content": greeting_reply})
            st.stop()

        # 2. Scope check
        in_scope, scope_reason = check_scope(query)
        if not in_scope:
            st.warning(scope_reason)
            st.session_state.messages.append({
                "role": "assistant",
                "content": "",
                "guardrail": scope_reason,
            })
            st.stop()

        # 3. Retrieve + generate
        with st.spinner("Thinking..."):
            result = answer(query, role)

        raw_answer = result["answer"]
        sources = result["sources"]

        # 4. PII filter on output
        clean_answer, pii_found = redact(raw_answer)

        # 5. Display
        st.markdown(clean_answer)
        if sources:
            with st.expander("Sources", expanded=False):
                for src in sources:
                    st.markdown(f"- `{src}`")
        if pii_found:
            st.caption(f"ℹ️ Some sensitive fields were redacted: {', '.join(pii_found)}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": clean_answer,
            "sources": sources,
        })
