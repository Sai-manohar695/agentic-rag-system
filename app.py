"""
Agentic RAG System — Streamlit UI
"""

import streamlit as st
import time
from agent import run_agent
from tools.rag_tool import ingest_documents

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title = "Agentic RAG System",
    page_icon  = "🤖",
    layout     = "wide"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f172a; }
    .stApp { background-color: #0f172a; }

    .title-box {
        background: linear-gradient(135deg, #1e3a5f, #0f172a);
        border: 1px solid #3b82f6;
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 24px;
    }
    .title-box h1 {
        color: #f1f5f9;
        font-size: 2rem;
        margin: 0;
    }
    .title-box p {
        color: #94a3b8;
        margin: 6px 0 0 0;
        font-size: 0.95rem;
    }

    .tool-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 3px;
    }
    .badge-rag       { background:#1e3a5f; color:#60a5fa; border:1px solid #3b82f6; }
    .badge-wiki      { background:#1a3a2a; color:#4ade80; border:1px solid #22c55e; }
    .badge-arxiv     { background:#3a1a2a; color:#f472b6; border:1px solid #ec4899; }
    .badge-web       { background:#3a2a1a; color:#fb923c; border:1px solid #f97316; }
    .badge-calc      { background:#2a1a3a; color:#c084fc; border:1px solid #a855f7; }

    .answer-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 20px;
        color: #f1f5f9;
        line-height: 1.7;
        margin: 12px 0;
    }
    .trace-box {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 14px;
        font-family: monospace;
        font-size: 0.82rem;
        color: #94a3b8;
        margin: 6px 0;
    }
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-card h3 {
        color: #3b82f6;
        font-size: 1.6rem;
        margin: 0;
    }
    .metric-card p {
        color: #94a3b8;
        font-size: 0.8rem;
        margin: 4px 0 0 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────
if "messages"      not in st.session_state:
    st.session_state.messages      = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "tools_used"    not in st.session_state:
    st.session_state.tools_used    = []

# ── Header ────────────────────────────────────────────────
st.markdown("""
<div class="title-box">
    <h1>🤖 Agentic RAG System</h1>
    <p>Multi-Tool Orchestration · Adaptive Query Routing ·
       Powered by Llama-3.3-70b via Groq · Pinecone Vector DB</p>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────
left, right = st.columns([2, 1])

with right:
    # ── Metrics ───────────────────────────────────────────
    st.markdown("### 📊 Session Stats")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{st.session_state.total_queries}</h3>
            <p>Queries</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        unique_tools = len(set(st.session_state.tools_used))
        st.markdown(f"""
        <div class="metric-card">
            <h3>{unique_tools}</h3>
            <p>Tools Used</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Tool Guide ────────────────────────────────────────
    st.markdown("### 🛠️ Available Tools")
    tools_info = [
        ("🔍", "RAG Search",   "badge-rag",
         "Internal knowledge base"),
        ("📖", "Wikipedia",    "badge-wiki",
         "Encyclopedic facts"),
        ("📄", "ArXiv",        "badge-arxiv",
         "Research papers"),
        ("🌐", "Web Search",   "badge-web",
         "Current web info"),
        ("🧮", "Calculator",   "badge-calc",
         "Math expressions"),
    ]
    for icon, name, badge, desc in tools_info:
        st.markdown(
            f'<span class="tool-badge {badge}">'
            f'{icon} {name}</span> '
            f'<small style="color:#64748b">{desc}</small><br>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ── Ingest Documents ──────────────────────────────────
    st.markdown("### 📥 Ingest Documents")
    doc_text = st.text_area(
        "Paste document text:",
        height      = 120,
        placeholder = "Paste any text to add to knowledge base..."
    )
    doc_source = st.text_input(
        "Source label:",
        placeholder = "e.g. Company Policy Doc"
    )
    if st.button("📤 Ingest", use_container_width=True):
        if doc_text.strip():
            with st.spinner("Ingesting..."):
                n = ingest_documents(
                    [doc_text],
                    [{"source": doc_source or "User Upload"}]
                )
            st.success(f"✅ Ingested {n} chunks")
        else:
            st.warning("Please paste some text first")

    st.markdown("---")

    # ── Example Queries ───────────────────────────────────
    st.markdown("### 💡 Example Queries")
    examples = [
        "What is RAG and how does it work?",
        "Find papers on LLM hallucination",
        "What is 23% of 150000?",
        "Latest news on AI agents",
        "Who invented the transformer architecture?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True,
                     key=f"ex_{ex[:20]}"):
            st.session_state.example_query = ex

with left:
    # ── Chat History ──────────────────────────────────────
    st.markdown("### 💬 Chat")

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div style="text-align:right;margin:8px 0">'
                    f'<span style="background:#1e3a5f;color:#f1f5f9;'
                    f'padding:10px 16px;border-radius:18px 18px 4px 18px;'
                    f'display:inline-block;max-width:80%">'
                    f'👤 {msg["content"]}</span></div>',
                    unsafe_allow_html=True
                )
            else:
                # Tool badges
                badges = ""
                tool_colors = {
                    "RAG_Search"      : "badge-rag",
                    "Wikipedia_Search": "badge-wiki",
                    "ArXiv_Search"    : "badge-arxiv",
                    "Web_Search"      : "badge-web",
                    "Calculator"      : "badge-calc"
                }
                tool_icons = {
                    "RAG_Search"      : "🔍",
                    "Wikipedia_Search": "📖",
                    "ArXiv_Search"    : "📄",
                    "Web_Search"      : "🌐",
                    "Calculator"      : "🧮"
                }
                for tool in msg.get("tools_used", []):
                    badge = tool_colors.get(tool, "badge-rag")
                    icon  = tool_icons.get(tool, "🔧")
                    label = tool.replace("_", " ")
                    badges += (
                        f'<span class="tool-badge {badge}">'
                        f'{icon} {label}</span>'
                    )

                st.markdown(
                    f'<div style="margin:8px 0">{badges}</div>'
                    f'<div class="answer-box">'
                    f'🤖 {msg["content"]}</div>',
                    unsafe_allow_html=True
                )

                # Tool trace expander
                if msg.get("tool_trace"):
                    with st.expander("🔎 Tool Trace"):
                        for step in msg["tool_trace"]:
                            st.markdown(
                                f'<div class="trace-box">'
                                f'<b style="color:#3b82f6">'
                                f'Tool:</b> {step["tool"]}<br>'
                                f'<b style="color:#3b82f6">'
                                f'Input:</b> {step["input"]}<br>'
                                f'<b style="color:#3b82f6">'
                                f'Output:</b> {step["output"][:200]}...'
                                f'</div>',
                                unsafe_allow_html=True
                            )

    # ── Input ─────────────────────────────────────────────
    default_q = st.session_state.pop("example_query", "")
    query = st.chat_input(
        "Ask anything — I'll route to the best tool..."
    )

    if query or default_q:
        q = query or default_q
        st.session_state.messages.append({
            "role": "user", "content": q})
        st.session_state.total_queries += 1

        with st.spinner("🤔 Agent thinking..."):
            start  = time.time()
            result = run_agent(q)
            elapsed = time.time() - start

        st.session_state.tools_used.extend(
            result.get("tools_used", []))
        st.session_state.messages.append({
            "role"       : "assistant",
            "content"    : result["answer"],
            "tools_used" : result["tools_used"],
            "tool_trace" : result["tool_trace"]
        })
        st.rerun()