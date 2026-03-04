import streamlit as st
import threading
import tempfile
import uuid
import re
import os
import time
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, AIMessageChunk
from streamlit.runtime.scriptrunner import add_script_run_ctx

# Import your backend components
from Langgraph_RAG_backend import chatbot, retrieve_all_threads, ingest_pdf, clear_all_history

# **************************************** Page Config & Styling **********************
st.set_page_config(page_title="AI Research Agent", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    
    section[data-testid="stSidebar"] {
        background-color: #f0f2f6;
        border-right: 1px solid #e0e0e0;
    }

    /* 1. Make the sidebar a flex container with a fixed height */
    [data-testid="stSidebarUserContent"] {
        display: flex;
        flex-direction: column;
        height: 100vh;
    }

    /* 2. Target the TOP section (Usage/Knowledge Base) to NOT scroll */
    .usage-header {
        flex-shrink: 0; 
        padding-bottom: 10px;
    }

    /* 3. Target the BOTTOM section (History) to GROW and SCROLL */
    .history-container {
        flex-grow: 1;
        overflow-y: auto; 
        border-top: 1px solid #e0e0e0;
        padding-top: 10px;
        padding-right: 10px; 
        scrollbar-width: thin;
    }

    .stButton>button {
        border-radius: 10px;
        height: 3em;
        transition: all 0.2s ease-in-out;
    }

    .stStatus { border-radius: 15px; border: 1px solid #f0f2f6; }
    
    .perf-caption {
        font-size: 0.75rem;
        color: #9ca3af;
        margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# **************************************** Session State Setup ************************
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = str(uuid.uuid4())

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'chat_threads' not in st.session_state:
    try:
        st.session_state['chat_threads'] = retrieve_all_threads()
    except Exception:
        st.session_state['chat_threads'] = []

# **************************************** Sidebar UI *********************************
with st.sidebar:
    st.title("🧬AI Knowledge RAG")
    st.caption("Powered by LangGraph & Groq")
    st.divider()
    
    st.markdown('<div class="usage-header">', unsafe_allow_html=True)
    
    st.subheader("📁 Knowledge Base")
    current_tid = st.session_state.get('thread_id')
    
    if st.session_state.get(f"ingested_{current_tid}"):
        st.success("✅ PDF Active for this chat")
        if st.button("🗑️ Remove PDF"):
            st.session_state[f"ingested_{current_tid}"] = False
            st.rerun()
    else:
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            def bg_ingest(path, tid):
                ingest_pdf(path, tid)
                st.session_state[f"ingested_{tid}"] = True
                if os.path.exists(path): os.remove(path)

            t = threading.Thread(target=bg_ingest, args=(tmp_path, current_tid))
            add_script_run_ctx(t)
            t.start()
            st.toast("Processing PDF in background...", icon="⚙️")

    st.divider()

    st.subheader("📊 Session Usage")

    total_tokens_session = sum(
        msg.get('performance', {}).get('tokens', 0) 
        for msg in st.session_state.get('message_history', []) 
        if msg.get('role') == 'assistant'
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Tokens", f"{total_tokens_session:,}")
    with col2:
        estimated_cost = (total_tokens_session / 1_000_000) * 0.15
        st.metric("Est. Cost", f"${estimated_cost:.4f}")

    st.caption("⚠️ Totals reflect current thread history only.")
    
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        st.session_state['thread_id'] = str(uuid.uuid4())
        st.session_state['message_history'] = []
        st.rerun()

    if st.button("🗑️ Delete History", use_container_width=True):
        if clear_all_history():
            st.session_state['thread_id'] = str(uuid.uuid4())
            st.session_state['message_history'] = []
            st.toast("Database cleared successfully!", icon="🗑️")
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True) 
    st.divider()

    st.subheader("📜 History")
    st.markdown('<div class="history-container">', unsafe_allow_html=True)

    all_threads = retrieve_all_threads()
    for thread_info in all_threads:
        tid = thread_info['id'] 
        title = thread_info.get('title', f"Chat: {tid[:8]}") 
        
        if st.button(f"💬 {title}", key=f"btn_{tid}", use_container_width=True):
            st.session_state['thread_id'] = tid
            state = chatbot.get_state(config={"configurable": {"thread_id": tid}})
            msgs = state.values.get("messages", [])
            st.session_state['message_history'] = [
                {
                    "role": "user" if isinstance(m, HumanMessage) else "assistant", 
                    "content": m.content,
                    "performance": m.additional_kwargs.get("performance", {}) 
                } 
                for m in msgs if m.content and not isinstance(m, (ToolMessage, AIMessageChunk))
            ]
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True) 

# **************************************** Main Chat UI *******************************

st.title("AI Knowledge RAG")
st.info(f"Active Thread: {st.session_state['thread_id']}")

# 1. Render History
for msg in st.session_state['message_history']:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])
        if msg.get('performance'):
            p = msg['performance']
            if p.get('tokens', 0) > 0 or p.get('active_model'):
                model_str = f"🧠 {p.get('active_model', 'AI')} | " if p.get('active_model') else ""
                st.markdown(
                    f"<div class='perf-caption'>{model_str}⏱️ {p['latency']} | 🪙 {p['tokens']} tokens</div>", 
                    unsafe_allow_html=True
                )

# 2. Chat Input
user_query = st.chat_input("Ask a question about your documents or the web...")

if user_query:
    ui_start_time = time.time() 

    st.session_state['message_history'].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        placeholder = st.empty() 
        full_response = ""

        with st.status("🔍 Processing...", expanded=True) as status:
            for msg, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_query)]},
                config={"configurable": {"thread_id": st.session_state['thread_id']}},
                stream_mode="messages"
            ):
                if msg.type == "tool":
                    status.write(f"✅ Used tool: {msg.name}")
                
                elif isinstance(msg, AIMessage) and msg.content:
                    if not msg.tool_calls:
                        full_response += msg.content
                        placeholder.markdown(full_response + "▌")
            
            status.update(label="Response complete!", state="complete", expanded=False)

        placeholder.markdown(full_response)

        # --- Performance Tracking (Now correctly indented inside if user_query) ---
        ui_latency = time.time() - ui_start_time
        final_state = chatbot.get_state(config={"configurable": {"thread_id": st.session_state['thread_id']}})
        node_metadata = final_state.values.get("metadata", {})
        
        display_latency = node_metadata.get("generation_latency", f"{ui_latency:.2f}s")
        
        total_t = node_metadata.get("tokens", 0)
        if total_t == 0:
            all_msgs = final_state.values.get("messages", [])
            for m in reversed(all_msgs[-2:]):
                if isinstance(m, AIMessage):
                    usage = getattr(m, "usage_metadata", None) or m.response_metadata.get("token_usage", {})
                    if isinstance(usage, dict):
                        total_t = usage.get("total_tokens", 0)
                        if total_t > 0:
                            break

        perf_data = {
            "latency": display_latency,
            "tokens": total_t,
            "active_model": node_metadata.get("active_model", "Primary")
        }

        st.session_state['message_history'].append({
            "role": "assistant", 
            "content": full_response, 
            "performance": perf_data
        })
        
        st.rerun()