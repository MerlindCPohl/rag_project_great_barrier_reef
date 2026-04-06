"""
ReefGuide Streamlit Frontend

Chat interface for retrieval function (RAG) withmessage display and source visualization.
"""

import streamlit as st
from src import get_answer

st.set_page_config(page_title="ReefGuide", page_icon="🌊")

# ==========================
# CSS styling and layout 
# ==========================

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ===============================
# UI Layout: logo and header 
# ===============================
st.markdown(
    f"""
    <div class="fixed-logo">
        <img src="data:image/png;base64,{__import__('base64').b64encode(open('./assets/Park_Authority_Logo.png', 'rb').read()).decode()}" width="100%">
    </div>
    """,
    unsafe_allow_html=True
)

st.title("G'day mate! 🪸 ")
st.subheader("Curious about the Great Barrier Reef?")
st.markdown(
    "Ask me anything — from marine life and conservation to travel tips and local insights!  \n"
    "Our ReefGuide provides clear, reliable answers to help you explore and understand this unique ecosystem."
)

# ===============================
# Chat interface with session state to store messages
# ===============================
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message["role"] == "user":
        avatar = "🐠"
    else:
        avatar = "🐙"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# ===============================
# Input Field and Responding Logic
# ===============================


prompt = st.chat_input("Ask me anything!")

if prompt:

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="🐠"):
        st.markdown(prompt)
    
    with st.spinner("🐙 ReefGuide is thinking..."):
        result = get_answer(prompt)

    response = result['response']
    sources = result.get('sources', [])
    confidence = result.get('confidence', 0.0)
    skip_sources = result.get('skip_sources', False)

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant", avatar="🐙"):
        st.markdown(response)

    if sources and not skip_sources:
        st.markdown("Sources")
        for i, source in enumerate(sources, 1):
            with st.expander(f"Source {i}: {source['source']}"):
                st.markdown(f"**Preview:** {source['preview']}")

if st.session_state.messages:
    st.divider()
    if st.button(type="primary", label="Clear out conversation"):
        st.session_state.messages = []
        st.rerun()
    
# ===============================
# Footer: AI disclaimer 
# ===============================

st.markdown(
    """
    <div class="custom-disclaimer">
        ReefGuide is an AI-based tool that generates answers to your questions.<br>
        While it aims to provide accurate information, responses may contain errors or be incomplete.<br>
        Please verify important information using reliable sources.<br>
        You can find the sources used for each answer at the end of the chat.
    </div>
    """,
    unsafe_allow_html=True
)