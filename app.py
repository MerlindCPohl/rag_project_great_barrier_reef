import streamlit as st
import PIL.Image as Image
from src import get_answer
import time

logo_img = Image.open("./assets/Park_Authority_Logo.png")

st.set_page_config(page_title="ReefGuide", page_icon="🌊")

# css for styling the app
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# fixed logo in top right using HTML container
st.markdown(
    f"""
    <div class="fixed-logo">
        <img src="data:image/png;base64,{__import__('base64').b64encode(open('./assets/Park_Authority_Logo.png', 'rb').read()).decode()}" width="100%">
    </div>
    """,
    unsafe_allow_html=True
)

# title and description
st.title("G'day mate! 🪸 ")
st.subheader("Curious about the Great Barrier Reef?")
st.markdown(
    "Ask me anything — from marine life and conservation to travel tips and local insights!  \n"
    "Our ReefGuide provides clear, reliable answers to help you explore and understand this unique ecosystem."
)

# chat interface with session state to store messages and avatars
if "messages" not in st.session_state:
    st.session_state.messages = []

# initialize last activity timestamp for inactivity timeout
if "last_activity_time" not in st.session_state:
    st.session_state.last_activity_time = time.time()

# check for inactivity timeout (3 minutes = 180 seconds)
inactivity_timeout = 180
current_time = time.time()
time_since_last_activity = current_time - st.session_state.last_activity_time

if st.session_state.messages and time_since_last_activity > inactivity_timeout:
    st.session_state.messages = []
    st.session_state.last_activity_time = time.time()
    st.rerun()

# previous messages
for message in st.session_state.messages:
    if message["role"] == "user":
        avatar = "🐠"
    else:
        avatar = "🐙"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


# input field (chat)
prompt = st.chat_input("Ask me anything!")

if prompt:
    # update last activity timestamp
    st.session_state.last_activity_time = time.time()
    
    # safe user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # show user messages 
    with st.chat_message("user", avatar="🐠"):
        st.markdown(prompt)
    
    # Get answer (all logic handled in backend)
    with st.spinner("🐙 ReefGuide is thinking..."):
        result = get_answer(prompt)

    # extract components
    response = result['response']
    sources = result.get('sources', [])
    confidence = result.get('confidence', 0.0)
    skip_sources = result.get('skip_sources', False)

    # save message
    st.session_state.messages.append({"role": "assistant", "content": response})

    # display answer
    with st.chat_message("assistant", avatar="🐙"):
        st.markdown(response)

    # display sources only if backend says to show them
    if sources and not skip_sources:
        st.markdown("Sources")
        for i, source in enumerate(sources, 1):
            with st.expander(f"Source {i}: {source['source']}"):
                st.markdown(f"**Preview:** {source['preview']}")

# button to clear conversation 
if st.session_state.messages:
    st.divider()
    if st.button(type="primary", label="Clear out conversation"):
        st.session_state.messages = []
        st.rerun()
    
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