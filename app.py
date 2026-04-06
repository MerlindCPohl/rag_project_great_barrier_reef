import streamlit as st
import PIL.Image as Image
from src import get_answer
import time

logo_img = Image.open("./assets/Park_Authority_Logo.png")

st.set_page_config(page_title="ReefGuide", page_icon="🌊")



col_text, col_logo = st.columns([0.75, 0.25, ], gap="large")

with col_logo:
    st.image(logo_img)

with col_text:
    # title and description
    st.title("G’day mate! 🪸 ")
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
    st.warning("⏱️ Your conversation has been inactive for 3 minutes and will be cleared shortly. Ask a new question to continue!")
    time.sleep(3)
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
    
    #get answer with sources
    with st.spinner("🐙 ReefGuide is thinking..."):
        result = get_answer(prompt)

    # extract components
    response = result['response']
    sources = result.get('sources', [])
    confidence = result.get('confidence', 0.0)

    # save message
    st.session_state.messages.append({"role": "assistant", "content": response})

    # display answer
    with st.chat_message("assistant", avatar="🐙"):
        st.markdown(response)
    
    # button to clear conversation
    if st.button(type="primary", label="Clear out conversation", icon_position="middle"):
        st.session_state.messages = []
        st.rerun()

    # display sources
    if sources:
        st.markdown("Sources")
        for i, source in enumerate(sources, 1):
            with st.expander(f"Source {i}: {source['source']}"):
                st.markdown(f"**Preview:** {source['preview']}")
    
   