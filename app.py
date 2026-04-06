import streamlit as st
import PIL.Image as Image
from src import get_answer

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


# previous messages
for message in st.session_state.messages:
    if message["role"] == "user":
        avatar_user = "🐠"
    else:
        avatar_bot = "🐙"
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# input field (chat)
prompt = st.chat_input("Ask me anything!")

if prompt:
    # safe user megssage 
    st.session_state.messages.append({"role": "user", "content": prompt})

    # show user messages 
    with st.chat_message("user", avatar="🐠"):
        st.markdown(prompt)
    
    #get answer with sources
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
    
    # display sources
    if sources:
        st.markdown("---")
        st.subheader("Sources")
        for i, source in enumerate(sources, 1):
            with st.expander(f"Source {i}: {source['source']}"):
                st.markdown(f"**Preview:** {source['preview']}")
    
   