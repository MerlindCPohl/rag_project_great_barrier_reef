import streamlit as st
from src import get_answer

st.set_page_config(page_title="ReefGuide", page_icon="🌊")

# title and description
st.title("G’day mate! 🐠 ")
st.subheader("Curious about the Great Barrier Reef?")
st.markdown(
    "Ask me anything — from marine life and conservation to travel tips and local insights!  \n"
    "Our ReefGuide provides clear, reliable answers to help you explore and understand this unique ecosystem."
)

# chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# input field
prompt = st.chat_input("Ask me anything!")

if prompt:
    # safe user megssage 
    st.session_state.messages.append({"role": "user", "content": prompt})

    # show user messages 
    with st.chat_message("user"):
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
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # display sources
    if sources:
        st.markdown("---")
        st.subheader("Sources")
        for i, source in enumerate(sources, 1):
            with st.expander(f"Source {i}: {source['source']} (Score: {source['score']})"):
                st.markdown(f"**Preview:** {source['preview']}")
    
   