import streamlit as st
import PIL.Image as Image
from src import get_answer
from src.pipeline import llm, invoke_llm_with_retry
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
    st.warning("⏱️ Your conversation has been inactive for 3 minutes and will be cleared shortly. Ask a new question to continue!")
    time.sleep(3)
    st.session_state.messages = []
    st.session_state.last_activity_time = time.time()
    st.rerun()
# works ???

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
    
    # quick detection for greetings (skips retrieval)
    greeting_keywords = ['hi', 'hello', 'hey', 'how are you', 'thanks', 'thank you', 'bye', "what's up", 'good morning', 'good afternoon', 'gday', 'g\'day', 'g\'day mate', 'greetings', 'welcome', 'nice to meet you', 'pleased to meet you', 'howdy', 'salutations', 'cheers', 'hiya', 'yo', 'sup', 'good evening', 'whats up', 'how are you doing', 'how are you today', 'how is it going', 'how have you been', 'long time no see', 'nice to see you', 'glad to see you', 'good to see you']
    is_greeting = any(keyword in prompt.lower() for keyword in greeting_keywords)
    
    if is_greeting:
        # returns a quick response without retrieval
        response = "Hi there! Feel free to ask me anything about the Great Barrier Reef!"
        sources = []
        confidence = 0.0
    else:
        # LLM Classification: Is this question about the Great Barrier Reef?
        classification_prompt = f"""Is this user question asking for information about the Great Barrier Reef, marine life, ocean ecosystems, conservation, tourism, fish, coral, or related topics?
        Answer with only: YES or NO

        Question: {prompt}
        Answer:"""
                
        try:
            with st.spinner("🐙 ReefGuide is thinking..."):
                classification = invoke_llm_with_retry(llm, classification_prompt).strip().lower()
        except:
            classification = "yes" 
        
        if "yes" in classification:
            # if question is about GBR - do RAG retrieval
            with st.spinner("🐙 ReefGuide is thinking..."):
                result = get_answer(prompt)

            # extract components
            response = result['response']
            sources = result.get('sources', [])
            confidence = result.get('confidence', 0.0)
            
            # only display sources if confidence is high enough
            if confidence < 0.5:
                sources = []
        else:
            # Question is off-topic
            response = "I'm specialized in answering questions about the Great Barrier Reef, marine life, and conservation. Feel free to ask me anything about those topics!"
            sources = []
            confidence = 0.0

    # save message
    st.session_state.messages.append({"role": "assistant", "content": response})

    # display answer
    with st.chat_message("assistant", avatar="🐙"):
        st.markdown(response)
    
    # button to clear conversation
    if st.button(type="primary", label="Clear out conversation"):
        st.session_state.messages = []
        st.rerun()

    # display sources only if they meet a confidence threshold
    if sources and confidence >= 0.5:
        st.markdown("Sources")
        for i, source in enumerate(sources, 1):
            with st.expander(f"Source {i}: {source['source']}"):
                st.markdown(f"**Preview:** {source['preview']}")

    
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