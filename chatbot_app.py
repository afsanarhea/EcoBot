import streamlit as st
from ecobot_rag import EcobotRAG
import os

st.set_page_config(
    page_title="Ecobot - Plant Care Assistant",
    page_icon="E",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        padding: 2rem;
        background-color: #f1f8f4;
    }
    .stTextInput > div > div > input {
        font-size: 18px;
        border-radius: 10px;
        border: 2px solid #81c784;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .user-message {
        background: linear-gradient(135deg, #66bb6a 0%, #43a047 100%);
        color: white;
        margin-left: 20%;
    }
    .bot-message {
        background: linear-gradient(135deg, #c8e6c9 0%, #a5d6a7 100%);
        color: #1b5e20;
        margin-right: 20%;
    }
    .stButton > button {
        border-radius: 10px;
        height: 3rem;
        font-size: 16px;
    }
    h1 {
        text-align: center;
        color: #2e7d32;
    }
    </style>
""", unsafe_allow_html=True)

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'bot' not in st.session_state:
    with st.spinner('Loading Ecobot...'):
        try:
            bot = EcobotRAG()
            if os.path.exists("vectorstore/faiss_index"):
                bot.load_vector_store()
                st.session_state.bot = bot
                st.session_state.bot_loaded = True
            else:
                st.error("Vector store not found!")
                st.session_state.bot_loaded = False
                st.stop()
        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.bot_loaded = False
            st.stop()

if 'llm_setup' not in st.session_state:
    st.session_state.bot.setup_qa_chain()
    st.session_state.llm_setup = True

st.title("Ecobot - Your Greenhouse Assistant")
st.markdown("**Ask me anything about Snake Plants, Spider Plants, and Aloe Vera!**")
st.markdown("---")

with st.sidebar:
    st.header("About Ecobot")
    st.write("""
    I'm your AI assistant for:
    
    **Snake Plants**
    - Care tips
    - Damage solutions
    
    **Spider Plants**
    - Maintenance guide
    - Common issues
    
    **Aloe Vera**
    - Causes of damage
    - Prevention tips
             
    **Greenhouse Care**
    - Solutions for problems
    - Dead plant actions                  
    """)
    
    st.markdown("---")
    
    st.header("System Info")
    if st.session_state.bot_loaded:
        st.success("Status: Active")
    else:
        st.error("Status: Not Loaded")
    
    st.markdown("---")
    
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.metric("Questions Asked", len([m for m in st.session_state.messages if m["role"] == "user"]))

chat_container = st.container()

with chat_container:
    if len(st.session_state.messages) == 0:
        st.info("""
         **Welcome to Ecobot!**
        
        Ask me about:
        - Plant health problems
        - How to fix plant damage
        - Plant care instructions
        
        **Try:** "My spider plant is damaged"
        """)
    
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <small><b>You</b></small><br>
                <div style="margin-top: 0.5rem; font-size: 16px;">{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <small><b>Ecobot</b></small><br>
                <div style="margin-top: 0.5rem; font-size: 16px;">{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")

input_col, button_col = st.columns([4, 1])

with input_col:
    user_question = st.text_input(
        "Type your question:",
        placeholder="e.g., How often should I water the spider plant?",
        label_visibility="collapsed",
        key="user_input"
    )

with button_col:
    send_button = st.button("Send", use_container_width=True, type="primary")

st.markdown("### Quick Actions")
quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)

with quick_col1:
    if st.button("Overwatering", use_container_width=True):
        user_question = "What are signs of overwatering?"
        send_button = True

with quick_col2:
    if st.button("Yellow Leaves", use_container_width=True):
        user_question = "Why are leaves turning yellow?"
        send_button = True

with quick_col3:
    if st.button("Fungal Infection", use_container_width=True):
        user_question = "How to treat fungal infections?"
        send_button = True

with quick_col4:
    if st.button("Dead Plant", use_container_width=True):
        user_question = "What to do if the plant is dead?"
        send_button = True

if send_button and user_question:
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })
    
    with st.spinner('Thinking...'):
        try:
            response = st.session_state.bot.query(user_question)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response['result']
            })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Error: {str(e)}"
            })
    
    st.rerun()

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>Thanks for staying with Ecobot</small>
</div>
""", unsafe_allow_html=True)