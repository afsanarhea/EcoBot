import streamlit as st
import cv2
import torch
from pathlib import Path
from ecobot_rag import EcobotRAG
import os

from inference.video_inference import VideoInferenceWrapper

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(
    page_title="EcoBot - Plant Health Assistant",
    page_icon="E",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# CSS for styling
# -----------------------------
st.markdown("""
    <style>
    .main { padding: 1rem; }
    .user-message { background: #e3f2fd; color: #000; padding: 10px 15px; border-radius: 10px; margin: 8px 0; border-left: 4px solid #2196F3; }
    .bot-message { background: #f1f8e9; color: #000; padding: 10px 15px; border-radius: 10px; margin: 8px 0; border-left: 4px solid #4CAF50; }
    .plant-message { background: #fff3e0; color: #000; padding: 10px 15px; border-radius: 10px; margin: 8px 0; border-left: 4px solid #ff9800; display: flex; align-items: center; }
    .plant-message img.plant-avatar {
        width: 30px !important;
        height: 30px !important;
        border-radius: 50% !important;
        margin-right: 10px !important;
        flex-shrink: 0 !important;
        object-fit: cover !important;
    }
    .detection-summary { background: #2e7d32; color: white; padding: 15px; border-radius: 8px; margin: 15px 0; }
    .summary-item { padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.2); }
    .summary-item:last-child { border-bottom: none; }
    .status-healthy { color: #76ff03; font-weight: bold; }
    .status-dead { color: #ff5252; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Initialize session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "bot" not in st.session_state:
    with st.spinner("Loading EcoBot (RAG)..."):
        bot = EcobotRAG()

        # Load vector store if exists
        if os.path.exists("vectorstore/faiss_index"):
            bot.load_vector_store()
        else:
            st.error("Vector store not found!")

        # Build QA chain (same as in chatbot_app)
        bot.setup_qa_chain()

        st.session_state.bot = bot


if "model" not in st.session_state:
    with st.spinner("Loading Plant Detection Model..."):
        st.session_state.model = VideoInferenceWrapper(
            model_type="squeezenet",
            weights_path="inference/best_squeezenet.pt",
            classes_path="inference/classes.txt"
        )

if "video_processed" not in st.session_state:
    st.session_state.video_processed = False

if "detection_results" not in st.session_state:
    st.session_state.detection_results = {}

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# -----------------------------
# Helper functions
# -----------------------------
def get_latest_video():
    inference_dir = Path("inference")
    if not inference_dir.exists():
        return None
    video_files = list(inference_dir.glob("*.mp4")) + list(inference_dir.glob("*.avi")) + list(inference_dir.glob("*.mov"))
    if video_files:
        return max(video_files, key=lambda x: x.stat().st_mtime)
    return None

def analyze_video_detection(video_path, model):
    counts = {}
    cap = cv2.VideoCapture(str(video_path))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pred = model.predict(frame)
        counts[pred] = counts.get(pred, 0) + 1
    cap.release()
    return counts

def generate_auto_question(counts):
    dead_plants = sum(v for k, v in counts.items() if 'Dead' in k)
    if dead_plants >= 3:
        return "Why did multiple plants die at the same time? Is there a greenhouse problem?"
    elif dead_plants > 0:
        dead_types = [k.replace("_Dead", "").replace("_", " ") for k, v in counts.items() if 'Dead' in k and v > 0]
        if len(dead_types) > 1:
            return f"My {' and '.join(dead_types)} plants died. What are common causes?"
        elif dead_types:
            return f"My {dead_types[0]} died. What are the most common causes and solutions?"
    return "What are best practices for maintaining healthy greenhouse plants?"

# -----------------------------
# Layout
# -----------------------------
st.title("EcoBot - Plant Health Monitoring System")
st.markdown("---")
left_col, right_col = st.columns([1, 1])

# -----------------------------
# Video Analysis
# -----------------------------
with left_col:
    st.subheader("Video Analysis")
    latest_video = get_latest_video()

    if latest_video:
        st.video(str(latest_video))
        if st.button("Analyze Video", use_container_width=True):
            with st.spinner("Analyzing plants..."):
                counts = analyze_video_detection(str(latest_video), st.session_state.model)
                st.session_state.detection_results = counts
                st.session_state.video_processed = True

                # Plant notification for most detected Dead plant
                dead_classes = [k for k, v in counts.items() if 'Dead' in k and v > 0]
                if dead_classes:
                    plant_name = dead_classes[0].replace("_Dead"," ").replace("_"," ")
                    plant_msg = f"Oh no... {plant_name} is feeling bad!"
                    plant_avatar = "https://imgs.search.brave.com/fb62CXfUJBFcqM_WQNUkqplmOkRDendgKAuKNMnSX0A/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9jZG4u/dmVjdG9yc3RvY2su/Y29tL2kvcHJldmll/dy0xeC85My83OC9w/bGFudC1wb3QtY2Fy/dG9vbi13aXRoLWNy/eWluZy1nZXN0dXJl/LXZlY3Rvci0zODg4/OTM3OC5qcGc"  # URL avatar
                    st.session_state.messages.append({
                        "role": plant_name,
                        "content": plant_msg,
                        "avatar": plant_avatar
                    })

# Detection summary
if st.session_state.video_processed and st.session_state.detection_results:
    st.markdown("### Detection Summary")
    st.markdown('<div class="detection-summary">', unsafe_allow_html=True)
    for plant, count in st.session_state.detection_results.items():
        if "Dead" in plant:
            st.markdown(f'<div class="summary-item"><span class="status-dead">✗ {plant.replace("_Dead"," ").replace("_"," ")}:</span> Dead ({count})</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="summary-item"><span class="status-healthy">✓ {plant.replace("_Healthy"," ").replace("_"," ")}:</span> Healthy ({count})</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Chat Section
# -----------------------------
with right_col:
    st.subheader("Chat Assistant")

    # Display chat messages
    chat_container = st.container()
    with chat_container:
        if len(st.session_state.messages) == 0:
            st.info("AI suggestions will appear here when a plant needs attention.")
        else:
            for message in st.session_state.messages:
                role = message["role"]
                content = message["content"]
                avatar = message.get("avatar", None)

                if role == "user":
                    st.markdown(f'<div class="user-message"><b>You:</b> {content}</div>', unsafe_allow_html=True)
                elif role == "EcoBot":
                    st.markdown(f'<div class="bot-message"><b>EcoBot:</b> {content}</div>', unsafe_allow_html=True)
                else:  # Plant message
                    avatar_html = f'<img src="{avatar}" style="width:30px;height:30px;border-radius:50%;margin-right:10px;object-fit:cover;flex-shrink:0;">'
                    st.markdown(f'<div class="plant-message">{avatar_html}<b>{role}:</b> {content}</div>',
                                unsafe_allow_html=True)


    # Callback to send message
    def send_message():
        msg = st.session_state.user_input.strip()
        if msg:
            st.session_state.messages.append({"role": "user", "content": msg})
            with st.spinner("EcoBot is thinking..."):
                response = st.session_state.bot.query(msg)
                st.session_state.messages.append({"role": "EcoBot", "content": response['result']})
            # Clear input safely
            st.session_state.user_input = ""

    # Text input
    st.text_input(
        "Ask your question:",
        placeholder="e.g., Why are my plants dying?",
        key="user_input",
        on_change=send_message
    )

    # Preset buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("Aloe Care", on_click=lambda: st.session_state.update({"user_input": "Why do aloe vera plants die?"}))
    with col2:
        st.button("Snake Care", on_click=lambda: st.session_state.update({"user_input": "What kills snake plants?"}))
    with col3:
        st.button("Greenhouse", on_click=lambda: st.session_state.update({"user_input": "How to maintain greenhouse health?"}))

    # Send button
    st.button("Send", on_click=send_message)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'><small>Powered by EcoBot AI</small></div>", unsafe_allow_html=True)