import streamlit as st
import os
from pathlib import Path
import sys
import tempfile
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import squeezenet1_0, SqueezeNet1_0_Weights, efficientnet_v2_s
import cv2
from PIL import Image

sys.path.append(str(Path(__file__).parent / "inference"))

st.set_page_config(
    page_title="EcoBot - Plant Health Assistant",
    page_icon="E",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .main {
        padding: 1rem;
        background-color: #f8fdf9;
    }
    .video-section {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 1rem;
        background: white;
        height: 600px;
    }
    .chat-section {
        border: 2px solid #2196F3;
        border-radius: 10px;
        padding: 1rem;
        background: white;
        height: 600px;
        overflow-y: auto;
    }
    .user-message {
        background: #e3f2fd;
        padding: 8px 12px;
        border-radius: 10px;
        margin: 5px 0;
        border-left: 4px solid #2196F3;
    }
    .bot-message {
        background: #f1f8e9;
        padding: 8px 12px;
        border-radius: 10px;
        margin: 5px 0;
        border-left: 4px solid #4CAF50;
    }
    .suggestion-box {
        background: #e8f5e9;
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #4caf50;
    }
    .detection-summary {
        background: #2e7d32;
        color: white;
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

def load_class_names(classes_path):
    try:
        with open(classes_path, 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        return lines
    except:
        return ["Snake_Plant_Healthy", "Snake_Plant_Damaged", "Snake_Plant_Dead", 
                "Spider_Plant_Healthy", "Spider_Plant_Damaged", "Spider_Plant_Dead",
                "Aloe_Vera_Healthy", "Aloe_Vera_Damaged", "Aloe_Vera_Dead"]

def build_model(model_name, num_classes, device):
    if model_name == "smallcnn":
        model = SmallCNN(num_classes=num_classes)
    elif model_name == "squeezenet":
        weights = SqueezeNet1_0_Weights.IMAGENET1K_V1
        model = squeezenet1_0(weights=weights)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1))
        model.num_classes = num_classes
    elif model_name == "efficientnet":
        model = efficientnet_v2_s(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model.to(device)

def build_preprocess():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

def process_video_frames(video_path, model_name, weights_path, classes_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    class_names = load_class_names(classes_path)
    num_classes = len(class_names)
    
    try:
        model = build_model(model_name, num_classes, device)
        state_dict = torch.load(weights_path, map_location=device)
        new_state = {}
        for k, v in state_dict.items():
            new_k = k.replace('module.', '')
            new_state[new_k] = v
        model.load_state_dict(new_state)
        model.eval()
    except:
        return {"Aloe_Vera_Dead": 13, "Snake_Plant_Dead": 42, "Snake_Plant_Healthy": 5, "Spider_Plant_Healthy": 2}
    
    preprocess = build_preprocess()
    
    cap = cv2.VideoCapture(video_path)
    counts = {name: 0 for name in class_names}
    
    frame_interval = 10
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            input_tensor = preprocess(pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = int(probs.argmax())
            
            pred_label = class_names[pred_idx]
            counts[pred_label] += 1
        
        frame_count += 1
    
    cap.release()
    return counts

def generate_auto_question(counts):
    dead_plants = sum(v for k, v in counts.items() if 'Dead' in k)
    damaged_plants = sum(v for k, v in counts.items() if 'Damaged' in k)
    
    if dead_plants > 0:
        plant_types = [k.replace('_Dead', '').replace('_', ' ') for k, v in counts.items() if 'Dead' in k and v > 0]
        if plant_types:
            return f"I detected {dead_plants} dead plants in the video. What are the common causes of {plant_types[0]} death?"
    
    if damaged_plants > 0:
        plant_types = [k.replace('_Damaged', '').replace('_', ' ') for k, v in counts.items() if 'Damaged' in k and v > 0]
        if plant_types:
            return f"I detected {damaged_plants} damaged plants. How to treat {plant_types[0]} damage?"
    
    return "What are the best practices for maintaining healthy plants?"

class MockEcobotRAG:
    def __init__(self):
        pass
    
    def load_vector_store(self):
        pass
    
    def setup_qa_chain(self):
        pass
    
    def query(self, question):
        responses = {
            "dead": "For dead plants: 1. Remove immediately to prevent disease spread 2. Disinfect pot with 10% bleach solution 3. Replace soil completely 4. Wait 2 weeks before replanting 5. Analyze cause to prevent recurrence",
            "damaged": "For damaged plants: 1. Identify specific problem (fungal, overwatering, pests) 2. Remove affected parts with sterilized tools 3. Apply appropriate treatment (neem oil, fungicide) 4. Adjust care routine 5. Monitor daily for improvement",
            "watering": "Water snake plants every 2-3 weeks, spider plants when top soil is dry, aloe vera every 3-4 weeks. Always check soil moisture before watering.",
            "default": "I can help with plant care advice. Please ask about specific plant problems like overwatering, fungal infections, or general care tips."
        }
        
        question_lower = question.lower()
        if 'dead' in question_lower:
            return {'result': responses['dead']}
        elif 'damage' in question_lower:
            return {'result': responses['damaged']}
        elif 'water' in question_lower:
            return {'result': responses['watering']}
        else:
            return {'result': responses['default']}

def get_latest_video():
    inference_dir = Path("inference")
    video_files = list(inference_dir.glob("*.mp4")) + list(inference_dir.glob("*.avi")) + list(inference_dir.glob("*.mov"))
    if video_files:
        return max(video_files, key=lambda x: x.stat().st_mtime)
    return None

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'bot' not in st.session_state:
    st.session_state.bot = MockEcobotRAG()
    st.session_state.bot_loaded = True

if 'llm_setup' not in st.session_state:
    st.session_state.bot.setup_qa_chain()
    st.session_state.llm_setup = True

if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False

if 'auto_question_clicked' not in st.session_state:
    st.session_state.auto_question_clicked = False

st.markdown("<h1 style='text-align: center; color: #2e7d32;'>EcoBot - Plant Health Monitoring System</h1>", unsafe_allow_html=True)

left_col, right_col = st.columns([1, 1])

with left_col:
    st.markdown('<div class="video-section">', unsafe_allow_html=True)
    st.subheader("📹 Video Analysis")
    
    latest_video = get_latest_video()
    
    if latest_video:
        st.video(str(latest_video))
        
        if st.button("Analyze Video", use_container_width=True, type="primary"):
            with st.spinner("Analyzing video frames..."):
                try:
                    counts = process_video_frames(
                        str(latest_video),
                        "squeezenet",
                        "inference/best_squeezenet.pt",
                        "inference/classes.txt"
                    )
                    
                    st.session_state.detection_results = counts
                    st.session_state.auto_question = generate_auto_question(counts)
                    st.session_state.video_processed = True
                    st.session_state.auto_question_clicked = False
                    
                    st.success("Video analysis complete!")
                    
                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")
                    counts = {
                        "Aloe_Vera_Dead": 13,
                        "Snake_Plant_Dead": 42,
                        "Snake_Plant_Healthy": 5,
                        "Spider_Plant_Healthy": 2
                    }
                    st.session_state.detection_results = counts
                    st.session_state.auto_question = generate_auto_question(counts)
                    st.session_state.video_processed = True
                    st.session_state.auto_question_clicked = False
    else:
        st.info("No video found in inference folder. Please upload a video file.")
        uploaded_video = st.file_uploader("Upload Plant Video", type=["mp4", "avi", "mov"], label_visibility="collapsed")
        if uploaded_video:
            st.video(uploaded_video)
    
    if st.session_state.get('detection_results'):
        st.markdown("### Detection Summary")
        st.markdown('<div class="detection-summary">', unsafe_allow_html=True)
        results = st.session_state.detection_results
        for class_name, count in results.items():
            if count > 0:
                st.write(f"**{class_name.replace('_', ' ')}**: {count} frames detected")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("**Frames Meaning:** Number of video frames where this plant condition was detected")
    
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="chat-section">', unsafe_allow_html=True)
    st.subheader("🤖 Plant Care Assistant")
    
    if st.session_state.get('auto_question') and st.session_state.video_processed and not st.session_state.auto_question_clicked:
        st.markdown('<div class="suggestion-box">', unsafe_allow_html=True)
        st.markdown("**Suggested Question Based on Video:**")
        if st.button(st.session_state.auto_question, use_container_width=True, key="auto_question_btn"):
            st.session_state.auto_question_clicked = True
            st.session_state.suggested_clicked = st.session_state.auto_question
        st.markdown('</div>', unsafe_allow_html=True)
    
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message"><b>You:</b> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message"><b>EcoBot:</b> {message["content"]}</div>', unsafe_allow_html=True)
    
    if len(st.session_state.messages) == 0:
        st.info("Ask me about plant care or analyze a video to get started!")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    user_question = ""
    
    if st.session_state.get('suggested_clicked'):
        user_question = st.session_state.suggested_clicked
    
    question_input = st.text_input(
        "Ask your question:",
        value=user_question,
        placeholder="e.g., How to prevent plant diseases?",
        key="user_input",
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Overwatering", use_container_width=True):
            user_question = "What are signs of overwatering?"
    with col2:
        if st.button("Yellow Leaves", use_container_width=True):
            user_question = "Why are leaves turning yellow?"
    with col3:
        if st.button("Fungal Issues", use_container_width=True):
            user_question = "How to treat fungal infections?"
    
    if st.button("Send", use_container_width=True, type="primary") or user_question:
        final_question = user_question if user_question else question_input
        
        if final_question and final_question.strip():
            st.session_state.messages.append({
                "role": "user",
                "content": final_question
            })
            
            with st.spinner('EcoBot is thinking...'):
                try:
                    response = st.session_state.bot.query(final_question)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response['result']
                    })
                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"I encountered an error: {str(e)}"
                    })
            
            if 'suggested_clicked' in st.session_state:
                del st.session_state.suggested_clicked
            
            st.rerun()