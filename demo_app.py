import streamlit as st
import os
from pathlib import Path
import sys
import subprocess

sys.path.append(str(Path(__file__).parent / "inference"))
from infer_smallcnn_video import load_class_names

st.set_page_config(
    page_title="Plant Health Classification",
    page_icon="E",
    layout="wide"
)

st.title("Plant Health Classification System")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Settings")
    
    model_choice = st.selectbox("Select Model", ["squeezenet", "efficientnet", "smallcnn"])
    
    weights_map = {
        "squeezenet": "inference/best_squeezenet.pt",
        "efficientnet": "inference/best_efficientnetv2s.pt",
        "smallcnn": "inference/best_smallcnn.pt"
    }
    
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    use_default = st.checkbox("Use default video", value=True)
    
    if st.button("Run Classification", use_container_width=True, type="primary"):
        video_path = "inference/video.mp4" if use_default else None
        
        if uploaded_video:
            temp_path = f"inference/temp_{uploaded_video.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_video.read())
            video_path = temp_path
        
        if video_path and os.path.exists(video_path):
            with st.spinner("Running inference..."):
                try:
                    classes_list = load_class_names("inference/classes.txt")
                    
                    cmd = [
                        sys.executable,
                        "inference/infer_smallcnn_video.py",
                        "--video_path", video_path,
                        "--model", model_choice,
                        "--weights", weights_map[model_choice],
                        "--classes", "inference/classes.txt",
                        "--no_display"
                    ]  
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    output_text = "Processing completed.\n"
                    output_text += f"Model: {model_choice}\n"
                    output_text += f"Video: {os.path.basename(video_path)}\n\n"
                    
                    if result.stdout:
                        output_text += result.stdout
                    
                    if result.stderr and "Error" in result.stderr:
                        output_text += "\n\nErrors:\n" + result.stderr
                    
                    st.session_state.classification_output = output_text
                    st.session_state.inference_done = True
                    st.success("Classification complete!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.error("No video selected")

with col2:
    st.subheader("Results")
    
    if st.session_state.get('inference_done', False):
        if 'classification_output' in st.session_state:
            st.text_area("Classification Results", st.session_state.classification_output, height=300)
        
        st.subheader("Input Video")
        if use_default:
            st.video("inference/video.mp4")
        elif uploaded_video:
            st.video(uploaded_video)
    else:
        st.info("Run classification to see results")