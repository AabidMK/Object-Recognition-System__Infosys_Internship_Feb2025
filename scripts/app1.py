import cv2
import numpy as np
import tempfile
import time
import streamlit as st
from ultralytics import YOLO

# Load the YOLO models (one for detection, one for segmentation)
detection_model = YOLO(r"C:\Users\eshaa\Downloads\best_yolo.pt")
segmentation_model = YOLO(r"C:\Users\eshaa\Downloads\yolo11n-seg.pt")  # Update with your segmentation model path

# Set Streamlit Page Configurations
st.set_page_config(page_title="YOLO Object Detection & Segmentation", layout="wide")

# Custom CSS for Styling
st.markdown("""
    <style>
        body {
            background-color: #f5f5f5;
        }
        .main-title {
            color: #4A90E2;
            text-align: center;
            font-size: 3em;
        }
        .subheader {
            color: #34495E;
            font-size: 2em;
        }
        .sidebar .sidebar-content {
            background-color: #2C3E50;
            color: white;
        }
        .stButton>button {
            background-color: #4A90E2;
            color: white;
            border-radius: 10px;
            padding: 8px 16px;
        }
        .tab-content {
            padding: 20px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.title("⚙️ Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
mode = st.sidebar.radio("Select Mode", ["Image", "Video", "Webcam"], index=0)
task = st.sidebar.radio("Select Task", ["Detection", "Segmentation"], index=0)

# Main Title
st.markdown("<h1 class='main-title'>🚀 YOLO Object Detection & Segmentation</h1>", unsafe_allow_html=True)
st.markdown("## Upload an image, video, or use webcam for analysis")

# Helper function to process results
def process_results(results, task_type):
    if task_type == "Detection":
        return results[0].plot()
    elif task_type == "Segmentation":
        return results[0].plot(boxes=False)  # Show masks by default, hide boxes

# Image Processing
if mode == "Image":
    st.markdown(f"<h2 class='subheader'>📷 {task} - Image</h2>", unsafe_allow_html=True)
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col1, col2 = st.columns(2)
        col1.image(image, channels="BGR", caption="Original Image", use_column_width=True)

        with st.spinner(f"🔍 Performing {task.lower()}..."):
            if task == "Detection":
                results = detection_model.predict(source=image, conf=confidence_threshold, show=False)
            else:
                results = segmentation_model.predict(source=image, conf=confidence_threshold, show=False)
            
            annotated_image = process_results(results, task)

        col2.image(annotated_image, channels="BGR", caption=f"{task} Result", use_column_width=True)

# Video Processing
elif mode == "Video":
    st.markdown(f"<h2 class='subheader'>🎥 {task} - Video</h2>", unsafe_allow_html=True)
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"], accept_multiple_files=False)

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        st.write(f"### Processing video for {task.lower()}...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if task == "Detection":
                results = detection_model.predict(source=frame, conf=confidence_threshold, show=False)
            else:
                results = segmentation_model.predict(source=frame, conf=confidence_threshold, show=False)
                
            annotated_frame = process_results(results, task)

            stframe.image(annotated_frame, channels="BGR")
            time.sleep(0.03)

        cap.release()
        st.success(f"✅ Video {task.lower()} complete!")

# Webcam Processing
elif mode == "Webcam":
    st.markdown(f"<h2 class='subheader'>📹 Live {task} - Webcam</h2>", unsafe_allow_html=True)

    if "run_webcam" not in st.session_state:
        st.session_state.run_webcam = False

    col1, col2 = st.columns(2)
    start_button = col1.button("▶ Start Webcam", key="start_webcam")
    stop_button = col2.button("⏹ Stop Webcam", key="stop_webcam")

    webcam_placeholder = st.empty()

    if start_button:
        st.session_state.run_webcam = True
    if stop_button:
        st.session_state.run_webcam = False

    if st.session_state.run_webcam:
        cap = cv2.VideoCapture(0)

        while st.session_state.run_webcam and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠ Webcam not detected!")
                break

            if task == "Detection":
                results = detection_model.predict(source=frame, conf=confidence_threshold, show=False)
            else:
                results = segmentation_model.predict(source=frame, conf=confidence_threshold, show=False)
                
            annotated_frame = process_results(results, task)

            webcam_placeholder.image(annotated_frame, channels="BGR")
            time.sleep(0.03)

        cap.release()
        st.success("✅ Webcam stopped.")

# Add some additional information
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.markdown("""
This application uses YOLO models for:
- Object Detection
- Image Segmentation

Adjust the confidence threshold to control detection sensitivity.
""")
