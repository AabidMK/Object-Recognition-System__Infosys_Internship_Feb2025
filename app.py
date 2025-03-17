import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from ultralytics import YOLO

# Load the YOLO model
model = YOLO(r"C:\Users\eshaa\Downloads\best_yolo.pt")

# Set Streamlit Page Configurations
st.set_page_config(page_title="YOLO Object Detection", layout="wide")

# Sidebar Configuration
st.sidebar.title("⚙️ Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
mode = st.sidebar.radio("Select Mode", ["Image", "Video", "Webcam"], index=0)

st.title("🚀 YOLO Object Detection System")
st.markdown("## Upload an image or video for object detection")

if mode == "Image":
    st.subheader("📷 Image Detection")
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        col1, col2 = st.columns(2)
        col1.image(image, channels="BGR", caption="Original Image", use_column_width=True)
        
        with st.spinner("Detecting objects..."):
            results = model.predict(source=image, conf=confidence_threshold, show=False)
            annotated_image = results[0].plot()
        
        col2.image(annotated_image, channels="BGR", caption="Detection Result", use_column_width=True)

elif mode == "Video":
    st.subheader("🎥 Video Detection")
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        
        st.write("### Processing video...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(source=frame, conf=confidence_threshold, show=False)
            annotated_frame = results[0].plot()
            stframe.image(annotated_frame, channels="BGR")
            time.sleep(0.03)
        cap.release()
        st.success("Video processing complete!")

elif mode == "Webcam":
    st.subheader("📹 Live Webcam Detection")
    if "run_webcam" not in st.session_state:
        st.session_state.run_webcam = False
    
    start_button = st.button("▶ Start Webcam", key="start_webcam")
    stop_button = st.button("⏹ Stop Webcam", key="stop_webcam")
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
            results = model.predict(source=frame, conf=confidence_threshold, show=False)
            annotated_frame = results[0].plot()
            webcam_placeholder.image(annotated_frame, channels="BGR")
            time.sleep(0.03)
        cap.release()
        st.success("Webcam stopped.")
