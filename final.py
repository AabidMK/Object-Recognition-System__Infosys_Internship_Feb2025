import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from ultralytics import YOLO

# Sidebar: Select Model Type
task = st.sidebar.radio("Choose Task", ["Object Detection", "Instance Segmentation"])

# Load the appropriate model
if task == "Object Detection":
    model = YOLO('best.pt')  # Detection model
    st.title("🔍 YOLO Object Detection System")
else:
    model = YOLO('yolo11n-seg.pt')  # Segmentation model
    st.title("🔍 YOLO Object Segmentation System")

# Sidebar Configuration
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
mode = st.sidebar.radio("Select Mode", ["Image", "Video", "Webcam"])

# Main Logic
if mode == "Image":
    st.header("🖼️ Image Processing")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(image, channels="BGR", caption="Original Image")
        results = model.predict(source=image, conf=confidence_threshold, show=False)
        annotated_image = results[0].plot()
        st.image(annotated_image, channels='BGR', caption='Processed Result')

elif mode == "Video":
    st.header("🎥 Video Processing")
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        st.write("Processing Video...⏳")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(source=frame, conf=confidence_threshold, show=False)
            annotated_frame = results[0].plot()
            stframe.image(annotated_frame, channels="BGR")
            time.sleep(0.03)
        cap.release()
        st.success("Video Processing Complete ✅")

elif mode == "Webcam":
    st.header("📷 Live Webcam Processing")
    if "run_webcam" not in st.session_state:
        st.session_state.run_webcam = False
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("▶ Start Webcam")
    with col2:
        stop_button = st.button("⏹ Stop Webcam")
    if start_button:
        st.session_state.run_webcam = True
    if stop_button:
        st.session_state.run_webcam = False
    webcam_placeholder = st.empty()
    if st.session_state.run_webcam:
        cap = cv2.VideoCapture(0)
        st.write("Webcam is running... 🎥")
        while st.session_state.run_webcam and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to grab frame")
                break
            results = model.predict(source=frame, conf=confidence_threshold, show=False)
            annotated_frame = results[0].plot()
            webcam_placeholder.image(annotated_frame, channels="BGR")
            time.sleep(0.03)
        cap.release()
        st.warning("Webcam Stopped 🛑")
