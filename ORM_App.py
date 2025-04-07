import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from ultralytics import YOLO
import os
import sys

# ======== Load the YOLO models =========
detection_model = YOLO(r"C:\Users\TEJASHWINI S\Documents\Projects\Infosys SpringBoard Internship\ORS\models\best.pt")
segmentation_model = YOLO(r"C:\Users\TEJASHWINI S\Documents\Projects\Infosys SpringBoard Internship\ORS\models\yolo11n-seg.pt")

# ======== Page Configuration =========
st.set_page_config(page_title="YOLO Object Detection & Segmentation", layout="wide")

# ======== Sidebar Configuration =========
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    mode = st.selectbox("Select Mode", ["Image", "Video", "Webcam"])
    task = st.selectbox("Select Task", ["Detection", "Segmentation"])
    st.markdown("---")
    
    # Close Application Button
    if st.button("❌ Close Application", use_container_width=True):
        st.warning("🔴 Closing Application...")
        time.sleep(1)
        os._exit(0)

# ======== Main Title =========
st.title("🎯 YOLO Object Detection & Segmentation with Streamlit")

# Select the model based on task
def get_model():
    return detection_model if task == "Detection" else segmentation_model

# ======== Image Mode =========
if mode == "Image":
    st.header("🖼️ Image Processing")
    uploaded_image = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.info("✅ Image Uploaded Successfully!")

        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, channels="BGR", caption="Original Image", use_container_width=True)

        with st.spinner(f"🔎 Running YOLO {task}..."):
            results = get_model().predict(source=image, conf=confidence_threshold, show=False)
            annotated_image = results[0].plot()

        with col2:
            st.image(annotated_image, channels="BGR", caption=f"{task} Result", use_container_width=True)

# ======== Video Mode =========
elif mode == "Video":
    st.header("🎬 Video Processing")
    uploaded_video = st.file_uploader("📥 Upload a Video", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:
        st.info("✅ Video Uploaded Successfully!")

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        progress_bar = st.progress(0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with st.spinner(f"🔎 Running YOLO {task}..."):
            frame_num = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = get_model().predict(source=frame, conf=confidence_threshold, show=False)
                annotated_frame = results[0].plot()

                stframe.image(annotated_frame, channels="BGR", use_container_width=True)
                
                frame_num += 1
                progress_bar.progress(min(frame_num / frame_count, 1.0))
                time.sleep(0.03)

            cap.release()

        st.success("🎉 Video Processing Complete!")

# ======== Webcam Mode =========
elif mode == "Webcam":
    st.header("📹 Webcam Live Processing")

    if "run_webcam" not in st.session_state:
        st.session_state.run_webcam = False

    col1, col2 = st.columns(2)

    with col1:
        start_button = st.button("▶️ Start Webcam", use_container_width=True)
    with col2:
        stop_button = st.button("⏹️ Stop Webcam", use_container_width=True)

    if start_button:
        st.session_state.run_webcam = True
        st.success("✅ Webcam Started")

    if stop_button:
        st.session_state.run_webcam = False
        st.warning("🛑 Webcam Stopped")

    webcam_placeholder = st.empty()

    if st.session_state.run_webcam:
        cap = cv2.VideoCapture(0)
        st.write("📸 **Capturing Live Feed...**")

        while st.session_state.run_webcam and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to grab frame")
                break
            
            results = get_model().predict(source=frame, conf=confidence_threshold, show=False)
            annotated_frame = results[0].plot()
            webcam_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
            time.sleep(0.03)

        cap.release()
        st.warning("🛑 Webcam Closed")

# ======== Footer =========
st.markdown("---")
st.markdown(
    """
    👨‍💻 **Developed by Tejashwini S**  
    🚀 Powered by YOLO & Streamlit  
    """
)
