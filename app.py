import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import tempfile

# Load YOLO model (update path as needed)
model = YOLO("yolov8n.pt")

st.title("YOLOv8 Object Detection")

# Sidebar mode selection
mode = st.sidebar.selectbox("Select Mode", ["Image", "Video", "Webcam"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Image Detection
if mode == "Image":
    st.header("Image Detection")
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        results = model.predict(source=image, conf=confidence_threshold, show=False)
        annotated_image = results[0].plot()
        st.image(annotated_image, channels="BGR", caption="Detection Result")

# Video Detection
elif mode == "Video":
    st.header("Video Detection")
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            results = model.predict(source=frame, conf=confidence_threshold, show=False)
            annotated_frame = results[0].plot()
            st.image(annotated_frame, channels="BGR", caption="Detection Frame")
        cap.release()

# Webcam Detection
elif mode == "Webcam":
    st.header("Webcam Detection")
    class VideoProcessor(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model.predict(source=img, conf=confidence_threshold, show=False)
            return results[0].plot()
    webrtc_streamer(key="webcam", video_processor_factory=VideoProcessor)