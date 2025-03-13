import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import numpy as np
from PIL import Image

st.set_page_config(page_title="YOLO Detection", layout="wide")

model = YOLO("best.pt")

st.markdown(
    """
    <style>
        body { font-family: 'Arial', sans-serif; }
        .sidebar .sidebar-content { background-color: #f4f4f4; }
        h1 { color: #FF5733; text-align: center; }
        .stButton>button { background-color: #FF5733; color: white; font-size: 16px; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔍 YOLO Object Detection")

st.sidebar.header("⚙️ Settings")
option = st.sidebar.radio("Select Input:", ("📷 Image", "📹 Video", "🎥 Webcam"))
confidence_threshold = st.sidebar.slider("🔍 Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

st.sidebar.write("---")
st.sidebar.write("📌 **How to Use:**")
st.sidebar.write("1️⃣ Choose an input (Image, Video, or Webcam)")
st.sidebar.write("2️⃣ Adjust detection confidence if needed")
st.sidebar.write("3️⃣ View object detection results in real-time")

st.write("---")

if option == "📷 Image":
    st.subheader("Upload an Image")
    uploaded_image = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])
    
    if uploaded_image:
        image = np.array(Image.open(uploaded_image))
        results = model.predict(image, conf=confidence_threshold)
        detected_image = results[0].plot()

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(detected_image, caption="Detected Objects", use_column_width=True)

elif option == "📹 Video":
    st.subheader("Upload a Video")
    uploaded_video = st.file_uploader("Choose a file", type=["mp4", "avi", "mov"])
    
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe1, stframe2 = st.columns(2)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, conf=confidence_threshold)
            detected_frame = results[0].plot()

            with stframe1:
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", caption="Original Frame")
            with stframe2:
                st.image(cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB), channels="RGB", caption="Detected Frame")

        cap.release()
        os.remove(tfile.name)

elif option == "🎥 Webcam":
    st.subheader("Live Webcam Detection")
    start_webcam = st.button("📡 Start Webcam")

    if start_webcam:
        cap = cv2.VideoCapture(0)
        stframe1, stframe2 = st.columns(2)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, conf=confidence_threshold)
            detected_frame = results[0].plot()

            with stframe1:
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", caption="Original Frame")
            with stframe2:
                st.image(cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB), channels="RGB", caption="Detected Frame")

        cap.release()
