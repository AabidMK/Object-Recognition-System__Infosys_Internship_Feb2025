import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO

# Load the YOLO model
model = YOLO(r"C:\Users\kumar\OneDrive\Desktop\coco_trained3.pt")  # Change to your model path

# Streamlit UI
st.title("🔍 YOLOv8 Object Detection")
st.sidebar.header("Upload an Image or Video")
st.sidebar.write("Or use the **Webcam** for live detection.")

# 📌 File uploader for image/video
uploaded_file = st.sidebar.file_uploader("Upload an Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

# 📌 Webcam button
use_webcam = st.sidebar.checkbox("Use Webcam")

# 📌 Function to process image
def process_image(image):
    results = model(image)  # Run YOLO detection
    for result in results:
        image = result.plot()  # Draw detections
    return image

# 📌 Function to process video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()  # Placeholder for video frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO detection
        results = model(frame)
        for result in results:
            frame = result.plot()
        
        # Convert frame to RGB (Streamlit uses RGB format)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Show frame in Streamlit
        stframe.image(frame, channels="RGB", use_container_width=True)

    cap.release()

# 📌 Process Uploaded File
if uploaded_file:
    file_type = uploaded_file.type
    tfile = tempfile.NamedTemporaryFile(delete=False)  
    tfile.write(uploaded_file.read())  # Save uploaded file

    if "image" in file_type:
        image = cv2.imread(tfile.name)
        image = process_image(image)  # Process image
        st.image(image, channels="BGR", caption="Detected Image", use_container_width=True)

    elif "video" in file_type:
        st.video(uploaded_file)  # Show original video
        process_video(tfile.name)  # Process video

# 📌 Process Webcam Stream
if use_webcam:
    st.write("Starting Webcam...")
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO detection
        results = model(frame)
        for result in results:
            frame = result.plot()

        # Convert frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB", use_container_width=True)

    cap.release()
