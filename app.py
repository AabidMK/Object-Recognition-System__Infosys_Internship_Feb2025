import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from ultralytics import YOLO

# Load the YOLO model (Ensure the path is correct)
model = YOLO(r"C:\Users\ashik\Downloads\best_yolo.pt")  # Use raw string or double \\

# Sidebar configuration
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
mode = st.sidebar.selectbox("Select Mode", ["Image", "Video", "Webcam"])

st.title("YOLO Object Detection with Streamlit")

# ----------------- IMAGE MODE -----------------
if mode == "Image":
    st.header("Image Detection")
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Convert the uploaded image to a NumPy array
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(image, channels="BGR", caption="Original Image")

        # Run YOLO detection on the image
        results = model.predict(source=image, conf=confidence_threshold, show=False)
        annotated_image = results[0].plot()  # Annotated image with detections
        st.image(annotated_image, channels="BGR", caption="Detection Result")

# ----------------- VIDEO MODE -----------------
elif mode == "Video":
    st.header("Video Detection")
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:
        # Save the uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()  # Placeholder to update video frames

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Exit if video ends
            
            # Run YOLO detection on the current frame
            results = model.predict(source=frame, conf=confidence_threshold, show=False)
            annotated_frame = results[0].plot()
            stframe.image(annotated_frame, channels="BGR")

            time.sleep(0.03)  # Control frame rate
        
        cap.release()

# ----------------- WEBCAM MODE -----------------
elif mode == "Webcam":
    st.header("Webcam Live Detection")

    # Initialize webcam state
    if "run_webcam" not in st.session_state:
        st.session_state.run_webcam = False

    # Buttons to start/stop webcam
    start_button = st.button("Start Webcam")
    stop_button = st.button("Stop Webcam")

    if start_button:
        st.session_state.run_webcam = True
    if stop_button:
        st.session_state.run_webcam = False

    webcam_placeholder = st.empty()  # Placeholder for webcam frames

    # Run webcam loop
    if st.session_state.run_webcam:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend

        st.write("Starting webcam...")

        while st.session_state.run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to grab frame")
                break

            # Run YOLO detection on the current webcam frame
            results = model.predict(source=frame, conf=confidence_threshold, show=False)
            annotated_frame = results[0].plot()
            webcam_placeholder.image(annotated_frame, channels="BGR")

            time.sleep(0.03)  # Control frame rate

        cap.release()
        st.write("Webcam stopped.")  # Print message after stopping
