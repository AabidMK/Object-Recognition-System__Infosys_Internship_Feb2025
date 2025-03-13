import streamlit as st
import numpy as np
import cv2
import tempfile
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load the YOLO model (adjust the model path as needed)
model = YOLO(r"C:\Users\bethi\OneDrive\Desktop\coco\best_yolo.pt")

st.title("YOLO Object Detection with Streamlit")

# Sidebar for mode selection and parameters
mode = st.sidebar.selectbox("Select Mode", ["Image", "Video", "Webcam"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

if mode == "Image":
    st.header("Image Detection")
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        # Read Image file as a numpy array
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(image, channels="BGR", caption="Original Image")
        
        # Run YOLO detection on the image
        results = model.predict(source=image, conf=confidence_threshold, show=False)
        annotated_image = results[0].plot()
        
        st.image(annotated_image, channels="BGR", caption="Detection Result")

elif mode == "Video":
    st.header("Video Detection")
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    
    if uploaded_video is not None:
        # Save the uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection on the current frame
            results = model.predict(source=frame, conf=confidence_threshold, show=False)
            annotated_frame = results[0].plot()
            
            stframe.image(annotated_frame, channels="BGR")
        
        cap.release()

elif mode == "Webcam":
    st.header("Webcam Live Detection")
    st.write("Streaming webcam with live YOLO detection...")
    
    class YOLOVideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            # Convert the frame to a numpy array in BGR format
            img = frame.to_ndarray(format="bgr24")
            
            # Run YOLO detection on the frame
            results = model.predict(source=img, conf=confidence_threshold, show=False)
            annotated_frame = results[0].plot()
            
            return annotated_frame
    
    webrtc_streamer(key="yolo_webcam", video_processor_factory=YOLOVideoTransformer)