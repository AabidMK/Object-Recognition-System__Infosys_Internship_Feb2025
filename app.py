import streamlit as st #To create web interface for YOLO
import cv2 #for img and video processing
import numpy as np #to convert img bytes into numpy arrays
import tempfile #to handle uploaded video files temporarily
import time  #to control frame rate when processing videos
from ultralytics import YOLO

#Loading the pretrained model
model=YOLO('best.pt')

#Sidebar configuration
st.sidebar.title("Settings")
confidence_threshold=st.sidebar.slider("Confidence Threshold",0.0,1.0,0.5,0.05)
mode=st.sidebar.radio("Select Mode",["Image","Video","Webcam"])

#main configuration
st.title("🔍 YOLO Object Recognition Sytem")
if mode=="Image":
    st.header("🖼️ Image Detection")
    uploaded_image=st.file_uploader("Upload an image",type=["jpg","jpeg","png"])
    if uploaded_image is not None:
        file_bytes=np.asarray(bytearray(uploaded_image.read()),dtype=np.uint8) #returns the numpy array of bytes->pixels
        image=cv2.imdecode(file_bytes,cv2.IMREAD_COLOR) #numpy array of pixels to image
        st.image(image,channels="BGR",caption="Original Image") #Display original image
        results=model.predict(source=image,conf=confidence_threshold,show=False)#making predictions
        annotated_image=results[0].plot() #Returns the image with detected objects drawn on it.
        st.image(annotated_image,channels='BGR',caption='Detection Result')
elif mode=="Video":
    st.header("🎥 Video Detection")
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
    if uploaded_video is not None:
        tfile=tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap=cv2.VideoCapture(tfile.name)
        stframe=st.empty()
        st.write("Processing Video...⏳")
        while cap.isOpened():
            ret,frame=cap.read()
            if not ret:
                break
            #yolo detection
            results=model.predict(source=frame,conf=confidence_threshold,show=False)
            annotated_frame=results[0].plot()
            #display vide frames
            stframe.image(annotated_frame,channels="BGR")
            time.sleep(0.03)#Control frame rate
        cap.release()
        st.success("Video Processing Complete ✅")
elif mode=="Webcam":
    st.header("📷 Live Webcam Detection")
    # Webcam session control
    if "run_webcam" not in st.session_state:
        st.session_state.run_webcam = False
        #Uses st.session_state to store whether the webcam is running or not.
    col1,col2=st.columns(2)
    with col1:
        start_button=st.button("▶ Start Webcam")
    with col2:
        stop_button = st.button("⏹ Stop Webcam")
    if start_button:
        st.session_state.run_webcam = True
    if stop_button:
        st.session_state.run_webcam = False
    webcam_placeholder = st.empty()
    if st.session_state.run_webcam:
        cap = cv2.VideoCapture(0)  # Open default webcam
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