import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from ultralytics import YOLO

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="YOLO Object Detection", page_icon="🔍", layout="wide")

# --- Custom CSS for Enhanced Colorful UI ---
st.markdown(
    """
    <style>
        /* Vibrant Gradient Background */
        body {
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: white;
        }

        /* Animated Sidebar Styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(to bottom, #16222A, #3A6073);
            border-radius: 0 20px 20px 0;
            box-shadow: 2px 0 10px rgba(0, 0, 0, 0.5);
        }

        /* Colorful Headers */
        .stMarkdown h1 {
            text-align: center;
            background: linear-gradient(45deg, #FF4E50, #F9D423);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 20px;
            padding: 10px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }

        .stMarkdown h2 {
            color: #00BFFF;
            border-bottom: 2px solid #00BFFF;
            padding-bottom: 10px;
        }

        /* Glowing Buttons */
        div.stButton > button {
            background: linear-gradient(45deg, #FF512F, #DD2476);
            color: white;
            font-size: 18px;
            font-weight: bold;
            border-radius: 15px;
            padding: 12px 20px;
            border: none;
            box-shadow: 0 0 15px rgba(221, 36, 118, 0.5);
            transition: all 0.3s ease-in-out;
        }

        div.stButton > button:hover {
            background: linear-gradient(45deg, #DD2476, #FF512F);
            transform: scale(1.05);
            box-shadow: 0 0 25px rgba(221, 36, 118, 0.8);
        }

        /* Slider Styling */
        [data-testid="stSlider"] {
            padding: 10px;
        }
        
        [data-testid="stSlider"] > div > div {
            background-color: #00BFFF !important;
        }

        /* Radio Button Styling */
        .stRadio > div {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
        }

        /* Image & Video Containers */
        .stImage img, .stVideo video {
            border-radius: 20px;
            box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.5);
            border: 3px solid #00BFFF;
            transition: transform 0.3s;
        }

        .stImage img:hover {
            transform: scale(1.02);
        }

        /* File Uploader Styling */
        [data-testid="stFileUploader"] {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 15px;
            border: 2px dashed #00BFFF;
        }
        
        /* Caption Styling */
        .caption {
            color: #F9D423;
            font-style: italic;
            text-align: center;
            padding: 5px;
        }
        
        /* Progress Bar */
        .stProgress > div > div > div {
            background-color: #FF4E50 !important;
        }
        
        /* Expander Styling */
        .streamlit-expanderHeader {
            background-color: rgba(0, 191, 255, 0.2);
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load the YOLO model (adjust the path as needed) ---
@st.cache_resource
def load_model():
    return YOLO(r"C:\Users\damod\OneDrive\Desktop\pra\yolo11n (1).pt")

model = load_model()

# --- Sidebar Configuration with Colorful Icons ---
st.sidebar.markdown("## 🌈 Detection Controls")
st.sidebar.markdown("<div style='text-align: center; margin-bottom: 20px;'><img src='https://img.icons8.com/fluency/96/000000/artificial-intelligence.png' width='80'/></div>", unsafe_allow_html=True)

confidence_threshold = st.sidebar.slider("🎯 Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
mode = st.sidebar.radio("📊 Detection Mode", ["Image", "Video", "Webcam"], index=0)

# Add colorful stats section to sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Detection Stats")
stat_placeholder = st.sidebar.empty()

# --- Title with Animation ---
st.markdown("<div class='title-animation'><h1>✨ AI Vision Detective ✨</h1></div>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #F9D423; margin-bottom: 30px;'>Powered by YOLO - Spot objects with incredible accuracy!</p>", unsafe_allow_html=True)

# --- Image Mode ---
if mode == "Image":
    st.markdown("<h2>📷 Image Detection Portal</h2>", unsafe_allow_html=True)
    
    # Create columns for better layout
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        uploaded_image = st.file_uploader("Drop your image here ⬇️", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        # Display a spinner while processing
        with st.spinner("🔮 Magic in progress... detecting objects!"):
            # Convert the uploaded image to a NumPy array
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Create two columns for original and processed images
            col1, col2 = st.columns(2)
            
            # Show Original Image
            with col1:
                st.markdown("<h3 style='text-align: center; color: #F9D423;'>📸 Original</h3>", unsafe_allow_html=True)
                st.image(image, channels="BGR", use_column_width=True)

            # Run YOLO detection on the image
            results = model.predict(source=image, conf=confidence_threshold, show=False)
            annotated_image = results[0].plot()
            
            # Display detection results
            with col2:
                st.markdown("<h3 style='text-align: center; color: #00BFFF;'>🔍 Detection Result</h3>", unsafe_allow_html=True)
                st.image(annotated_image, channels="BGR", use_column_width=True)
            
            # Update stats in sidebar
            detected_objects = results[0].boxes.cls.cpu().numpy()
            unique_classes = np.unique(detected_objects)
            names = results[0].names
            
            stats_html = f"""
            <div style='background-color: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; border-left: 5px solid #00BFFF;'>
                <p>🎯 Objects Detected: {len(detected_objects)}</p>
                <p>🏷️ Unique Classes: {len(unique_classes)}</p>
                <p>🥇 Top Detections:</p>
                <ul>
            """
            
            # Add up to 3 detected classes to the stats
            for cls in unique_classes[:3]:
                stats_html += f"<li>{names[int(cls)]}</li>"
            
            stats_html += """
                </ul>
            </div>
            """
            
            stat_placeholder.markdown(stats_html, unsafe_allow_html=True)

# --- Video Mode ---
elif mode == "Video":
    st.markdown("<h2>🎬 Video Detection Studio</h2>", unsafe_allow_html=True)
    uploaded_video = st.file_uploader("Upload your video ⬇️", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:
        # Progress bar for video processing
        progress_bar = st.progress(0)
        st.markdown("<p style='text-align: center; color: #FF4E50;'>🎥 Processing video frames... Please wait!</p>", unsafe_allow_html=True)
        
        # Save the uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        
        # Get video info for progress tracking
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create a video display frame with animation
        st.markdown("<div style='text-align: center;'><h3 style='color: #00BFFF;'>🔮 AI Vision in Action</h3></div>", unsafe_allow_html=True)
        video_placeholder = st.empty()
        
        # Stats container
        stats_container = st.container()
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Update progress
            progress = min(frame_count / total_frames, 1.0)
            progress_bar.progress(progress)
            
            # Run YOLO detection on the current frame
            results = model.predict(source=frame, conf=confidence_threshold, show=False)
            annotated_frame = results[0].plot()
            
            # Show detected frames with animations
            video_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
            
            # Update detection stats periodically
            if frame_count % 10 == 0:
                detected_objects = results[0].boxes.cls.cpu().numpy()
                if len(detected_objects) > 0:
                    unique_classes = np.unique(detected_objects)
                    names = results[0].names
                    
                    class_names = []
                    for cls in unique_classes[:3]:
                        class_names.append(names[int(cls)])
                    
                    stat_html = f"""
                    <div style='background-color: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; border-left: 5px solid #FF4E50;'>
                        <p>⏱️ Frame: {frame_count}/{total_frames}</p>
                        <p>🎯 Objects Detected: {len(detected_objects)}</p>
                        <p>🏷️ Classes: {", ".join(class_names)}</p>
                    </div>
                    """
                    
                    stat_placeholder.markdown(stat_html, unsafe_allow_html=True)
            
            # Control frame rate
            time.sleep(1/fps)
            frame_count += 1
        
        progress_bar.progress(1.0)
        cap.release()
        st.markdown("<p style='text-align: center; color: #F9D423; font-weight: bold;'>✅ Video processing complete!</p>", unsafe_allow_html=True)

# --- Webcam Mode ---
elif mode == "Webcam":
    st.markdown("<h2>🎥 Live Webcam Detection</h2>", unsafe_allow_html=True)
    
    # Initialize webcam run state in session state
    if "run_webcam" not in st.session_state:
        st.session_state.run_webcam = False

    # Create stylish buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start Detection"):
            st.session_state.run_webcam = True
    with col2:
        if st.button("⏹ Stop Detection"):
            st.session_state.run_webcam = False

    # Create a colorful frame for webcam feed
    st.markdown("""
    <div style='padding: 10px; border-radius: 20px; background: linear-gradient(45deg, rgba(0,0,0,0.2), rgba(0,0,0,0.5)); margin: 10px 0; border: 2px solid #00BFFF;'>
        <h3 style='text-align: center; color: #F9D423;'>🌟 Live Detection Feed 🌟</h3>
    </div>
    """, unsafe_allow_html=True)
    
    webcam_placeholder = st.empty()
    fps_counter = st.empty()

    if st.session_state.run_webcam:
        cap = cv2.VideoCapture(0)  # Open default webcam
        st.markdown("<p style='text-align: center; color: #FF4E50; font-weight: bold;'>🔴 AI DETECTION ACTIVE</p>", unsafe_allow_html=True)

        # FPS calculation variables
        frame_times = []
        start_time = time.time()
        
        while st.session_state.run_webcam and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("⚠️ Failed to grab frame from webcam")
                break

            # Calculate FPS
            current_time = time.time()
            frame_times.append(current_time)
            # Keep only the last 30 frames for FPS calculation
            frame_times = [t for t in frame_times if current_time - t < 1.0]
            fps = len(frame_times)
            
            # Run YOLO detection on the current webcam frame
            results = model.predict(source=frame, conf=confidence_threshold, show=False)
            annotated_frame = results[0].plot()
            
            # Add some decorative elements to the frame
            cv2.putText(annotated_frame, f"FPS: {fps}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Show webcam feed with detections
            webcam_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
            
            # Update stats
            detected_objects = results[0].boxes.cls.cpu().numpy()
            if len(detected_objects) > 0:
                unique_classes = np.unique(detected_objects)
                names = results[0].names
                
                class_names = []
                for cls in unique_classes[:3]:
                    class_names.append(names[int(cls)])