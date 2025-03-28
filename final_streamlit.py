import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from ultralytics import YOLO
from collections import Counter


# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Interactive YOLO Object Recognition", page_icon="🔍", layout="wide")

# --- Session State Initialization ---
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'object_counts' not in st.session_state:
    st.session_state.object_counts = Counter()
if 'tracking_enabled' not in st.session_state:
    st.session_state.tracking_enabled = False
if 'alerts_enabled' not in st.session_state:
    st.session_state.alerts_enabled = False
if 'alert_objects' not in st.session_state:
    st.session_state.alert_objects = []
if 'alert_count' not in st.session_state:
    st.session_state.alert_count = 0

# --- Sidebar Configuration ---
st.sidebar.markdown("## 🌈 Model Configuration")
model_type = st.sidebar.radio("Select Model Type", ["Detection", "Segmentation"])

# --- Load the YOLO model dynamically based on user selection ---
@st.cache_resource
def load_model(model_type):
    model_path = r"C:\Users\bethi\OneDrive\Desktop\coco\best_yolo.pt" if model_type == "Detection" else r"C:\Users\bethi\OneDrive\Desktop\coco\yolo11n-seg (5).pt"
    return YOLO(model_path)

model = load_model(model_type)

# --- Detection Controls ---
st.sidebar.markdown("## 🎯 Detection Controls")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
mode = st.sidebar.radio("Select Input Mode", ["Image", "Video", "Webcam"])

# --- Detection UI ---
st.title("🚀 Interactive YOLO Object Recognition")
st.markdown("<p style='text-align:center; color: #007BFF;'>AI-powered real-time object detection & segmentation</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📷 Detection", "⚙️ Settings"])

with tab1:
    # --- Detection UI ---
    detection_col = st.columns([3])[0]  # Only one column for detection display
    
    with detection_col:
        main_display = st.empty()

    def process_frame(frame, conf_thresh):
        # Apply tracking if enabled
        if st.session_state.tracking_enabled:
            results = model.track(source=frame, conf=conf_thresh, show=False, persist=True)
        else:
            results = model.predict(source=frame, conf=conf_thresh, show=False)
        
        # Process results
        annotated_frame = results[0].plot()
        
        # Extract detected objects for this frame
        detected_classes = []
        for box in results[0].boxes:
            cls_id = int(box.cls.item())
            class_name = results[0].names[cls_id]
            conf = float(box.conf.item())
            detected_classes.append((class_name, conf))
        
        # Add to history
        if detected_classes:
            st.session_state.detection_history.append({
                'timestamp': time.strftime("%H:%M:%S"),
                'objects': detected_classes
            })
        
        # Check for alerts
        alerts = []
        if st.session_state.alerts_enabled:
            for obj, _ in detected_classes:
                if obj in st.session_state.alert_objects:
                    alerts.append(f"⚠️ {obj.upper()} detected!")
                    st.session_state.alert_count += 1
        
        # Add alerts to the frame (optional)
        for alert in alerts:
            cv2.putText(annotated_frame, alert, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return annotated_frame

    if mode == "Image":
        uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            processed_image = process_frame(image, confidence_threshold)
            main_display.image(processed_image, channels="BGR", use_column_width=True)

    elif mode == "Video":
        uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            cap = cv2.VideoCapture(tfile.name)
            
            # Extract video information
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Add progress bar
            progress_bar = st.progress(0)
            current_frame = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                processed_frame = process_frame(frame, confidence_threshold)
                main_display.image(processed_frame, channels="BGR", use_column_width=True)
                
                # Update progress
                current_frame += 1
                progress_bar.progress(min(current_frame / frame_count, 1.0))
                
                # Add a small delay to make the video watchable
                time.sleep(0.02)
            
            cap.release()

    elif mode == "Webcam":
        st.markdown("<h3 style='text-align: center; color: #007BFF;'>🎥 Live Webcam Detection</h3>", unsafe_allow_html=True)
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("🛑 Stopping Webcam Stream")
                break
                
            processed_frame = process_frame(frame, confidence_threshold)
            main_display.image(processed_frame, channels="BGR", use_column_width=True)
            
            # Add a small delay to reduce CPU usage
            time.sleep(0.01)
        
        cap.release()

with tab2:
    # --- Settings UI ---
    st.markdown("### ⚙️ Application Settings")
    
    settings_col1, settings_col2 = st.columns(2)
    
    with settings_col1:
        st.subheader("Display Settings")
        display_bbox = st.checkbox("Show Bounding Boxes", value=True)
        display_labels = st.checkbox("Show Labels", value=True)
        display_conf = st.checkbox("Show Confidence", value=True)
        
        st.subheader("Detection Settings")
        nms_threshold = st.slider("NMS Threshold", 0.0, 1.0, 0.45, 0.05, 
                                 help="Non-Maximum Suppression threshold to filter overlapping detections")
        
    with settings_col2:
        st.subheader("Export Options")
        export_format = st.selectbox("Export Format", ["CSV", "JSON"])
        
        if st.button("Export Detection History"):
            if st.session_state.detection_history:
                # Create a flattened dataframe from detection history
                export_data = []
                for detection in st.session_state.detection_history:
                    for obj, conf in detection['objects']:
                        export_data.append({
                            'Timestamp': detection['timestamp'],
                            'Object': obj,
                            'Confidence': conf
                        })
                
                export_df = pd.DataFrame(export_data)
                
                # Create a download link based on selected format
                if export_format == "CSV":
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        "detection_history.csv",
                        "text/csv",
                        key='download-csv'
                    )
                elif export_format == "JSON":
                    json_str = export_df.to_json(orient="records")
                    st.download_button(
                        "Download JSON",
                        json_str,
                        "detection_history.json",
                        "application/json",
                        key='download-json'
                    )
            else:
                st.warning("No detection history to export")
        
        st.subheader("Reset Options")
        if st.button("Reset All Statistics"):
            st.session_state.detection_history = []
            st.session_state.object_counts = Counter()
            st.session_state.alert_count = 0
            st.success("All statistics have been reset")
