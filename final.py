import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import pandas as pd
from ultralytics import YOLO
from collections import Counter, deque

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Interactive YOLO Object Recognition", page_icon="🔍", layout="wide")

# --- Custom CSS for Enhanced UI ---
st.markdown(
    """
    <style>
        body {
            background: #F0F0F0;
            color: #1E1E1E;
        }
        [data-testid="stSidebar"] {
            background: #E0E0E0;
            border-radius: 0 20px 20px 0;
        }
        .stButton > button {
            background: #007BFF;
            color: white;
            border-radius: 15px;
            transition: all 0.3s ease-in-out;
        }
        .stButton > button:hover {
            transform: scale(1.05);
            background: #0056b3;
        }
        .stImage img, .stVideo video {
            border-radius: 15px;
            border: 3px solid #007BFF;
        }
        .stFileUploader label, .stRadio label {
            color: #1E1E1E;
            font-size: 16px;
        }
        .stats-container {
            background: #F5F5F5;
            color: #1E1E1E;
            padding: 15px;
            border-radius: 15px;
            margin-top: 10px;
        }
        .custom-tab {
            background-color: #E0E0E0;
            color: #1E1E1E;
            padding: 10px 15px;
            border-radius: 10px 10px 0 0;
            font-weight: bold;
        }
        .custom-tab-active {
            background-color: #007BFF;
            color: white;
        }
        /* Ensure text visibility in Streamlit components */
        [data-testid="stMarkdownContainer"] {
            color: #1E1E1E;
        }
        .stDataFrame, .stTable {
            color: #1E1E1E;
        }
        /* Improve contrast for headers and text */
        h1, h2, h3, h4, h5, h6 {
            color: #007BFF;
        }
        p, span, div {
            color: #1E1E1E;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Session State Initialization ---
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'object_counts' not in st.session_state:
    st.session_state.object_counts = Counter()
if 'detection_times' not in st.session_state:
    st.session_state.detection_times = deque(maxlen=100)
if 'tracking_enabled' not in st.session_state:
    st.session_state.tracking_enabled = False
if 'alerts_enabled' not in st.session_state:
    st.session_state.alerts_enabled = False
if 'alert_objects' not in st.session_state:
    st.session_state.alert_objects = []
if 'alert_count' not in st.session_state:
    st.session_state.alert_count = 0
if 'processing_mode' not in st.session_state:
    st.session_state.processing_mode = "Standard"

# --- Sidebar Configuration ---
st.sidebar.markdown("## 🌈 Model Configuration")
model_type = st.sidebar.radio("Select Model Type", ["Detection", "Segmentation"])

# --- Load the YOLO model dynamically based on user selection ---
@st.cache_resource
def load_model(model_type):
    model_path = r"C:\Users\damod\Downloads\yolo11n (1).pt" if model_type == "Detection" else r"C:\Users\damod\Downloads\yolo11n-seg.pt"
    return YOLO(model_path)

model = load_model(model_type)

# --- Detection Controls ---
st.sidebar.markdown("## 🎯 Detection Controls")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
mode = st.sidebar.radio("Select Input Mode", ["Image", "Video", "Webcam"])

# --- Advanced Settings ---
with st.sidebar.expander("⚙️ Advanced Settings"):
    st.session_state.tracking_enabled = st.checkbox("Enable Object Tracking", value=st.session_state.tracking_enabled)
    st.session_state.processing_mode = st.selectbox(
        "Processing Mode", 
        ["Standard", "High Performance", "High Accuracy"],
        index=["Standard", "High Performance", "High Accuracy"].index(st.session_state.processing_mode)
    )
    
    st.session_state.alerts_enabled = st.checkbox("Enable Object Alerts", value=st.session_state.alerts_enabled)
    if st.session_state.alerts_enabled:
        all_classes = ["person", "car", "truck", "dog", "cat", "bicycle", "chair", "table"]  # Example classes
        st.session_state.alert_objects = st.multiselect("Alert when these objects are detected:", all_classes, default=st.session_state.alert_objects)

# --- Stop Button ---
stop_button = st.sidebar.button("🛑 Stop")

# --- Main UI ---
st.title("🚀 Interactive YOLO Object Recognition")
st.markdown("<p style='text-align:center; color: #007BFF;'>AI-powered real-time object detection & segmentation with analytics</p>", unsafe_allow_html=True)

# --- Tabs for Main Content ---
tab1, tab2, tab3 = st.tabs(["📷 Detection", "📊 Analytics", "⚙️ Settings"])

with tab1:
    # --- Detection UI ---
    detection_col, info_col = st.columns([3, 1])
    
    with detection_col:
        main_display = st.empty()
    
    with info_col:
        st.markdown("### 🔍 Detection Info")
        info_box = st.empty()
        current_fps = st.empty()
        alert_box = st.empty()
    
    def process_frame(frame, conf_thresh):
        # Apply different processing based on mode
        if st.session_state.processing_mode == "High Performance":
            frame = cv2.resize(frame, (640, 480))  # Lower resolution for performance
            
        if st.session_state.tracking_enabled:
            results = model.track(source=frame, conf=conf_thresh, show=False, persist=True)
        else:
            results = model.predict(source=frame, conf=conf_thresh, show=False)
        
        # Enhanced accuracy mode
        if st.session_state.processing_mode == "High Accuracy":
            # Simulate multiple-model ensemble or higher quality processing
            temp_results = model.predict(source=frame, conf=conf_thresh-0.1, show=False)
            # In a real app, you might combine multiple models here
        
        # Calculate FPS
        current_time = time.time()
        st.session_state.detection_times.append(current_time)
        recent_times = [t for t in st.session_state.detection_times if current_time - t < 1.0]
        fps = len(recent_times)
        
        # Process results
        annotated_frame = results[0].plot()
        
        # Extract detected objects for this frame
        detected_classes = []
        for box in results[0].boxes:
            cls_id = int(box.cls.item())
            class_name = results[0].names[cls_id]
            conf = float(box.conf.item())
            detected_classes.append((class_name, conf))
            st.session_state.object_counts[class_name] += 1
        
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
        
        # Add info to frame
        cv2.putText(annotated_frame, f"FPS: {fps}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"Mode: {st.session_state.processing_mode}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Update info displays
        obj_text = "\n".join([f"{cls} ({conf:.2f})" for cls, conf in detected_classes])
        info_box.markdown(f"""
        ### Detected Objects:
        {obj_text if obj_text else "No objects detected"}
        
        **Total objects tracked:** {sum(st.session_state.object_counts.values())}
        """)
        
        current_fps.markdown(f"### FPS: {fps}")
        
        if alerts:
            alert_box.error("\n".join(alerts))
        else:
            alert_box.empty()
        
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
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Add progress bar
            progress_bar = st.progress(0)
            current_frame = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or stop_button:
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
            if not ret or stop_button:
                st.warning("🛑 Stopping Webcam Stream")
                break
                
            processed_frame = process_frame(frame, confidence_threshold)
            main_display.image(processed_frame, channels="BGR", use_column_width=True)
            
            # Add a small delay to reduce CPU usage
            time.sleep(0.01)
        
        cap.release()

with tab2:
    # --- Analytics UI using native Streamlit components ---
    st.markdown("### 📊 Detection Analytics")
    
    analytics_col1, analytics_col2 = st.columns(2)
    
    with analytics_col1:
        st.subheader("Object Distribution")
        if st.session_state.object_counts:
            # Create a dataframe of object counts
            df = pd.DataFrame({
                'Object': list(st.session_state.object_counts.keys()),
                'Count': list(st.session_state.object_counts.values())
            })
            
            # Use native Streamlit bar chart
            st.bar_chart(df.set_index('Object'))
            
            # Show data table
            st.dataframe(df.sort_values('Count', ascending=False))
        else:
            st.info("No objects detected yet")
    
    with analytics_col2:
        st.subheader("Detection Timeline")
        if st.session_state.detection_history:
            # Create a simplified timeline view
            st.write("Recent Detections:")
            
            # Show last 10 detections
            for i, detection in enumerate(reversed(st.session_state.detection_history[-10:])):
                with st.expander(f"Detection at {detection['timestamp']}"):
                    for obj, conf in detection['objects']:
                        st.write(f"- {obj}: {conf:.2f} confidence")
            
            # Count objects by type
            timeline_counts = Counter()
            for detection in st.session_state.detection_history:
                for obj, _ in detection['objects']:
                    timeline_counts[obj] += 1
            
            # Display timeline counts
            timeline_df = pd.DataFrame({
                'Object': list(timeline_counts.keys()),
                'Occurrences': list(timeline_counts.values())
            })
            
            st.write("Object Occurrence Count:")
            st.dataframe(timeline_df.sort_values('Occurrences', ascending=False))
        else:
            st.info("No detection history available")
    
    # Alert Statistics
    if st.session_state.alerts_enabled:
        st.subheader("⚠️ Alert Statistics")
        st.metric("Total Alerts", st.session_state.alert_count)
        
        # Create alert statistics
        alert_counts = {obj: 0 for obj in st.session_state.alert_objects}
        for detection in st.session_state.detection_history:
            for obj, _ in detection['objects']:
                if obj in st.session_state.alert_objects:
                    alert_counts[obj] += 1
        
        # Display alert data
        alert_df = pd.DataFrame({
            'Object': list(alert_counts.keys()),
            'Alerts': list(alert_counts.values())
        })
        
        if not alert_df.empty and alert_df['Alerts'].sum() > 0:
            st.write("Alert Counts by Object:")
            st.dataframe(alert_df.sort_values('Alerts', ascending=False))
            
            # Use native Streamlit chart
            st.bar_chart(alert_df.set_index('Object'))

with tab3:
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

# --- Footer ---
st.markdown("""
<div style='text-align: center; margin-top: 40px; padding: 20px; background-color: #E0E0E0; border-radius: 15px; color: #1E1E1E;'>
    <h3>🤖 Interactive YOLO Object Recognition</h3>
    <p>Powered by Ultralytics YOLO and Streamlit</p>
</div>
""", unsafe_allow_html=True)

# --- Installation Instructions ---
with st.sidebar.expander("📋 Installation Help"):
    st.markdown("""
    ### Required packages:
    ```
    pip install streamlit opencv-python-headless numpy pandas ultralytics
    ```
    
    ### Optional packages:
    If you want enhanced visualizations:
    ```
    pip install plotly
    ```
    """)