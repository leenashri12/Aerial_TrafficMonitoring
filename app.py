import streamlit as st
import cv2
import supervision as sv
from ultralytics import YOLO
import tempfile
import numpy as np
import os
import time

# 1. Page Config
st.set_page_config(page_title="Aerial Traffic Monitor", layout="wide")
st.title("🚁 Aerial Traffic Monitoring & Analysis")
st.markdown("Upload a **Photo** or **Video** for tracking and counting.")

# 2. Load Model
@st.cache_resource
def load_model():
    if not os.path.exists("best.pt"):
        st.error("❌ Model file not found!")
        return None
    return YOLO("best.pt")

model = load_model()
if model is None:
    st.stop()

# 3. Sidebar Settings
st.sidebar.header("Settings")

mode = st.sidebar.selectbox("Select Mode", ["Image", "Video"])  # ✅ NEW

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3)

line_orientation = st.sidebar.selectbox(
    "Line Orientation",
    ["Vertical", "Horizontal"]
)

line_x = st.sidebar.slider("Vertical Line Position", 100, 2000, 600)
line_y = st.sidebar.slider("Horizontal Line Position", 100, 2000, 400)

# Upload
uploaded_file = st.file_uploader(
    "Upload File",
    type=["jpg", "jpeg", "png", "mp4", "mov", "avi"]
)

# Annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()

VEHICLE_CLASSES = [2, 3, 5, 7]

# ================= IMAGE MODE =================
if uploaded_file is not None and mode == "Image":
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    results = model(image, conf=conf_threshold)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Safe filtering
    if detections.class_id is not None and len(detections.class_id) == len(detections):
        mask = np.isin(detections.class_id, VEHICLE_CLASSES)
        detections = detections[mask]

    detections.tracker_id = np.arange(1, len(detections) + 1)

    car_count = len(detections)

    if car_count < 20:
        status = "Low"
    elif car_count <= 50:
        status = "Medium"
    else:
        status = "High"

    annotated = box_annotator.annotate(image.copy(), detections)
    labels = [f"Vehicle {i}" for i in detections.tracker_id]

    annotated = label_annotator.annotate(annotated, detections, labels=labels)

    st.image(annotated, channels="BGR", use_container_width=True)
    st.metric("Total Vehicles", car_count)
    st.markdown(f"**Traffic Status:** {status}")

# ================= VIDEO MODE =================
elif uploaded_file is not None and mode == "Video":

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    input_path = tfile.name
    tfile.close()

    video_info = sv.VideoInfo.from_video_path(input_path)

    # Line selection
    if line_orientation == "Vertical":
        LINE_START = sv.Point(line_x, 0)
        LINE_END = sv.Point(line_x, video_info.height)
    else:
        LINE_START = sv.Point(0, line_y)
        LINE_END = sv.Point(video_info.width, line_y)

    line_counter = sv.LineZone(
        start=LINE_START,
        end=LINE_END,
        triggering_anchors=[sv.Position.CENTER]
    )

    line_annotator = sv.LineZoneAnnotator(thickness=4, text_scale=1.5)

    generator = sv.get_video_frames_generator(source_path=input_path)

    col1, col2 = st.columns([3, 1])
    video_placeholder = col1.empty()
    metric_placeholder = col2.empty()

    start_time = time.time()

    for frame in generator:
        try:
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=conf_threshold,
                iou=0.5
            )[0]

            detections = sv.Detections.from_ultralytics(results)

            # Safe filtering
            if detections.class_id is not None and len(detections.class_id) == len(detections):
                mask = np.isin(detections.class_id, VEHICLE_CLASSES)
                detections = detections[mask]

            # Safe tracker mapping
            if results.boxes.id is not None:
                tracker_ids = results.boxes.id.cpu().numpy().astype(int)
                if len(tracker_ids) == len(detections):
                    detections.tracker_id = tracker_ids

            line_counter.trigger(detections=detections)

            annotated = frame.copy()

            if detections.tracker_id is not None and len(detections.tracker_id) > 0:
                annotated = trace_annotator.annotate(annotated, detections)
                labels = [f"ID:{tid}" for tid in detections.tracker_id]
            else:
                labels = ["..." for _ in range(len(detections))]

            annotated = box_annotator.annotate(annotated, detections)
            annotated = label_annotator.annotate(annotated, detections, labels=labels)

            annotated = line_annotator.annotate(annotated, line_counter=line_counter)

            total_flow = line_counter.in_count + line_counter.out_count
            elapsed_time = time.time() - start_time
            flow_rate = total_flow / max(elapsed_time, 1)

            if flow_rate < 2:
                status = "Low"
            elif flow_rate <= 5:
                status = "Medium"
            else:
                status = "High"

            video_placeholder.image(annotated, channels="BGR", use_container_width=True)

            metric_placeholder.markdown(f"""
            ### 📊 Live Stats
            - Vehicles in Frame: {len(detections)}
            - Direction 1: {line_counter.in_count}
            - Direction 2: {line_counter.out_count}
            - Total Flow: {total_flow}
            - Flow Rate: {flow_rate:.2f} veh/sec
            - Congestion: {status}
            """)

            time.sleep(0.03)

        except Exception as e:
            st.error(f"Error: {e}")
            break

    st.success("✅ Video Processing Complete!")