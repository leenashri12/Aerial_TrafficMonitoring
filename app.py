import streamlit as st
import cv2
import supervision as sv
from ultralytics import YOLO
import tempfile
import numpy as np
import os

# 1. Page Configuration
st.set_page_config(page_title="Aerial Traffic Monitor", layout="wide")
st.title("🚁 Aerial Traffic Monitoring & Analysis ")
st.markdown("Upload a **Photo** or **Video** for tracking and counting.")

# 2. Load Model
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# 3. Sidebar
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3)

# 🔴 CHANGED: Vertical line position (X instead of Y)
line_x = st.sidebar.slider("Vertical Line Position", 100, 2000, 600)

# 4. File Upload
uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

# Annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()

# ================= IMAGE MODE =================
if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()

    if file_extension in ["jpg", "jpeg", "png"]:
        st.sidebar.info("Mode: Image Detection")

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        results = model(image, conf=conf_threshold)[0]
        detections = sv.Detections.from_ultralytics(results)

        detections.tracker_id = np.arange(1, len(detections) + 1)

        car_count = len(detections)

        if car_count < 20:
            status = "Low"
        elif car_count <= 50:
            status = "Medium"
        else:
            status = "High"

        annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
        labels = [f"ID:{tid}" for tid in detections.tracker_id]
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        st.image(annotated_image, channels="BGR", use_container_width=True)
        st.metric("Total Vehicles", car_count)
        st.markdown(f"**Traffic Status:** {status}")

# ================= VIDEO MODE =================
    else:
        st.sidebar.info("Mode: Video Tracking")

        import time

        # Save uploaded video
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        input_path = tfile.name
        tfile.close()

        # Get video info
        video_info = sv.VideoInfo.from_video_path(input_path)

        # 🔴 VERTICAL LINE (KEY CHANGE)
        LINE_START = sv.Point(line_x, 0)
        LINE_END = sv.Point(line_x, video_info.height)

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

        for frame in generator:

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=conf_threshold,
                iou=0.5
            )[0]

            detections = sv.Detections.from_ultralytics(results)

            # Map tracker IDs
            if results.boxes.id is not None:
                detections.tracker_id = results.boxes.id.cpu().numpy().astype(int)

            # Count crossings
            line_counter.trigger(detections=detections)

            annotated_frame = frame.copy()

            if detections.tracker_id is not None and len(detections.tracker_id) > 0:
                annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
                labels = [f"ID:{tid}" for tid in detections.tracker_id]
            else:
                labels = ["..." for _ in range(len(detections))]

            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

            # Draw vertical line
            annotated_frame = line_annotator.annotate(
                frame=annotated_frame,
                line_counter=line_counter
            )

            # Congestion logic
            total_flow = line_counter.in_count + line_counter.out_count

            if total_flow < 20:
                status = "Low"
            elif total_flow <=45 :
                status = "Medium"
            else:
                status = "High"

            video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)

            metric_placeholder.markdown(f"""
            ### 📊 Live Stats
            - Vehicles in Frame: {len(detections)}
            - Left → Right: {line_counter.in_count}
            - Right → Left: {line_counter.out_count}
            - Total Flow: {total_flow}
            - Congestion: {status}
            """)

        st.success("✅ Video Processing Complete!")

        time.sleep(1)