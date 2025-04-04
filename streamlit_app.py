import tempfile

import cv2
import streamlit as st
from pathlib import Path
from ultralytics import YOLO

def display_tracker_options():
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def analysis_video(video_path, conf):
    model = YOLO(model_path)
    vid_cap = cv2.VideoCapture(video_path)
    st_frame = st.empty()
    is_display_tracker, tracker = display_tracker_options()
    while vid_cap.isOpened():
        success, image = vid_cap.read()
        if success:
            image = cv2.resize(image, (720, int(720 * (9 / 16))))
            # Display object tracking, if specified
            if is_display_tracker:
                res = model.track(image, conf=conf, persist=True, tracker=tracker)
            else:
                # Predict the objects in the image using the YOLOv8 model
                res = model.predict(image, conf=conf)

            # # Plot the detected objects on the video frame
            res_plotted = res[0].plot()
            st_frame.image(res_plotted,
                           caption='Detected Video',
                           channels="BGR",
                           use_container_width=True
                           )
        else:
            vid_cap.release()
            break


if __name__ == '__main__':

    st.set_page_config(
        page_title="Object Detection using YOLOv8",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("Object Detection And Tracking using YOLOv8")
    st.sidebar.header("ML Model Config")
    model_type = st.sidebar.radio(
        "Select Task", ['Detection', 'Segmentation'])
    confidence = float(st.sidebar.slider(
        "Select Model Confidence", 25, 100, 40)) / 100
    if model_type == 'Detection':
        model_path = Path('weights/yolov8n.pt')
    elif model_type == 'Segmentation':
        model_path = Path('weights/yolov8n-seg.pt')

    display_tracker = st.sidebar.radio("Display Tracker", ('Yes', 'No'))
    tracker_type = st.sidebar.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))

    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name
        analysis_video(temp_path, confidence)
