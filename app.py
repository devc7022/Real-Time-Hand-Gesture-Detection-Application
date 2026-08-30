import time
from datetime import datetime, timezone
import logging
import streamlit as st
import cv2
import numpy as np

from src.config.settings import settings
from src.utils.logging_config import setup_logging
from src.models.gesture import GestureType, GestureResult
from src.camera.webcam import WebcamStream
from src.detection.hand_detector import HandDetector
from src.detection.gesture_classifier import GestureClassifier
from src.webhook.webhook_client import WebhookClient

# Initialize central application logging
logger = setup_logging(settings.LOG_LEVEL)

# Page Configuration
st.set_page_config(
    page_title="Real-Time Hand Gesture Detector",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark Glassmorphism UI)
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #9CA3AF;
        font-size: 1.0rem;
        margin-bottom: 1.5rem;
    }
    .gesture-badge {
        padding: 0.8rem 1.2rem;
        border-radius: 12px;
        font-size: 1.4rem;
        font-weight: 700;
        text-align: center;
        margin-top: 0.5rem;
        margin-bottom: 1.0rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .finger-pill {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        margin: 0.2rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .finger-on {
        background-color: #10B981;
        color: #FFFFFF;
    }
    .finger-off {
        background-color: #374151;
        color: #9CA3AF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Session State Initialization
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False
if "webhook_url" not in st.session_state:
    st.session_state.webhook_url = settings.DEFAULT_WEBHOOK_URL
if "cooldown_seconds" not in st.session_state:
    st.session_state.cooldown_seconds = settings.WEBHOOK_COOLDOWN_SECONDS
if "last_sent_gesture" not in st.session_state:
    st.session_state.last_sent_gesture = None
if "last_sent_time" not in st.session_state:
    st.session_state.last_sent_time = 0.0
if "last_webhook_status" not in st.session_state:
    st.session_state.last_webhook_status = "Not Configured"
if "event_history" not in st.session_state:
    st.session_state.event_history = []
if "camera_index" not in st.session_state:
    st.session_state.camera_index = settings.CAMERA_INDEX

webhook_client = WebhookClient(timeout=settings.WEBHOOK_TIMEOUT)

# Header Section
st.markdown('<div class="main-title">🖐️ Real-Time Hand Gesture Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Computer vision gesture recognition powered by MediaPipe & Streamlit. Configured with webhook event debouncing.</div>',
    unsafe_allow_html=True,
)

# Sidebar Configuration Controls
st.sidebar.header("📷 Camera Setup")
camera_idx_input = st.sidebar.number_input(
    "Camera Device Index", min_value=0, max_value=5, value=st.session_state.camera_index
)
st.session_state.camera_index = int(camera_idx_input)

col_cam1, col_cam2 = st.sidebar.columns(2)
if col_cam1.button("▶️ Start Camera", use_container_width=True, type="primary"):
    st.session_state.camera_running = True
    st.rerun()

if col_cam2.button("⏹️ Stop Camera", use_container_width=True):
    st.session_state.camera_running = False
    st.rerun()

st.sidebar.divider()
st.sidebar.header("🔗 Webhook Settings")
url_input = st.sidebar.text_input(
    "Webhook Endpoint URL",
    value=st.session_state.webhook_url,
    placeholder="https://webhook.site/your-uuid-here",
    help="Enter an HTTP/HTTPS URL to receive detected gesture POST payloads.",
)
st.session_state.webhook_url = url_input.strip()

cooldown_slider = st.sidebar.slider(
    "Cooldown / Debounce (Seconds)",
    min_value=0.5,
    max_value=10.0,
    value=float(st.session_state.cooldown_seconds),
    step=0.5,
    help="Minimum seconds required before re-triggering a webhook for the same repeated gesture.",
)
st.session_state.cooldown_seconds = cooldown_slider

# Test Webhook Connectivity Button
if st.sidebar.button("🧪 Test Webhook Connection", use_container_width=True):
    if not st.session_state.webhook_url:
        st.sidebar.warning("Please enter a valid Webhook URL first.")
    else:
        with st.sidebar.status("Testing webhook connection..."):
            success, code, msg = webhook_client.test_connection(st.session_state.webhook_url)
            if success:
                st.sidebar.success(f"Webhook Connected! {msg}")
                st.session_state.last_webhook_status = f"Test OK ({code})"
            else:
                st.sidebar.error(f"Connection Failed: {msg}")
                st.session_state.last_webhook_status = f"Test Failed: {msg}"

st.sidebar.divider()
st.sidebar.header("📊 System Info")
st.sidebar.info(
    f"**Status**: {'🟢 Running' if st.session_state.camera_running else '🔴 Stopped'}\n\n"
    f"**Last Webhook**: {st.session_state.last_webhook_status}\n\n"
    f"**Total Events**: {len(st.session_state.event_history)}"
)

# Main UI Grid Layout
col_left, col_right = st.columns([2.2, 1.0])

with col_left:
    st.subheader("Live Video Stream")
    image_placeholder = st.empty()
    if not st.session_state.camera_running:
        image_placeholder.info("👈 Click **Start Camera** in the sidebar to begin real-time gesture detection.")

with col_right:
    st.subheader("Detection Metrics")
    gesture_metric_placeholder = st.empty()
    details_placeholder = st.empty()
    fingers_placeholder = st.empty()
    webhook_status_placeholder = st.empty()

    with st.expander("📜 Recent Webhook Events", expanded=True):
        history_placeholder = st.empty()


def render_event_history_table():
    if not st.session_state.event_history:
        return "*(No events triggered yet)*"
    
    rows = []
    for ev in reversed(st.session_state.event_history[-8:]):
        rows.append(f"- **{ev['gesture']}** at `{ev['time']}` ({ev['status']})")
    return "\n".join(rows)


history_placeholder.markdown(render_event_history_table())

# Real-Time Webcam Stream Detection Loop
if st.session_state.camera_running:
    logger.info("Starting live webcam stream processing loop...")
    webcam = WebcamStream(
        camera_index=st.session_state.camera_index,
        width=settings.FRAME_WIDTH,
        height=settings.FRAME_HEIGHT,
    )

    if not webcam.start():
        st.error(
            f"❌ Unable to open camera at device index {st.session_state.camera_index}. "
            "Please verify webcam hardware availability, permissions, or try another camera index."
        )
        st.session_state.camera_running = False
    else:
        detector = HandDetector(
            max_num_hands=settings.MAX_NUM_HANDS,
            min_detection_confidence=settings.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=settings.MIN_TRACKING_CONFIDENCE,
        )
        classifier = GestureClassifier()

        try:
            while st.session_state.camera_running:
                success, bgr_frame, rgb_frame = webcam.read_frame()
                if not success or rgb_frame is None:
                    st.warning("⚠️ Camera frame capture failed. Retrying...")
                    time.sleep(0.1)
                    continue

                # Run Hand Detection & Gesture Classification
                detection_result = detector.detect(rgb_frame, draw_landmarks=True)
                gesture_result = classifier.classify(detection_result)

                # Render Visual Overlay Text on OpenCV BGR Frame (for crisp output)
                annotated_rgb = detection_result.annotated_rgb_frame
                label_text = f"GESTURE: {gesture_result.gesture.value}"
                cv2.putText(
                    annotated_rgb,
                    label_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0) if gesture_result.is_valid_gesture() else (200, 200, 200),
                    2,
                    cv2.LINE_AA,
                )

                # Render Frame to Streamlit UI
                image_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

                # Update Gesture UI Metrics
                gesture_metric_placeholder.markdown(
                    f'<div class="gesture-badge">{gesture_result.gesture.display_name}</div>',
                    unsafe_allow_html=True,
                )

                details_placeholder.markdown(
                    f"**Hand Detected**: {'Yes ✅' if detection_result.has_hand else 'No ❌'}  \n"
                    f"**Confidence**: `{gesture_result.confidence * 100:.1f}%`  \n"
                    f"**Hands Count**: `{detection_result.hand_count}` ({detection_result.hand_label})"
                )

                # Finger States Render
                if gesture_result.finger_state:
                    fs = gesture_result.finger_state
                    fingers_html = (
                        f"**Fingers**: "
                        f'<span class="finger-pill {"finger-on" if fs.thumb else "finger-off"}">Thumb</span>'
                        f'<span class="finger-pill {"finger-on" if fs.index else "finger-off"}">Index</span>'
                        f'<span class="finger-pill {"finger-on" if fs.middle else "finger-off"}">Middle</span>'
                        f'<span class="finger-pill {"finger-on" if fs.ring else "finger-off"}">Ring</span>'
                        f'<span class="finger-pill {"finger-on" if fs.pinky else "finger-off"}">Pinky</span>'
                    )
                    fingers_placeholder.markdown(fingers_html, unsafe_allow_html=True)
                else:
                    fingers_placeholder.empty()

                # Webhook Dispatch & Cooldown Logic
                current_time = time.time()
                is_valid = gesture_result.is_valid_gesture()
                
                if is_valid and st.session_state.webhook_url:
                    last_g = st.session_state.last_sent_gesture
                    last_t = st.session_state.last_sent_time
                    cooldown = st.session_state.cooldown_seconds

                    is_new_gesture = (gesture_result.gesture != last_g)
                    cooldown_passed = (current_time - last_t) >= cooldown

                    if is_new_gesture or cooldown_passed:
                        # Send Webhook POST request
                        send_ok, http_code, msg = webhook_client.send_event(
                            st.session_state.webhook_url, gesture_result
                        )
                        st.session_state.last_sent_gesture = gesture_result.gesture
                        st.session_state.last_sent_time = current_time
                        st.session_state.last_webhook_status = msg

                        # Log event
                        now_str = datetime.now().strftime("%H:%M:%S")
                        st.session_state.event_history.append({
                            "gesture": gesture_result.gesture.value,
                            "time": now_str,
                            "status": "Sent OK" if send_ok else f"Failed ({msg})",
                        })
                        history_placeholder.markdown(render_event_history_table())

                # Webhook status display
                if not st.session_state.webhook_url:
                    webhook_status_placeholder.caption("ℹ️ Webhook URL not configured.")
                else:
                    time_remaining = max(0.0, st.session_state.cooldown_seconds - (current_time - st.session_state.last_sent_time))
                    cooldown_text = f" (Cooldown active: {time_remaining:.1f}s)" if time_remaining > 0 else ""
                    webhook_status_placeholder.caption(f"📡 Last Webhook Status: **{st.session_state.last_webhook_status}**{cooldown_text}")

                # Brief sleep to allow Streamlit UI events & CPU headroom
                time.sleep(0.03)

        except Exception as e:
            logger.error("Unexpected error in webcam processing loop: %s", str(e), exc_info=True)
            st.error(f"Runtime error in video processing stream: {str(e)}")
        finally:
            webcam.release()
            detector.close()
            logger.info("Webcam stream released.")
