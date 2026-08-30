# Real-Time Hand Gesture Detection Application

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red.svg)](https://streamlit.io/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-green.svg)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-orange.svg)](https://opencv.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

A production-quality computer vision application that captures live laptop webcam video, detects hand gestures in real time using 3D landmark geometry, renders visual feedback in a Streamlit web dashboard, and dispatches debounced HTTP POST webhook events to external endpoints.

---

## 🌟 Key Features

- **Real-Time Webcam Feed**: Captures frames directly from your laptop webcam via a custom OpenCV hardware abstraction layer.
- **Landmark-Based Gesture Recognition**: Leverages MediaPipe Hands for fast, lightweight CPU gesture classification without requiring large neural networks or GPUs.
- **5 Core Gestures + No-Hand State**: Classifies `OPEN_PALM`, `FIST`, `THUMBS_UP`, `PEACE`, `POINTING`, and handles `NO_HAND`.
- **Configurable Webhook Integration**: Validates HTTP/HTTPS URLs and dispatches JSON payloads to external systems.
- **Duplicate Event Prevention**: Built-in debouncing/cooldown algorithm suppresses repetitive POST calls for sustained gestures.
- **Interactive Streamlit Dashboard**: Dark modern glassmorphism UI displaying real-time metrics, finger state indicators, and webhook delivery status.
- **Containerized & Git-Ready**: Complete with Dockerfile, Docker Compose setup, environment variable configuration, pytest suite, and structured logging.

---

## 🏗️ Architecture & Project Structure

The project follows a clean, modular software design separating camera hardware abstraction, landmark extraction, rule classification, webhook delivery, configuration, and UI rendering.

```
Real-Time Hand Gesture Detection Application/
├── app.py                      # Main Streamlit dashboard & real-time loop
├── requirements.txt            # Pinned Python dependencies
├── Dockerfile                  # Production Docker container manifest
├── docker-compose.yml          # Container orchestration service configuration
├── .dockerignore               # Docker build exclusions
├── .gitignore                  # Git repository ignore rules
├── .env.example                # Environment variable configuration template
├── README.md                   # Technical documentation
│
├── src/                        # Modular application source code
│   ├── __init__.py
│   ├── camera/
│   │   ├── __init__.py
│   │   └── webcam.py           # OpenCV VideoCapture abstraction layer
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── hand_detector.py    # MediaPipe 21 3D landmark extractor & drawer
│   │   └── gesture_classifier.py # Deterministic finger extension rule engine
│   ├── webhook/
│   │   ├── __init__.py
│   │   └── webhook_client.py   # HTTP POST client with URL validation & retries
│   ├── models/
│   │   ├── __init__.py
│   │   └── gesture.py          # Enums, dataclasses, and payload schemas
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py         # Central configuration manager
│   └── utils/
│       ├── __init__.py
│       └── logging_config.py   # Structured logging configuration
│
└── tests/                      # Automated unit test suite
    ├── __init__.py
    ├── test_gesture_classifier.py
    ├── test_webhook_client.py
    └── test_config.py
```

---

## ✋ Supported Gestures

| Gesture Name | UI Icon | Finger State Requirements | Description |
| :--- | :---: | :--- | :--- |
| **`OPEN_PALM`** | ✋ | Thumb, Index, Middle, Ring, Pinky extended | Open hand facing camera |
| **`FIST`** | ✊ | All 5 fingers folded into palm | Closed hand fist |
| **`THUMBS_UP`** | 👍 | Thumb extended upward, Index, Middle, Ring, Pinky folded | Approval gesture |
| **`PEACE`** | ✌️ | Index & Middle fingers extended, Ring & Pinky folded | V-sign / Peace gesture |
| **`POINTING`** | 👈 | Index finger extended, Middle, Ring, Pinky folded | Directional pointing gesture |
| **`NO_HAND`** | 🖐️❌ | No hand detected by landmark extractor | Camera active, no hand in view |
| **`UNKNOWN`** | ❓ | Hand detected, finger state does not match rules | Transition or custom pose |

---

## 🧠 How Gesture Detection Works

Rather than deploying heavy deep-learning classifiers that require dedicated GPUs, this system utilizes a two-stage geometric pipeline optimized for standard laptop CPUs:

1. **Landmark Extraction (`HandDetector`)**:
   - MediaPipe Hands processes RGB image frames and identifies 21 3D hand coordinates ($X, Y, Z$) normalized relative to image dimensions.
   - Key joint landmarks include Wrist (`0`), MCP joints (`2, 5, 9, 13, 17`), PIP joints (`6, 10, 14, 18`), and Tip landmarks (`4, 8, 12, 16, 20`).

2. **Finger Extension Evaluation (`GestureClassifier`)**:
   - For fingers (Index, Middle, Ring, Pinky), a finger is classified as **extended** if its Tip landmark $Y$ coordinate is higher (smaller screen $Y$) than its PIP/MCP joints, or if its Euclidean distance from the Wrist exceeds $1.2\times$ the MCP-Wrist distance.
   - For the Thumb, extension is evaluated using $X$/$Y$ spatial offsets and distance relative to IP/MCP joints based on hand orientation (`Left` vs `Right`).

3. **Deterministic Pattern Matching**:
   - The combined boolean states (`thumb`, `index`, `middle`, `ring`, `pinky`) are passed through deterministic rule evaluations to classify the gesture.

---

## 📡 Webhook Integration & Debouncing

### JSON Event Payload Schema

When a gesture event triggers a webhook dispatch, an HTTP POST request is sent with the following JSON structure:

```json
{
  "event": "gesture_detected",
  "gesture": "THUMBS_UP",
  "timestamp": "2026-08-30T14:30:00.000000+00:00",
  "source": "hand-gesture-detector",
  "confidence": 0.96,
  "hand_count": 1
}
```

### Webhook Connection Test Button Payload

Clicking **Test Webhook Connection** sends a lightweight test payload:

```json
{
  "event": "webhook_test",
  "message": "Webhook connection test from Hand Gesture Detector",
  "timestamp": "2026-08-30T14:30:00.000000+00:00",
  "source": "hand-gesture-detector"
}
```

### Duplicate Prevention (Cooldown / Debouncing)

To prevent firing continuous HTTP requests every video frame (~30 requests/sec):
- **Gesture Switch**: When the user switches to a different gesture (e.g. from `PEACE` to `THUMBS_UP`), an event is dispatched immediately.
- **Sustained Gesture**: If the same gesture is held continuously, duplicate webhook events are suppressed until a configurable cooldown period (e.g. `WEBHOOK_COOLDOWN_SECONDS = 2.0`) has elapsed.
- **No-Hand Exclusion**: `NO_HAND` and `UNKNOWN` states are treated as internal UI states and do not trigger external webhook HTTP requests.

---

## 💻 Local Setup Instructions

### Prerequisites
- Python 3.11+
- Laptop with webcam hardware

### 1. Clone & Environment Setup
```bash
git clone https://github.com/your-username/Real-Time-Hand-Gesture-Detection-Application.git
cd Real-Time-Hand-Gesture-Detection-Application

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables (Optional)
```bash
cp .env.example .env
```

### 4. Run Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🐳 Docker Deployment

### 1. Build Docker Image
```bash
docker build -t hand-gesture-detector .
```

### 2. Run Container
```bash
docker run --rm -p 8501:8501 hand-gesture-detector
```

### 3. Run with Docker Compose
```bash
docker compose up --build
```

### ⚠️ Docker Webcam Hardware Limitations
> [!IMPORTANT]
> - On **Linux hosts**, direct host webcam passthrough into Docker containers is supported by adding `--device /dev/video0:/dev/video0` or enabling device passthrough in `docker-compose.yml`.
> - On **Windows and macOS hosts**, Docker Desktop runs inside a virtual machine (WSL2 / Hyper-V) that does not natively expose host USB webcams to Linux containers without third-party USB passthrough drivers (e.g., `usbipd-win`).
> - **Recommendation**: For local development and full hardware webcam access on Windows/macOS, running the application via local Python environment (`streamlit run app.py`) is recommended.

---

## 🧪 Testing Instructions

The application includes unit tests for gesture classification, webhook URL validation, event dispatching, and configuration parsing.

To execute tests:
```bash
pytest -v
```

Expected test coverage output:
```
tests/test_gesture_classifier.py::test_classify_no_hand PASSED
tests/test_gesture_classifier.py::test_classify_open_palm PASSED
tests/test_gesture_classifier.py::test_classify_fist PASSED
tests/test_gesture_classifier.py::test_classify_thumbs_up PASSED
tests/test_gesture_classifier.py::test_classify_peace PASSED
tests/test_gesture_classifier.py::test_classify_pointing PASSED
tests/test_webhook_client.py::test_validate_url_valid PASSED
tests/test_webhook_client.py::test_validate_url_invalid PASSED
tests/test_webhook_client.py::test_send_event_success PASSED
tests/test_webhook_client.py::test_send_event_timeout PASSED
tests/test_webhook_client.py::test_send_event_server_error PASSED
tests/test_webhook_client.py::test_test_connection_success PASSED
tests/test_config.py::test_settings_defaults PASSED
tests/test_config.py::test_settings_env_override PASSED
```

---

## ⚡ Performance & Optimization Decisions

1. **MediaPipe CPU Inference**: Uses lightweight landmark tracking pipelines (`min_detection_confidence=0.7`, `min_tracking_confidence=0.5`) running smoothly at ~30 FPS on standard dual-core laptop CPUs.
2. **Streamlit Execution Optimization**: Operates frame capture and processing inside a tight while-loop with non-blocking UI sleeping (`time.sleep(0.03)`), preventing CPU starvation while maintaining responsive widget interactions.
3. **No Model Download Overhead**: MediaPipe Hands utilizes bundled Google MediaPipe C++ bindings without requiring runtime model downloading.

---

## 🛡️ Error & Edge Case Handling

- **Webcam Unavailable**: Catches `cv2.VideoCapture` opening failures gracefully and renders friendly UI alerts without crashing Streamlit.
- **Webhook Failure**: HTTP timeouts, connection errors, DNS lookup failures, and 4xx/5xx server errors are logged and displayed as UI status pills without breaking the webcam streaming loop.
- **Invalid URLs**: Malformed or non-HTTP/HTTPS webhook URLs are rejected prior to network dispatch.
- **Multiple Hands**: Primary hand (highest confidence) is selected for classification while visual landmarks are drawn for all hands in view.

---

## 🔮 Future Improvements

- **WebRTC Integration**: Support `streamlit-webrtc` for zero-setup browser webcam streaming inside Docker containers on Windows/macOS.
- **Custom Gesture Training**: Allow users to record landmark samples and train custom KNN gesture classifiers.
- **Configurable Gesture Mappings**: Custom webhook actions mapped to specific gestures.
- **Authentication**: Add Bearer token / HMAC signature verification to webhook HTTP headers.
