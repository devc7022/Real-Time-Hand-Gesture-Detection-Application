import pytest
from src.models.gesture import GestureType, FingerState
from src.detection.hand_detector import HandDetectionResult
from src.detection.gesture_classifier import GestureClassifier


@pytest.fixture
def classifier():
    return GestureClassifier()


def create_mock_landmarks(
    thumb_ext=False, index_ext=False, middle_ext=False, ring_ext=False, pinky_ext=False
):
    """
    Helper to generate a 21-point mock hand landmark list representing specific finger states.
    Y-axis: 0.0 is top of image, 1.0 is bottom.
    """
    # Base landmarks initialized at mid-screen (y=0.5)
    landmarks = [(0.5, 0.8, 0.0)] * 21  # Wrist at (0.5, 0.8)

    # MCP / PIP base positions
    landmarks[2] = (0.4, 0.6, 0.0)  # Thumb MCP
    landmarks[3] = (0.35, 0.5, 0.0)  # Thumb IP
    landmarks[5] = (0.45, 0.5, 0.0)  # Index MCP
    landmarks[6] = (0.45, 0.4, 0.0)  # Index PIP
    landmarks[9] = (0.5, 0.5, 0.0)  # Middle MCP
    landmarks[10] = (0.5, 0.4, 0.0)  # Middle PIP
    landmarks[13] = (0.55, 0.5, 0.0)  # Ring MCP
    landmarks[14] = (0.55, 0.4, 0.0)  # Ring PIP
    landmarks[17] = (0.6, 0.5, 0.0)  # Pinky MCP
    landmarks[18] = (0.6, 0.4, 0.0)  # Pinky PIP

    # Thumb Tip (4)
    if thumb_ext:
        landmarks[4] = (0.3, 0.2, 0.0)  # Extended upward/outward
    else:
        landmarks[4] = (0.45, 0.55, 0.0)  # Folded

    # Index Tip (8)
    landmarks[8] = (0.45, 0.1, 0.0) if index_ext else (0.45, 0.6, 0.0)

    # Middle Tip (12)
    landmarks[12] = (0.5, 0.1, 0.0) if middle_ext else (0.5, 0.6, 0.0)

    # Ring Tip (16)
    landmarks[16] = (0.55, 0.1, 0.0) if ring_ext else (0.55, 0.6, 0.0)

    # Pinky Tip (20)
    landmarks[20] = (0.6, 0.1, 0.0) if pinky_ext else (0.6, 0.6, 0.0)

    return landmarks


def test_classify_no_hand(classifier):
    detection = HandDetectionResult(has_hand=False)
    result = classifier.classify(detection)
    assert result.gesture == GestureType.NO_HAND
    assert result.confidence == 1.0


def test_classify_open_palm(classifier):
    raw_lm = create_mock_landmarks(thumb_ext=True, index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True)
    detection = HandDetectionResult(has_hand=True, raw_landmarks=raw_lm, confidence=0.95)
    result = classifier.classify(detection)
    assert result.gesture == GestureType.OPEN_PALM
    assert result.confidence == 0.95


def test_classify_fist(classifier):
    raw_lm = create_mock_landmarks(thumb_ext=False, index_ext=False, middle_ext=False, ring_ext=False, pinky_ext=False)
    detection = HandDetectionResult(has_hand=True, raw_landmarks=raw_lm, confidence=0.92)
    result = classifier.classify(detection)
    assert result.gesture == GestureType.FIST


def test_classify_thumbs_up(classifier):
    raw_lm = create_mock_landmarks(thumb_ext=True, index_ext=False, middle_ext=False, ring_ext=False, pinky_ext=False)
    detection = HandDetectionResult(has_hand=True, raw_landmarks=raw_lm, confidence=0.96)
    result = classifier.classify(detection)
    assert result.gesture == GestureType.THUMBS_UP


def test_classify_peace(classifier):
    raw_lm = create_mock_landmarks(thumb_ext=False, index_ext=True, middle_ext=True, ring_ext=False, pinky_ext=False)
    detection = HandDetectionResult(has_hand=True, raw_landmarks=raw_lm, confidence=0.90)
    result = classifier.classify(detection)
    assert result.gesture == GestureType.PEACE


def test_classify_pointing(classifier):
    raw_lm = create_mock_landmarks(thumb_ext=False, index_ext=True, middle_ext=False, ring_ext=False, pinky_ext=False)
    detection = HandDetectionResult(has_hand=True, raw_landmarks=raw_lm, confidence=0.91)
    result = classifier.classify(detection)
    assert result.gesture == GestureType.POINTING
