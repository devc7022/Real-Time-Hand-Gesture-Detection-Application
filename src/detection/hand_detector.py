from dataclasses import dataclass
import logging
from typing import List, Optional, Tuple, Any
import cv2
import numpy as np
import mediapipe as mp

logger = logging.getLogger("hand_gesture_app.hand_detector")


@dataclass
class HandDetectionResult:
    """Structure holding hand detection output and visualization frame."""
    has_hand: bool
    landmarks: Optional[Any] = None  # NormalizedLandmarkList from MediaPipe
    raw_landmarks: Optional[List[Tuple[float, float, float]]] = None  # [(x, y, z), ...]
    annotated_rgb_frame: Optional[np.ndarray] = None
    hand_count: int = 0
    hand_label: str = "Right"
    confidence: float = 0.0


try:
    import mediapipe.solutions.hands as mp_hands
    import mediapipe.solutions.drawing_utils as mp_drawing
    import mediapipe.solutions.drawing_styles as mp_drawing_styles
except (ImportError, AttributeError):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = getattr(mp.solutions, "drawing_styles", None)


class HandDetector:
    """
    MediaPipe Hands abstraction for detecting 21 3D hand landmark coordinates
    and drawing visual overlays on video frames.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ):
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self._mp_hands = mp_hands
        self._mp_drawing = mp_drawing

        self._hands = self._mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # Drawing specs: cyan connections, magenta landmark joints
        self._landmark_style = self._mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=3)
        self._connection_style = self._mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

    def detect(self, rgb_frame: np.ndarray, draw_landmarks: bool = True) -> HandDetectionResult:
        """
        Processes an RGB image frame to detect hand landmarks.

        Args:
            rgb_frame: Image frame in RGB color space.
            draw_landmarks: Whether to draw landmark connections on the returned frame copy.

        Returns:
            HandDetectionResult containing detection state, landmarks, and visual overlay.
        """
        if rgb_frame is None or rgb_frame.size == 0:
            return HandDetectionResult(has_hand=False, annotated_rgb_frame=rgb_frame)

        annotated_frame = rgb_frame.copy() if draw_landmarks else rgb_frame

        try:
            results = self._hands.process(rgb_frame)
        except Exception as e:
            logger.error("MediaPipe detection error: %s", str(e))
            return HandDetectionResult(has_hand=False, annotated_rgb_frame=annotated_frame)

        if not results.multi_hand_landmarks:
            return HandDetectionResult(
                has_hand=False,
                annotated_rgb_frame=annotated_frame,
                hand_count=0
            )

        hand_count = len(results.multi_hand_landmarks)
        # Select the primary hand (first detected hand)
        primary_hand_landmarks = results.multi_hand_landmarks[0]
        
        # Determine handedness label if available
        hand_label = "Right"
        confidence = 0.9  # Default estimate
        if results.multi_handedness and len(results.multi_handedness) > 0:
            classification = results.multi_handedness[0].classification[0]
            hand_label = classification.label  # "Left" or "Right"
            confidence = float(classification.score)

        # Draw visual landmarks on frame for all detected hands
        if draw_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self._mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._landmark_style,
                    self._connection_style,
                )

        # Convert MediaPipe landmark objects to list of tuples [(x, y, z), ...]
        raw_coords = [
            (lm.x, lm.y, lm.z) for lm in primary_hand_landmarks.landmark
        ]

        return HandDetectionResult(
            has_hand=True,
            landmarks=primary_hand_landmarks,
            raw_landmarks=raw_coords,
            annotated_rgb_frame=annotated_frame,
            hand_count=hand_count,
            hand_label=hand_label,
            confidence=confidence
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        if hasattr(self, "_hands") and self._hands:
            self._hands.close()
