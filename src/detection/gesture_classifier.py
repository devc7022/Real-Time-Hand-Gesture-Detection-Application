import math
import logging
from typing import List, Tuple, Optional, Any
import numpy as np

from src.models.gesture import GestureType, FingerState, GestureResult
from src.detection.hand_detector import HandDetectionResult

logger = logging.getLogger("hand_gesture_app.gesture_classifier")


class GestureClassifier:
    """
    Deterministic rule-based gesture classifier for MediaPipe hand landmarks.
    Evaluates finger extension states and maps landmark geometry to standard gestures.
    """

    @staticmethod
    def _euclidean_dist(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        """Calculates 3D Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

    def determine_finger_states(
        self, landmarks: List[Tuple[float, float, float]], hand_label: str = "Right"
    ) -> FingerState:
        """
        Determines individual finger extension states (True if extended, False if folded).

        MediaPipe Landmark Indices:
            Wrist: 0
            Thumb: 1(CMC), 2(MCP), 3(IP), 4(TIP)
            Index: 5(MCP), 6(PIP), 7(DIP), 8(TIP)
            Middle: 9(MCP), 10(PIP), 11(DIP), 12(TIP)
            Ring: 13(MCP), 14(PIP), 15(DIP), 16(TIP)
            Pinky: 17(MCP), 18(PIP), 19(DIP), 20(TIP)
        """
        wrist = landmarks[0]

        # Finger tip vs PIP/MCP comparison
        # Index finger
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        index_mcp = landmarks[5]
        index_ext = (index_tip[1] < index_pip[1]) or (
            self._euclidean_dist(wrist, index_tip) > self._euclidean_dist(wrist, index_mcp) * 1.2
        )

        # Middle finger
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        middle_mcp = landmarks[9]
        middle_ext = (middle_tip[1] < middle_pip[1]) or (
            self._euclidean_dist(wrist, middle_tip) > self._euclidean_dist(wrist, middle_mcp) * 1.2
        )

        # Ring finger
        ring_tip = landmarks[16]
        ring_pip = landmarks[14]
        ring_mcp = landmarks[13]
        ring_ext = (ring_tip[1] < ring_pip[1]) or (
            self._euclidean_dist(wrist, ring_tip) > self._euclidean_dist(wrist, ring_mcp) * 1.2
        )

        # Pinky finger
        pinky_tip = landmarks[20]
        pinky_pip = landmarks[18]
        pinky_mcp = landmarks[17]
        pinky_ext = (pinky_tip[1] < pinky_pip[1]) or (
            self._euclidean_dist(wrist, pinky_tip) > self._euclidean_dist(wrist, pinky_mcp) * 1.2
        )

        # Thumb finger extension: distance from pinky MCP or x-offset depending on handedness
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        
        # Check thumb distance to wrist vs IP distance to wrist
        thumb_dist_wrist = self._euclidean_dist(wrist, thumb_tip)
        thumb_ip_dist_wrist = self._euclidean_dist(wrist, thumb_ip)
        
        # For thumbs up, thumb tip is pointing upwards (y coordinate smaller than IP)
        thumb_upward = thumb_tip[1] < thumb_ip[1] < thumb_mcp[1]
        
        # Horizontal extension based on hand label
        if hand_label == "Right":
            thumb_out = thumb_tip[0] < thumb_ip[0]
        else:
            thumb_out = thumb_tip[0] > thumb_ip[0]

        thumb_ext = (thumb_dist_wrist > thumb_ip_dist_wrist * 1.1) and (thumb_upward or thumb_out)

        return FingerState(
            thumb=thumb_ext,
            index=index_ext,
            middle=middle_ext,
            ring=ring_ext,
            pinky=pinky_ext,
        )

    def classify(self, detection_result: HandDetectionResult) -> GestureResult:
        """
        Classifies the current hand landmarks into a specific gesture.

        Args:
            detection_result: Output from HandDetector.

        Returns:
            GestureResult dataclass with gesture type, confidence, and finger states.
        """
        if not detection_result.has_hand or not detection_result.raw_landmarks:
            return GestureResult(
                gesture=GestureType.NO_HAND,
                confidence=1.0,
                hand_count=0,
            )

        landmarks = detection_result.raw_landmarks
        finger_state = self.determine_finger_states(landmarks, detection_result.hand_label)

        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        index_mcp = landmarks[5]
        wrist = landmarks[0]

        # 1. THUMBS_UP: Thumb extended upwards, all 4 fingers folded
        is_thumb_pointing_up = (thumb_tip[1] < thumb_ip[1] < thumb_mcp[1]) and (thumb_tip[1] < index_mcp[1])
        if is_thumb_pointing_up and not finger_state.index and not finger_state.middle and not finger_state.ring and not finger_state.pinky:
            return GestureResult(
                gesture=GestureType.THUMBS_UP,
                confidence=detection_result.confidence,
                finger_state=finger_state,
                hand_count=detection_result.hand_count,
                hand_label=detection_result.hand_label,
            )

        # 2. OPEN_PALM: All 5 fingers extended (or index, middle, ring, pinky all extended)
        if finger_state.index and finger_state.middle and finger_state.ring and finger_state.pinky:
            return GestureResult(
                gesture=GestureType.OPEN_PALM,
                confidence=detection_result.confidence,
                finger_state=finger_state,
                hand_count=detection_result.hand_count,
                hand_label=detection_result.hand_label,
            )

        # 3. PEACE (V sign): Index and Middle extended, Ring and Pinky folded
        if finger_state.index and finger_state.middle and not finger_state.ring and not finger_state.pinky:
            return GestureResult(
                gesture=GestureType.PEACE,
                confidence=detection_result.confidence,
                finger_state=finger_state,
                hand_count=detection_result.hand_count,
                hand_label=detection_result.hand_label,
            )

        # 4. POINTING: Index extended, Middle, Ring, Pinky folded
        if finger_state.index and not finger_state.middle and not finger_state.ring and not finger_state.pinky:
            return GestureResult(
                gesture=GestureType.POINTING,
                confidence=detection_result.confidence,
                finger_state=finger_state,
                hand_count=detection_result.hand_count,
                hand_label=detection_result.hand_label,
            )

        # 5. FIST: All 4 main fingers folded, thumb not pointing up
        if not finger_state.index and not finger_state.middle and not finger_state.ring and not finger_state.pinky:
            return GestureResult(
                gesture=GestureType.FIST,
                confidence=detection_result.confidence,
                finger_state=finger_state,
                hand_count=detection_result.hand_count,
                hand_label=detection_result.hand_label,
            )

        # Fallback UNKNOWN
        return GestureResult(
            gesture=GestureType.UNKNOWN,
            confidence=0.5,
            finger_state=finger_state,
            hand_count=detection_result.hand_count,
            hand_label=detection_result.hand_label,
        )
