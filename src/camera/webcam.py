import logging
from typing import Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger("hand_gesture_app.webcam")


class WebcamStream:
    """
    Dedicated abstraction for accessing and capturing frames from a laptop webcam using OpenCV.
    Provides safety checks, frame conversion, and clean resource cleanup.
    """

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_running: bool = False

    def start(self) -> bool:
        """
        Initializes and opens the camera capture hardware device.
        Returns True if successful, False if device cannot be accessed.
        """
        if self._is_running and self._cap is not None and self._cap.isOpened():
            logger.info("Webcam stream is already open and running.")
            return True

        logger.info("Initializing webcam at index %d...", self.camera_index)
        try:
            # On Windows, cv2.CAP_DSHOW or cv2.CAP_MSMF can improve opening speed
            self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                # Fallback to default backend if CAP_DSHOW fails
                self._cap = cv2.VideoCapture(self.camera_index)
            
            if not self._cap.isOpened():
                logger.error("Failed to open camera index %d.", self.camera_index)
                self._is_running = False
                return False

            # Request desired capture resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            self._is_running = True
            logger.info("Webcam successfully initialized at %dx%d.", self.width, self.height)
            return True
        except Exception as e:
            logger.error("Unexpected error opening webcam: %s", str(e), exc_info=True)
            self._is_running = False
            return False

    def is_opened(self) -> bool:
        """Check if camera device is active and open."""
        return self._is_running and self._cap is not None and self._cap.isOpened()

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Reads a frame from webcam stream.
        
        Returns:
            Tuple of (success: bool, bgr_frame: np.ndarray, rgb_frame: np.ndarray)
        """
        if not self.is_opened():
            logger.warning("Attempted to read from an unopened camera stream.")
            return False, None, None

        try:
            ret, frame = self._cap.read()
            if not ret or frame is None or frame.size == 0:
                logger.warning("Failed to read frame from webcam.")
                return False, None, None

            # Convert OpenCV BGR format to RGB for MediaPipe and Streamlit
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return True, frame, rgb_frame
        except Exception as e:
            logger.error("Error reading camera frame: %s", str(e))
            return False, None, None

    def release(self) -> None:
        """Safely release webcam hardware resources."""
        if self._cap is not None:
            logger.info("Releasing webcam hardware resource.")
            try:
                self._cap.release()
            except Exception as e:
                logger.error("Error while releasing camera: %s", str(e))
            finally:
                self._cap = None
        self._is_running = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
