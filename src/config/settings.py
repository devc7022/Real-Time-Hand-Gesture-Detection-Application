import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()


@dataclass
class Settings:
    """Application configuration management with environment variable overrides."""

    # Webhook configuration
    WEBHOOK_TIMEOUT: int = 5
    WEBHOOK_COOLDOWN_SECONDS: float = 2.0
    DEFAULT_WEBHOOK_URL: str = ""

    # Camera configuration
    CAMERA_INDEX: int = 0
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    TARGET_FPS: int = 30

    # MediaPipe Hand Detector configuration
    MIN_DETECTION_CONFIDENCE: float = 0.7
    MIN_TRACKING_CONFIDENCE: float = 0.5
    MAX_NUM_HANDS: int = 2

    # Logging configuration
    LOG_LEVEL: str = "INFO"

    def __post_init__(self):
        """Allows environment variables to override default fields upon instantiation."""
        if "WEBHOOK_TIMEOUT" in os.environ:
            self.WEBHOOK_TIMEOUT = int(os.getenv("WEBHOOK_TIMEOUT", str(self.WEBHOOK_TIMEOUT)))
        if "WEBHOOK_COOLDOWN_SECONDS" in os.environ:
            self.WEBHOOK_COOLDOWN_SECONDS = float(os.getenv("WEBHOOK_COOLDOWN_SECONDS", str(self.WEBHOOK_COOLDOWN_SECONDS)))
        if "WEBHOOK_URL" in os.environ:
            self.DEFAULT_WEBHOOK_URL = os.getenv("WEBHOOK_URL", self.DEFAULT_WEBHOOK_URL)
        if "CAMERA_INDEX" in os.environ:
            self.CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", str(self.CAMERA_INDEX)))
        if "FRAME_WIDTH" in os.environ:
            self.FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", str(self.FRAME_WIDTH)))
        if "FRAME_HEIGHT" in os.environ:
            self.FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", str(self.FRAME_HEIGHT)))
        if "TARGET_FPS" in os.environ:
            self.TARGET_FPS = int(os.getenv("TARGET_FPS", str(self.TARGET_FPS)))
        if "MIN_DETECTION_CONFIDENCE" in os.environ:
            self.MIN_DETECTION_CONFIDENCE = float(os.getenv("MIN_DETECTION_CONFIDENCE", str(self.MIN_DETECTION_CONFIDENCE)))
        if "MIN_TRACKING_CONFIDENCE" in os.environ:
            self.MIN_TRACKING_CONFIDENCE = float(os.getenv("MIN_TRACKING_CONFIDENCE", str(self.MIN_TRACKING_CONFIDENCE)))
        if "MAX_NUM_HANDS" in os.environ:
            self.MAX_NUM_HANDS = int(os.getenv("MAX_NUM_HANDS", str(self.MAX_NUM_HANDS)))
        if "LOG_LEVEL" in os.environ:
            self.LOG_LEVEL = os.getenv("LOG_LEVEL", self.LOG_LEVEL).upper()


# Default global settings singleton instance
settings = Settings()

