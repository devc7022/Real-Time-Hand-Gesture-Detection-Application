from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime, timezone


class GestureType(str, Enum):
    """Supported hand gesture types."""
    OPEN_PALM = "OPEN_PALM"
    FIST = "FIST"
    THUMBS_UP = "THUMBS_UP"
    PEACE = "PEACE"
    POINTING = "POINTING"
    NO_HAND = "NO_HAND"
    UNKNOWN = "UNKNOWN"

    @property
    def display_name(self) -> str:
        """User-friendly display name for UI rendering."""
        mapping = {
            GestureType.OPEN_PALM: "Open Palm ✋",
            GestureType.FIST: "Fist ✊",
            GestureType.THUMBS_UP: "Thumbs Up 👍",
            GestureType.PEACE: "Peace ✌️",
            GestureType.POINTING: "Pointing 👈",
            GestureType.NO_HAND: "No Hand Detected 🖐️❌",
            GestureType.UNKNOWN: "Unknown Gesture ❓",
        }
        return mapping.get(self, self.value)


@dataclass(frozen=True)
class FingerState:
    """Extension state for each finger."""
    thumb: bool
    index: bool
    middle: bool
    ring: bool
    pinky: bool

    def to_dict(self) -> Dict[str, bool]:
        return asdict(self)


@dataclass
class GestureResult:
    """Encapsulates gesture detection metadata and result."""
    gesture: GestureType
    confidence: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finger_state: Optional[FingerState] = None
    hand_count: int = 0
    hand_label: str = "Right"  # Left or Right

    def is_valid_gesture(self) -> bool:
        """Returns True if a valid non-empty hand gesture was recognized."""
        return self.gesture not in (GestureType.NO_HAND, GestureType.UNKNOWN)


@dataclass
class WebhookPayload:
    """JSON payload structure for external webhook notifications."""
    event: str
    gesture: str
    timestamp: str
    source: str = "hand-gesture-detector"
    confidence: float = 0.0
    hand_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert payload to JSON-serializable dictionary."""
        return asdict(self)
