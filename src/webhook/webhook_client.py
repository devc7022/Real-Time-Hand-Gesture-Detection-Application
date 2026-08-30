import logging
from typing import Tuple, Optional, Dict, Any
from urllib.parse import urlparse
from datetime import datetime, timezone
import requests

from src.models.gesture import GestureResult, WebhookPayload

logger = logging.getLogger("hand_gesture_app.webhook_client")


class WebhookClient:
    """
    Client for validating URLs and dispatching gesture event payloads to external webhooks.
    Includes error handling for network failures, timeouts, and non-2xx HTTP responses.
    """

    def __init__(self, timeout: int = 5):
        self.timeout = timeout

    @staticmethod
    def validate_url(url: str) -> Tuple[bool, str]:
        """
        Validates URL string structure and protocol scheme.

        Returns:
            Tuple of (is_valid: bool, error_message_or_normalized_url: str)
        """
        if not url or not url.strip():
            return False, "Webhook URL cannot be empty."

        clean_url = url.strip()
        try:
            parsed = urlparse(clean_url)
            if parsed.scheme not in ("http", "https"):
                return False, "Invalid URL scheme. URL must begin with http:// or https://"
            if not parsed.netloc:
                return False, "Invalid URL structure. Missing host domain or IP address."
            return True, clean_url
        except Exception as e:
            return False, f"Malformed URL format: {str(e)}"

    def send_event(self, url: str, gesture_result: GestureResult) -> Tuple[bool, Optional[int], str]:
        """
        Sends a gesture detection event payload to the configured webhook URL via HTTP POST.

        Returns:
            Tuple of (success: bool, status_code: Optional[int], status_message: str)
        """
        is_valid, msg = self.validate_url(url)
        if not is_valid:
            logger.warning("Attempted to send webhook event to invalid URL: %s", msg)
            return False, None, msg

        payload = WebhookPayload(
            event="gesture_detected",
            gesture=gesture_result.gesture.value,
            timestamp=gesture_result.timestamp,
            confidence=round(gesture_result.confidence, 3),
            hand_count=gesture_result.hand_count,
        )

        return self._dispatch_post(msg, payload.to_dict())

    def test_connection(self, url: str) -> Tuple[bool, Optional[int], str]:
        """
        Sends a test event payload to verify webhook connectivity.

        Returns:
            Tuple of (success: bool, status_code: Optional[int], status_message: str)
        """
        is_valid, msg = self.validate_url(url)
        if not is_valid:
            return False, None, msg

        payload = {
            "event": "webhook_test",
            "message": "Webhook connection test from Hand Gesture Detector",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "hand-gesture-detector",
        }

        return self._dispatch_post(msg, payload)

    def _dispatch_post(self, url: str, payload_dict: Dict[str, Any]) -> Tuple[bool, Optional[int], str]:
        """Executes HTTP POST request with structured exception handling."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "HandGestureDetector/1.0",
        }
        try:
            logger.info("Sending webhook POST request to %s...", url)
            response = requests.post(
                url,
                json=payload_dict,
                headers=headers,
                timeout=self.timeout,
            )

            status_code = response.status_code
            if 200 <= status_code < 300:
                logger.info("Webhook delivered successfully (HTTP %d).", status_code)
                return True, status_code, f"Success (HTTP {status_code})"
            else:
                logger.warning("Webhook returned error status code HTTP %d.", status_code)
                return False, status_code, f"Server returned error HTTP {status_code}"

        except requests.exceptions.Timeout:
            logger.error("Webhook request timed out after %d seconds.", self.timeout)
            return False, None, f"Request timed out after {self.timeout}s"

        except requests.exceptions.ConnectionError as ce:
            logger.error("Webhook connection error: %s", str(ce))
            return False, None, "Connection failed. Check endpoint URL and network connectivity."

        except requests.exceptions.RequestException as re:
            logger.error("Webhook request failed: %s", str(re))
            return False, None, f"HTTP request failed: {str(re)}"

        except Exception as e:
            logger.error("Unexpected error delivering webhook: %s", str(e), exc_info=True)
            return False, None, f"Unexpected error: {str(e)}"
