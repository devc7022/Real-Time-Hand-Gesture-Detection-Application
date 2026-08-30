from unittest.mock import patch, MagicMock
import pytest
import requests

from src.models.gesture import GestureType, GestureResult
from src.webhook.webhook_client import WebhookClient


@pytest.fixture
def webhook_client():
    return WebhookClient(timeout=3)


def test_validate_url_valid(webhook_client):
    is_valid, url = webhook_client.validate_url("https://webhook.site/test-uuid")
    assert is_valid is True
    assert url == "https://webhook.site/test-uuid"


def test_validate_url_invalid(webhook_client):
    assert webhook_client.validate_url("")[0] is False
    assert webhook_client.validate_url("ftp://example.com")[0] is False
    assert webhook_client.validate_url("not-a-url")[0] is False


@patch("requests.post")
def test_send_event_success(mock_post, webhook_client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_post.return_value = mock_response

    gesture_res = GestureResult(gesture=GestureType.THUMBS_UP, confidence=0.95)
    success, code, msg = webhook_client.send_event("https://example.com/webhook", gesture_res)

    assert success is True
    assert code == 200
    assert "Success" in msg
    mock_post.assert_called_once()


@patch("requests.post")
def test_send_event_timeout(mock_post, webhook_client):
    mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")

    gesture_res = GestureResult(gesture=GestureType.OPEN_PALM)
    success, code, msg = webhook_client.send_event("https://example.com/webhook", gesture_res)

    assert success is False
    assert code is None
    assert "timed out" in msg.lower()


@patch("requests.post")
def test_send_event_server_error(mock_post, webhook_client):
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_post.return_value = mock_response

    gesture_res = GestureResult(gesture=GestureType.PEACE)
    success, code, msg = webhook_client.send_event("https://example.com/webhook", gesture_res)

    assert success is False
    assert code == 500
    assert "HTTP 500" in msg


@patch("requests.post")
def test_test_connection_success(mock_post, webhook_client):
    mock_response = MagicMock()
    mock_response.status_code = 204
    mock_post.return_value = mock_response

    success, code, msg = webhook_client.test_connection("https://example.com/webhook")

    assert success is True
    assert code == 204
    assert "Success" in msg
