import os
from unittest.mock import patch
from src.config.settings import Settings


def test_settings_defaults():
    config = Settings()
    assert config.WEBHOOK_TIMEOUT == 5
    assert config.WEBHOOK_COOLDOWN_SECONDS == 2.0
    assert config.CAMERA_INDEX == 0
    assert config.LOG_LEVEL == "INFO"


def test_settings_env_override():
    with patch.dict(os.environ, {
        "WEBHOOK_TIMEOUT": "10",
        "WEBHOOK_COOLDOWN_SECONDS": "4.5",
        "CAMERA_INDEX": "1",
        "LOG_LEVEL": "DEBUG"
    }):
        config = Settings()
        assert config.WEBHOOK_TIMEOUT == 10
        assert config.WEBHOOK_COOLDOWN_SECONDS == 4.5
        assert config.CAMERA_INDEX == 1
        assert config.LOG_LEVEL == "DEBUG"
