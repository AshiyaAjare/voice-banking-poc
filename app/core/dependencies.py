from app.config import get_settings
from app.services.speaker_model import speaker_model


def get_speaker_model():
    """Dependency to get the speaker model."""
    return speaker_model


def get_app_settings():
    """Dependency to get application settings."""
    return get_settings()
