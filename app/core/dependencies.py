from app.config import get_settings
from app.services.speaker_models.speaker_model_provider import get_speaker_model as _get_speaker_model



def get_speaker_model():
    """Dependency to get the speaker model (config-driven)."""
    return _get_speaker_model()


def get_app_settings():
    """Dependency to get application settings."""
    return get_settings()
