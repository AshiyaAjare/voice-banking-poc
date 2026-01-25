"""
Speaker Model Provider - Centralized model selection.

Returns the active speaker model based on SPEAKER_EMBEDDING_BACKEND config.

Options:
- "ecapa" (default): Standard ECAPA-TDNN + optional X-Vector
- "whisper_ecapa": Whisper as acoustic frontend + ECAPA-TDNN
"""
from typing import TYPE_CHECKING

from app.config import get_settings

if TYPE_CHECKING:
    from app.services.speaker_models.speaker_model import SpeakerModel


def get_speaker_model() -> "SpeakerModel":
    """
    Get the active speaker model.

    Returns:
        SpeakerModel instance
    """
    from app.services.speaker_models.speaker_model import speaker_model
    return speaker_model


def get_wespeaker_model():
    """
    Get the WeSpeaker model for parallel comparison.

    Returns:
        WeSpeakerModel instance (or None if disabled)
    """
    from app.config import get_settings
    settings = get_settings()
    
    if not settings.enable_wespeaker_comparison:
        return None
    
    from app.services.speaker_models.wespeaker_model import wespeaker_model
    return wespeaker_model
