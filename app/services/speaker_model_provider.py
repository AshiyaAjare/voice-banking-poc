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
    from app.services.speaker_model import SpeakerModel
    from app.services.whisper_ecapa_model import WhisperECAPASpeakerModel


def get_speaker_model() -> "SpeakerModel | WhisperECAPASpeakerModel":
    """
    Get the active speaker model based on configuration.

    Returns:
        SpeakerModel or WhisperECAPASpeakerModel instance
    """
    settings = get_settings()

    if settings.speaker_embedding_backend == "whisper_ecapa":
        from app.services.whisper_ecapa_model import whisper_ecapa_speaker_model
        return whisper_ecapa_speaker_model
    else:
        # Default: standard ECAPA-TDNN + X-Vector
        from app.services.speaker_model import speaker_model
        return speaker_model
