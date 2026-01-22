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


def get_speaker_model() -> "SpeakerModel":
    """
    Get the active speaker model.

    Returns:
        SpeakerModel instance
    """
    from app.services.speaker_model import speaker_model
    return speaker_model
