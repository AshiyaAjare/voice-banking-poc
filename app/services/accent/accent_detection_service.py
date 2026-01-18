from typing import Optional, Dict
import io

import torch
import torchaudio

from app.models.accent import AccentDetectionResult


class AccentDetectionService:
    """
    Advisory service to detect language / accent from audio.

    This service NEVER makes enrollment or verification decisions.
    It only provides signals that higher-level services may use.
    """

    def __init__(self):
        # Placeholder for future language-id / accent models
        # (e.g., SpeechBrain LID, Whisper, Indic models)
        self.model_name = "heuristic-placeholder"

    def detect_accent(self, audio_bytes: bytes) -> AccentDetectionResult:
        """
        Detect language/accent from raw audio bytes.

        Args:
            audio_bytes: Raw audio file bytes

        Returns:
            AccentDetectionResult
        """
        try:
            waveform, sample_rate = self._load_audio(audio_bytes)

            # -----------------------------------------
            # TEMPORARY HEURISTIC (safe default)
            # -----------------------------------------
            # Until a proper LID model is integrated,
            # we return 'unknown' with low confidence.
            #
            # This ensures the pipeline never breaks.
            # -----------------------------------------
            detected_language = "unknown"
            confidence = 0.0

            return AccentDetectionResult(
                detected_language=detected_language,
                confidence=confidence,
                model_name=self.model_name,
                raw_scores=None
            )

        except Exception:
            # Absolute fallback: never raise from this service
            return AccentDetectionResult(
                detected_language="unknown",
                confidence=0.0,
                model_name=self.model_name,
                raw_scores=None
            )

    # -------------------------------------------------
    # Internal helpers
    # -------------------------------------------------

    def _load_audio(self, audio_bytes: bytes) -> torch.Tensor:
        """
        Load audio from bytes and normalize to mono waveform.
        """
        audio_buffer = io.BytesIO(audio_bytes)
        waveform, sample_rate = torchaudio.load(audio_buffer)

        # Ensure mono
        if waveform.dim() > 1 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        return waveform, sample_rate


# Global singleton (consistent with your existing pattern)
accent_detection_service = AccentDetectionService()
