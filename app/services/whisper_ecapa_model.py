"""
Whisper + ECAPA-TDNN Speaker Embedding Integration

Whisper acts as a language-robust acoustic frontend.
ECAPA-TDNN remains the speaker identity extractor.

Design goals:
- Reduce phoneme / language bias
- Improve cross-language speaker verification
- No change to enrollment / verification semantics
"""

import io
from typing import Optional, Tuple

import torch
import torchaudio
import whisper
from speechbrain.inference import SpeakerRecognition

from app.config import get_settings


class WhisperECAPASpeakerModel:
    """
    Speaker model that integrates Whisper as an acoustic frontend
    before ECAPA-TDNN embedding extraction.
    """

    _instance: Optional["WhisperECAPASpeakerModel"] = None

    def __new__(cls) -> "WhisperECAPASpeakerModel":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.settings = get_settings()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._whisper_model = None
        self._ecapa_model = None

    # -------------------------------------------------
    # Model loading
    # -------------------------------------------------

    def _load_whisper(self) -> None:
        if self._whisper_model is None:
            self._whisper_model = whisper.load_model(
                self.settings.whisper_model_name,
                device=self.device
            )

    def _load_ecapa(self) -> None:
        if self._ecapa_model is None:
            self._ecapa_model = SpeakerRecognition.from_hparams(
                source=self.settings.model_source,
                savedir=str(self.settings.model_savedir),
                run_opts={"device": self.device}
            )

    @property
    def whisper_model(self):
        self._load_whisper()
        return self._whisper_model

    @property
    def ecapa_model(self):
        self._load_ecapa()
        return self._ecapa_model

    def load_models(self) -> None:
        """Load all models upfront (matches SpeakerModel interface)."""
        self._load_whisper()
        self._load_ecapa()

    # -------------------------------------------------
    # Audio loading & normalization
    # -------------------------------------------------

    def _load_audio(self, audio_bytes: bytes) -> torch.Tensor:
        """
        Load audio bytes and normalize to mono 16kHz waveform.
        """
        audio_buffer = io.BytesIO(audio_bytes)
        waveform, sample_rate = torchaudio.load(audio_buffer)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample to 16kHz (required by Whisper)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        return waveform

    # -------------------------------------------------
    # Whisper frontend
    # -------------------------------------------------

    def _extract_whisper_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract Whisper encoder representations (not transcription).

        Returns:
            Tensor of shape [T, D]
        """
        with torch.no_grad():
            mel = whisper.log_mel_spectrogram(waveform.squeeze(0))
            mel = mel.to(self.device)

            encoder_output = self.whisper_model.encoder(mel.unsqueeze(0))
            return encoder_output.squeeze(0)

    def _whisper_features_to_waveform(
        self, features: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert Whisper encoder features into a pseudo-waveform
        representation suitable for ECAPA.

        Strategy:
        - Temporal mean pooling
        - Projection into 1D signal
        """
        # [T, D] → [D]
        pooled = torch.mean(features, dim=0)

        # Normalize
        pooled = torch.nn.functional.normalize(pooled, dim=0)

        # Convert to fake waveform: [1, N]
        return pooled.unsqueeze(0)

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def encode_audio(
        self, audio_bytes: bytes
    ) -> torch.Tensor:
        """
        Extract speaker embedding using Whisper + ECAPA.

        Flow:
            Audio → Whisper encoder → normalized features → ECAPA
        """
        waveform = self._load_audio(audio_bytes)

        whisper_features = self._extract_whisper_features(waveform)
        ecapa_input = self._whisper_features_to_waveform(whisper_features)

        embedding = self.ecapa_model.encode_batch(ecapa_input)
        return embedding

    def compute_similarity(
        self, embedding1: torch.Tensor, embedding2: torch.Tensor
    ) -> float:
        """
        Compute cosine similarity using ECAPA backend.
        """
        return self.ecapa_model.similarity(embedding1, embedding2).item()

    def encode_audio_dual(
        self, audio_bytes: bytes
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract speaker embedding using Whisper + ECAPA.

        Note: Whisper+ECAPA does not support secondary model (X-Vector).
        Returns (primary_embedding, None).
        """
        return self.encode_audio(audio_bytes), None

    def encode_audio_secondary(self, audio_bytes: bytes) -> Optional[torch.Tensor]:
        """
        Secondary model not supported in Whisper+ECAPA pipeline.

        Returns None.
        """
        return None

    def is_secondary_enabled(self) -> bool:
        """
        Whisper+ECAPA does not support secondary model.

        Returns False.
        """
        return False

    def compute_similarity_secondary(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor
    ) -> Optional[float]:
        """
        Secondary model not supported in Whisper+ECAPA pipeline.

        Returns None.
        """
        return None


# Global singleton
whisper_ecapa_speaker_model = WhisperECAPASpeakerModel()
