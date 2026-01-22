"""
Language Detection Service (Control Plane)

Purpose:
- Detect spoken language from raw audio
- Provide metadata for routing and verification policies
- MUST NOT modify audio or generate embeddings

IMPORTANT:
- This service is advisory only
- Output must never affect speaker embeddings
"""

from typing import Optional
import torch
import torchaudio

from speechbrain.inference import EncoderClassifier

from app.services.language.language_detection_result import LanguageDetectionResult


class LanguageDetectionService:
    """
    Singleton-style language detection service using SpeechBrain LID.
    """

    _instance: Optional["LanguageDetectionService"] = None
    _model: Optional[EncoderClassifier] = None

    def __new__(cls) -> "LanguageDetectionService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load_model(self) -> None:
        """
        Load SpeechBrain language identification model.
        """
        if self._model is None:
            self._model = EncoderClassifier.from_hparams(
                source="speechbrain/lang-id-voxlingua107-ecapa",
                savedir="pretrained_models/lang-id-voxlingua107-ecapa"
            )

    @property
    def model(self) -> EncoderClassifier:
        if self._model is None:
            self._load_model()
        return self._model

    def detect_language(
        self,
        waveform: torch.Tensor,
        sample_rate: int
    ) -> LanguageDetectionResult:
        """
        Detect language from raw audio waveform.

        Args:
            waveform: torch.Tensor (mono waveform)
            sample_rate: sampling rate of audio

        Returns:
            LanguageDetectionResult
        """
        if waveform.dim() == 2 and waveform.shape[0] > 1:
            # Ensure mono
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if required (SpeechBrain LID expects 16kHz)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=16000
            )
            waveform = resampler(waveform)

        # SpeechBrain expects shape: [batch, time]
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)

        # Run language classification
        with torch.no_grad():
            prediction = self.model.classify_batch(waveform.unsqueeze(0))

        # Extract top prediction
        predicted_language = prediction[3][0]
        confidence = float(prediction[1][0].max().item())

        return LanguageDetectionResult(
            language=predicted_language,
            confidence=confidence,
            model_name="speechbrain/lang-id-voxlingua107-ecapa"
        )


# Global singleton instance
language_detection_service = LanguageDetectionService()
