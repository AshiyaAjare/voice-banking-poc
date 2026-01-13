"""
Audio validation service for Voice Activity Detection (VAD) and audio quality checks.

Uses Silero VAD for speech detection and RMS energy analysis for quality validation.
"""

import io
import torch
import torchaudio
from dataclasses import dataclass
from typing import Optional, Tuple

from app.config import get_settings


@dataclass
class AudioValidationResult:
    """Result of audio validation checks."""
    is_valid: bool
    has_speech: bool
    speech_duration_ms: int
    speech_percentage: float
    audio_rms: float
    error_message: Optional[str] = None


class AudioValidator:
    """
    Validates audio quality and speech presence before processing.
    
    Uses Silero VAD for voice activity detection and RMS energy
    analysis to ensure audio contains sufficient speech content.
    """
    
    _instance: Optional["AudioValidator"] = None
    _vad_model: Optional[torch.nn.Module] = None
    _vad_utils: Optional[Tuple] = None
    
    def __new__(cls) -> "AudioValidator":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _load_vad_model(self) -> None:
        """Load Silero VAD model if not already loaded."""
        if self._vad_model is None:
            self._vad_model, self._vad_utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
    
    @property
    def vad_model(self) -> torch.nn.Module:
        """Get the loaded VAD model, loading if necessary."""
        if self._vad_model is None:
            self._load_vad_model()
        return self._vad_model
    
    @property
    def vad_utils(self) -> Tuple:
        """Get the VAD utils, loading if necessary."""
        if self._vad_utils is None:
            self._load_vad_model()
        return self._vad_utils
    
    def _calculate_rms(self, waveform: torch.Tensor) -> float:
        """Calculate Root Mean Square (RMS) energy of audio."""
        return torch.sqrt(torch.mean(waveform ** 2)).item()
    
    def _resample_if_needed(
        self, 
        waveform: torch.Tensor, 
        sample_rate: int, 
        target_rate: int = 16000
    ) -> torch.Tensor:
        """Resample audio to target rate if needed."""
        if sample_rate != target_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, target_rate)
            waveform = resampler(waveform)
        return waveform
    
    def _detect_speech_segments(
        self, 
        waveform: torch.Tensor, 
        sample_rate: int = 16000
    ) -> list:
        """
        Detect speech segments using Silero VAD.
        
        Returns:
            List of dicts with 'start' and 'end' timestamps in seconds
        """
        settings = get_settings()
        
        # Get VAD utils
        (get_speech_timestamps, _, _, _, _) = self.vad_utils
        
        # Ensure mono audio
        if waveform.dim() > 1 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0)
        elif waveform.dim() > 1:
            waveform = waveform.squeeze(0)
        
        # Detect speech timestamps
        speech_timestamps = get_speech_timestamps(
            waveform,
            self.vad_model,
            sampling_rate=sample_rate,
            threshold=settings.vad_threshold,
            min_speech_duration_ms=100,  # Minimum segment duration
            min_silence_duration_ms=100,  # Minimum silence between segments
        )
        
        return speech_timestamps
    
    def _calculate_speech_duration_ms(
        self, 
        speech_timestamps: list, 
        sample_rate: int = 16000
    ) -> int:
        """Calculate total speech duration in milliseconds."""
        total_samples = sum(
            segment['end'] - segment['start'] 
            for segment in speech_timestamps
        )
        return int((total_samples / sample_rate) * 1000)
    
    def validate_audio_bytes(self, audio_bytes: bytes) -> AudioValidationResult:
        """
        Validate audio bytes for speech presence and quality.
        
        Args:
            audio_bytes: Raw audio file bytes
            
        Returns:
            AudioValidationResult with validation status and metrics
        """
        settings = get_settings()
        
        try:
            # Load audio from bytes
            audio_buffer = io.BytesIO(audio_bytes)
            waveform, sample_rate = torchaudio.load(audio_buffer)
            
            # Ensure mono audio
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Calculate RMS energy
            audio_rms = self._calculate_rms(waveform)
            
            # Check minimum RMS threshold
            if audio_rms < settings.min_audio_rms:
                return AudioValidationResult(
                    is_valid=False,
                    has_speech=False,
                    speech_duration_ms=0,
                    speech_percentage=0.0,
                    audio_rms=audio_rms,
                    error_message="Audio too quiet - please speak louder"
                )
            
            # Resample to 16kHz for VAD (Silero requires 16kHz or 8kHz)
            waveform_16k = self._resample_if_needed(waveform, sample_rate, 16000)
            
            # Calculate audio duration in ms
            audio_duration_ms = int((waveform_16k.shape[-1] / 16000) * 1000)
            
            # Detect speech segments
            speech_timestamps = self._detect_speech_segments(waveform_16k, 16000)
            
            # Calculate speech metrics
            speech_duration_ms = self._calculate_speech_duration_ms(speech_timestamps, 16000)
            speech_percentage = speech_duration_ms / audio_duration_ms if audio_duration_ms > 0 else 0.0
            has_speech = len(speech_timestamps) > 0
            
            # Check if audio has sufficient speech
            if not has_speech or speech_duration_ms < settings.vad_min_speech_duration_ms:
                return AudioValidationResult(
                    is_valid=False,
                    has_speech=has_speech,
                    speech_duration_ms=speech_duration_ms,
                    speech_percentage=speech_percentage,
                    audio_rms=audio_rms,
                    error_message="No speech detected in audio - please speak clearly"
                )
            
            # Check speech percentage
            if speech_percentage < settings.vad_min_speech_percentage:
                return AudioValidationResult(
                    is_valid=False,
                    has_speech=True,
                    speech_duration_ms=speech_duration_ms,
                    speech_percentage=speech_percentage,
                    audio_rms=audio_rms,
                    error_message=f"Insufficient speech content ({speech_percentage:.0%}) - please speak for longer"
                )
            
            # All checks passed
            return AudioValidationResult(
                is_valid=True,
                has_speech=True,
                speech_duration_ms=speech_duration_ms,
                speech_percentage=speech_percentage,
                audio_rms=audio_rms,
                error_message=None
            )
            
        except Exception as e:
            return AudioValidationResult(
                is_valid=False,
                has_speech=False,
                speech_duration_ms=0,
                speech_percentage=0.0,
                audio_rms=0.0,
                error_message=f"Audio validation failed: {str(e)}"
            )


# Global instance
audio_validator = AudioValidator()
