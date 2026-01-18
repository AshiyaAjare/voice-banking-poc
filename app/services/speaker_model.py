"""
Speaker model manager supporting dual SpeechBrain models for comparison.

Primary: ECAPA-TDNN (default)
Secondary: X-Vector (optional, for comparison)
"""
import torch
import torchaudio
import io
from typing import Optional, Tuple, Dict
from speechbrain.inference import SpeakerRecognition

from app.config import get_settings


class SpeakerModel:
    """Singleton manager for dual SpeechBrain speaker recognition models."""
    
    _instance: Optional["SpeakerModel"] = None
    _primary_model: Optional[SpeakerRecognition] = None
    _secondary_model: Optional[SpeakerRecognition] = None
    
    def __new__(cls) -> "SpeakerModel":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_primary_model(self) -> None:
        """Load the primary SpeechBrain model (ECAPA-TDNN) if not already loaded."""
        if self._primary_model is None:
            settings = get_settings()
            self._primary_model = SpeakerRecognition.from_hparams(
                source=settings.model_source,
                savedir=str(settings.model_savedir)
            )
    
    def load_secondary_model(self) -> None:
        """Load the secondary SpeechBrain model (X-Vector) if enabled and not loaded."""
        settings = get_settings()
        if settings.enable_secondary_model and self._secondary_model is None:
            self._secondary_model = SpeakerRecognition.from_hparams(
                source=settings.secondary_model_source,
                savedir=str(settings.secondary_model_savedir)
            )
    
    def load_models(self) -> None:
        """Load all configured models."""
        self.load_primary_model()
        self.load_secondary_model()
    
    @property
    def primary_model(self) -> SpeakerRecognition:
        """Get the primary model, loading if necessary."""
        if self._primary_model is None:
            self.load_primary_model()
        return self._primary_model
    
    @property
    def secondary_model(self) -> Optional[SpeakerRecognition]:
        """Get the secondary model if enabled, loading if necessary."""
        settings = get_settings()
        if not settings.enable_secondary_model:
            return None
        if self._secondary_model is None:
            self.load_secondary_model()
        return self._secondary_model
    
    @property
    def model(self) -> SpeakerRecognition:
        """Backward compatible: get the primary model."""
        return self.primary_model
    
    def is_secondary_enabled(self) -> bool:
        """Check if secondary model is enabled."""
        return get_settings().enable_secondary_model
    
    def _load_audio(self, audio_bytes: bytes) -> torch.Tensor:
        """Load and preprocess audio from bytes."""
        audio_buffer = io.BytesIO(audio_bytes)
        waveform, sample_rate = torchaudio.load(audio_buffer)
        
        # Ensure mono audio
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return waveform
    
    def encode_audio(self, audio_bytes: bytes) -> torch.Tensor:
        """
        Extract speaker embedding from audio bytes using primary model.
        
        Args:
            audio_bytes: Raw audio file bytes (supports common formats)
            
        Returns:
            Speaker embedding tensor
        """
        waveform = self._load_audio(audio_bytes)
        embedding = self.primary_model.encode_batch(waveform)
        return embedding
    
    def encode_audio_secondary(self, audio_bytes: bytes) -> Optional[torch.Tensor]:
        """
        Extract speaker embedding from audio bytes using secondary model.
        
        Args:
            audio_bytes: Raw audio file bytes
            
        Returns:
            Speaker embedding tensor, or None if secondary model disabled
        """
        if not self.is_secondary_enabled():
            return None
        
        waveform = self._load_audio(audio_bytes)
        embedding = self.secondary_model.encode_batch(waveform)
        return embedding
    
    def encode_audio_dual(self, audio_bytes: bytes) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract speaker embeddings from both models.
        
        Args:
            audio_bytes: Raw audio file bytes
            
        Returns:
            Tuple of (primary_embedding, secondary_embedding)
            Secondary will be None if disabled
        """
        waveform = self._load_audio(audio_bytes)
        
        primary_embedding = self.primary_model.encode_batch(waveform)
        
        secondary_embedding = None
        if self.is_secondary_enabled():
            secondary_embedding = self.secondary_model.encode_batch(waveform)
        
        return primary_embedding, secondary_embedding
    
    def encode_audio_file(self, file_path: str) -> torch.Tensor:
        """
        Extract speaker embedding from an audio file path.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Speaker embedding tensor
        """
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Ensure mono audio
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        embedding = self.primary_model.encode_batch(waveform)
        return embedding
    
    def compute_similarity(
        self, 
        embedding1: torch.Tensor, 
        embedding2: torch.Tensor
    ) -> float:
        """
        Compute similarity score between two embeddings using primary model.
        
        Args:
            embedding1: First speaker embedding
            embedding2: Second speaker embedding
            
        Returns:
            Similarity score (higher = more similar)
        """
        return self.primary_model.similarity(embedding1, embedding2).item()
    
    def compute_similarity_secondary(
        self, 
        embedding1: torch.Tensor, 
        embedding2: torch.Tensor
    ) -> Optional[float]:
        """
        Compute similarity score using secondary model.
        
        Returns:
            Similarity score, or None if secondary model disabled
        """
        if not self.is_secondary_enabled():
            return None
        return self.secondary_model.similarity(embedding1, embedding2).item()
    
    def compute_similarity_dual(
        self,
        primary_emb1: torch.Tensor,
        primary_emb2: torch.Tensor,
        secondary_emb1: Optional[torch.Tensor] = None,
        secondary_emb2: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute similarity scores from both models.
        
        Returns:
            Dict with 'primary_score' and optionally 'secondary_score'
        """
        result = {
            "primary_model": "ECAPA-TDNN",
            "primary_score": self.primary_model.similarity(primary_emb1, primary_emb2).item()
        }
        
        if self.is_secondary_enabled() and secondary_emb1 is not None and secondary_emb2 is not None:
            result["secondary_model"] = "X-Vector"
            result["secondary_score"] = self.secondary_model.similarity(secondary_emb1, secondary_emb2).item()
        
        return result
    
    def get_model_names(self) -> Dict[str, str]:
        """Get the names of configured models."""
        result = {"primary": "ECAPA-TDNN"}
        if self.is_secondary_enabled():
            result["secondary"] = "X-Vector"
        return result


# Global instance
speaker_model = SpeakerModel()
