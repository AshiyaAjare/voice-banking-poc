"""
Speaker model manager supporting SpeechBrain ECAPA-TDNN model with optional WeSpeaker comparison.

Primary: ECAPA-TDNN
Comparison: WeSpeaker (for A/B testing)
"""
import torch
import torchaudio
import io
from typing import Optional, Tuple, Dict

from speechbrain.inference import SpeakerRecognition

from app.config import get_settings


class SpeakerModel:
    """Singleton manager for speaker recognition models with optional WeSpeaker comparison."""
    
    _instance: Optional["SpeakerModel"] = None
    _primary_model: Optional[SpeakerRecognition] = None
    _wespeaker_model = None  # Lazy-loaded WeSpeaker model
    
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
    
    def load_wespeaker_model(self) -> None:
        """Load WeSpeaker model if enabled and not already loaded."""
        settings = get_settings()
        if not settings.enable_wespeaker_comparison:
            return
        
        if self._wespeaker_model is None:
            from app.services.speaker_models.wespeaker_model import wespeaker_model
            self._wespeaker_model = wespeaker_model
            self._wespeaker_model.load_model()
    
    def load_models(self) -> None:
        """Load all configured models."""
        self.load_primary_model()
        self.load_wespeaker_model()
    
    @property
    def primary_model(self) -> SpeakerRecognition:
        """Get the primary model, loading if necessary."""
        if self._primary_model is None:
            self.load_primary_model()
        return self._primary_model
    
    @property
    def model(self) -> SpeakerRecognition:
        """Get the model."""
        return self.primary_model
    
    @property
    def wespeaker(self):
        """Get the WeSpeaker model, loading if necessary."""
        settings = get_settings()
        if not settings.enable_wespeaker_comparison:
            return None
        if self._wespeaker_model is None:
            self.load_wespeaker_model()
        return self._wespeaker_model
    
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
    
    def encode_audio_wespeaker(self, audio_bytes: bytes) -> Optional[torch.Tensor]:
        """
        Extract speaker embedding using WeSpeaker model.
        
        Returns:
            WeSpeaker embedding tensor, or None if WeSpeaker is disabled
        """
        if self.wespeaker is None:
            return None
        return self.wespeaker.encode_audio(audio_bytes)
    
    def encode_audio_dual(self, audio_bytes: bytes) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract speaker embeddings from both ECAPA-TDNN and WeSpeaker in parallel.
        
        Args:
            audio_bytes: Raw audio file bytes
            
        Returns:
            Tuple of (ECAPA embedding, WeSpeaker embedding or None)
        """
        ecapa_embedding = self.encode_audio(audio_bytes)
        wespeaker_embedding = self.encode_audio_wespeaker(audio_bytes)
        return ecapa_embedding, wespeaker_embedding
    
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
    
    def compute_similarity_wespeaker(
        self, 
        embedding1: torch.Tensor, 
        embedding2: torch.Tensor
    ) -> Optional[float]:
        """
        Compute similarity score using WeSpeaker model.
        
        Returns:
            Similarity score, or None if WeSpeaker is disabled
        """
        if self.wespeaker is None:
            return None
        return self.wespeaker.compute_similarity(embedding1, embedding2)
    
    def compute_similarity_dual(
        self,
        primary_emb1: torch.Tensor,
        primary_emb2: torch.Tensor,
        wespeaker_emb1: Optional[torch.Tensor] = None,
        wespeaker_emb2: Optional[torch.Tensor] = None
    ) -> Dict[str, any]:
        """
        Compute similarity scores from both ECAPA-TDNN and WeSpeaker models.
        
        Returns:
            Dict with primary_model, primary_score, wespeaker_model, wespeaker_score
        """
        primary_score = self.primary_model.similarity(primary_emb1, primary_emb2).item()
        
        wespeaker_score = None
        wespeaker_model_name = None
        
        if wespeaker_emb1 is not None and wespeaker_emb2 is not None and self.wespeaker:
            wespeaker_score = self.wespeaker.compute_similarity(wespeaker_emb1, wespeaker_emb2)
            wespeaker_model_name = self.wespeaker.get_model_name()
        
        return {
            "primary_model": "ECAPA-TDNN",
            "primary_score": primary_score,
            "wespeaker_model": wespeaker_model_name,
            "wespeaker_score": wespeaker_score
        }
    
    def get_model_names(self) -> Dict[str, str]:
        """Get the names of configured models."""
        names = {"primary": "ECAPA-TDNN"}
        if self.wespeaker:
            names["wespeaker"] = self.wespeaker.get_model_name()
        return names


# Global instance
speaker_model = SpeakerModel()
