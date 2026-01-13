import torch
import torchaudio
import io
from typing import Optional
from speechbrain.inference import SpeakerRecognition

from app.config import get_settings


class SpeakerModel:
    """Singleton manager for the SpeechBrain speaker recognition model."""
    
    _instance: Optional["SpeakerModel"] = None
    _model: Optional[SpeakerRecognition] = None
    
    def __new__(cls) -> "SpeakerModel":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_model(self) -> None:
        """Load the SpeechBrain model if not already loaded."""
        if self._model is None:
            settings = get_settings()
            self._model = SpeakerRecognition.from_hparams(
                source=settings.model_source,
                savedir=str(settings.model_savedir)
            )
    
    @property
    def model(self) -> SpeakerRecognition:
        """Get the loaded model, loading if necessary."""
        if self._model is None:
            self.load_model()
        return self._model
    
    def encode_audio(self, audio_bytes: bytes) -> torch.Tensor:
        """
        Extract speaker embedding from audio bytes.
        
        Args:
            audio_bytes: Raw audio file bytes (supports common formats)
            
        Returns:
            Speaker embedding tensor
        """
        # Load audio from bytes
        audio_buffer = io.BytesIO(audio_bytes)
        waveform, sample_rate = torchaudio.load(audio_buffer)
        
        # Ensure mono audio
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Extract embedding
        embedding = self.model.encode_batch(waveform)
        return embedding
    
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
        
        # Extract embedding
        embedding = self.model.encode_batch(waveform)
        return embedding
    
    def compute_similarity(
        self, 
        embedding1: torch.Tensor, 
        embedding2: torch.Tensor
    ) -> float:
        """
        Compute similarity score between two embeddings.
        
        Args:
            embedding1: First speaker embedding
            embedding2: Second speaker embedding
            
        Returns:
            Similarity score (higher = more similar)
        """
        return self.model.similarity(embedding1, embedding2).item()


# Global instance
speaker_model = SpeakerModel()
