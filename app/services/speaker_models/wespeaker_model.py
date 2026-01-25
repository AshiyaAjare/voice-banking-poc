"""
WeSpeaker model wrapper for speaker embedding extraction and similarity.

This module uses WeSpeaker's high-level Speaker class from the CLI module
which handles model loading, feature extraction, and embedding computation.

Requires a model directory with:
- config.yaml: model configuration
- avg_model.pt: model weights
"""

import io
from typing import Optional

import torch
import torchaudio

from app.config import get_settings


class WeSpeakerModel:
    """Singleton manager for WeSpeaker speaker embedding model."""

    _instance: Optional["WeSpeakerModel"] = None
    _speaker: Optional[object] = None  # wespeaker.cli.speaker.Speaker
    _device: str = "cpu"

    def __new__(cls) -> "WeSpeakerModel":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # -----------------------------
    # Model loading
    # -----------------------------

    def load_model(self) -> None:
        """Load WeSpeaker model if not already loaded."""
        if self._speaker is not None:
            return

        settings = get_settings()
        
        # Import WeSpeaker's Speaker class (handles all model loading internally)
        from wespeaker.cli.speaker import Speaker
        
        # WeSpeaker expects a model directory path (not checkpoint file)
        # The directory should contain config.yaml and avg_model.pt
        model_dir = str(settings.wespeaker_model_dir)
        
        print(f"[WeSpeaker] Loading model from: {model_dir}")
        self._speaker = Speaker(model_dir)
        
        # Set device
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._speaker.set_device(self._device)
        
        print(f"[WeSpeaker] Model loaded on {self._device}")

    @property
    def speaker(self):
        """Get the WeSpeaker Speaker instance."""
        if self._speaker is None:
            self.load_model()
        return self._speaker

    # -----------------------------
    # Embedding extraction
    # -----------------------------

    @torch.no_grad()
    def encode_audio(self, audio_bytes: bytes) -> Optional[torch.Tensor]:
        """
        Extract speaker embedding using WeSpeaker.

        Args:
            audio_bytes: Raw audio file bytes
            
        Returns:
            Tensor of shape (embedding_dim,) or None if extraction fails
        """
        # Load audio from bytes
        audio_buffer = io.BytesIO(audio_bytes)
        waveform, sample_rate = torchaudio.load(audio_buffer)
        
        # Use Speaker's extract_embedding_from_pcm method
        embedding = self.speaker.extract_embedding_from_pcm(waveform, sample_rate)
        
        if embedding is None:
            return None
            
        # Normalize the embedding
        embedding = torch.nn.functional.normalize(embedding.unsqueeze(0), dim=-1)
        return embedding

    @torch.no_grad()
    def encode_audio_file(self, file_path: str) -> Optional[torch.Tensor]:
        """Extract embedding from audio file path."""
        embedding = self.speaker.extract_embedding(file_path)
        
        if embedding is None:
            return None
            
        # Normalize the embedding
        embedding = torch.nn.functional.normalize(embedding.unsqueeze(0), dim=-1)
        return embedding

    # -----------------------------
    # Similarity
    # -----------------------------

    def compute_similarity(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
    ) -> float:
        """
        Cosine similarity between two WeSpeaker embeddings.
        
        Uses WeSpeaker's normalized cosine similarity (0-1 range).
        """
        # Ensure embeddings are 1D for cosine similarity
        e1 = embedding1.squeeze()
        e2 = embedding2.squeeze()
        
        # Use WeSpeaker's cosine_similarity method (normalizes to 0-1)
        return self.speaker.cosine_similarity(e1, e2)

    # -----------------------------
    # Metadata
    # -----------------------------

    def get_model_name(self) -> str:
        return "WeSpeaker"


# Global singleton
wespeaker_model = WeSpeakerModel()
