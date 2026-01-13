import torch
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

from app.config import get_settings
from app.services.speaker_model import speaker_model


class VoiceService:
    """Service for voice enrollment and verification operations."""
    
    def __init__(self):
        self.settings = get_settings()
        self._ensure_storage_dir()
    
    def _ensure_storage_dir(self) -> None:
        """Ensure the embeddings storage directory exists."""
        self.settings.embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_embedding_path(self, user_id: str) -> Path:
        """Get the file path for a user's embedding."""
        return self.settings.embeddings_dir / f"{user_id}.pt"
    
    def enroll_user(self, user_id: str, audio_bytes: bytes) -> Tuple[bool, str]:
        """
        Enroll a user by saving their voice embedding.
        
        Args:
            user_id: Unique identifier for the user
            audio_bytes: Raw audio file bytes
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Extract embedding from audio
            embedding = speaker_model.encode_audio(audio_bytes)
            
            # Save embedding to storage
            embedding_path = self._get_embedding_path(user_id)
            torch.save({
                "embedding": embedding,
                "created_at": datetime.utcnow().isoformat()
            }, embedding_path)
            
            return True, f"User '{user_id}' enrolled successfully"
        except Exception as e:
            return False, f"Enrollment failed: {str(e)}"
    
    def verify_user(
        self, 
        user_id: str, 
        audio_bytes: bytes
    ) -> Tuple[bool, float, str]:
        """
        Verify a user's voice against their enrollment.
        
        Args:
            user_id: Unique identifier for the user
            audio_bytes: Raw audio file bytes to verify
            
        Returns:
            Tuple of (matched, score, message)
        """
        try:
            # Check if user is enrolled
            embedding_path = self._get_embedding_path(user_id)
            if not embedding_path.exists():
                return False, 0.0, f"User '{user_id}' is not enrolled"
            
            # Load enrolled embedding
            data = torch.load(embedding_path, weights_only=False)
            enrolled_embedding = data["embedding"]
            
            # Extract test embedding
            test_embedding = speaker_model.encode_audio(audio_bytes)
            
            # Compute similarity
            score = speaker_model.compute_similarity(enrolled_embedding, test_embedding)
            
            # Make decision
            matched = score >= self.settings.similarity_threshold
            
            if matched:
                message = "Voice matched - authentication successful"
            else:
                message = "Voice not matched - authentication failed"
            
            return matched, score, message
            
        except Exception as e:
            return False, 0.0, f"Verification failed: {str(e)}"
    
    def get_enrollment_status(self, user_id: str) -> Tuple[bool, Optional[datetime]]:
        """
        Check if a user is enrolled and when.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Tuple of (enrolled, created_at)
        """
        embedding_path = self._get_embedding_path(user_id)
        
        if not embedding_path.exists():
            return False, None
        
        try:
            data = torch.load(embedding_path, weights_only=False)
            created_at = datetime.fromisoformat(data.get("created_at", ""))
            return True, created_at
        except Exception:
            return True, None
    
    def delete_enrollment(self, user_id: str) -> Tuple[bool, str]:
        """
        Delete a user's voice enrollment.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Tuple of (success, message)
        """
        embedding_path = self._get_embedding_path(user_id)
        
        if not embedding_path.exists():
            return False, f"User '{user_id}' is not enrolled"
        
        try:
            embedding_path.unlink()
            return True, f"Enrollment for user '{user_id}' deleted successfully"
        except Exception as e:
            return False, f"Failed to delete enrollment: {str(e)}"


# Global instance
voice_service = VoiceService()
