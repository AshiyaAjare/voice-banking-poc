"""
Voice service for enrollment and verification operations.

Supports multi-sample enrollment where users provide multiple audio samples
that are combined into a robust speaker profile (centroid).
"""
import json
import shutil
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

from app.config import get_settings
from app.services.speaker_model import speaker_model
from app.services.embedding_utils import combine_embeddings, validate_embedding
from app.services.audio_validator import audio_validator


class VoiceService:
    """Service for voice enrollment and verification operations."""
    
    # Schema version for multi-sample enrollment
    SCHEMA_VERSION = 2
    
    def __init__(self):
        self.settings = get_settings()
        self._ensure_storage_dirs()
    
    def _ensure_storage_dirs(self) -> None:
        """Ensure the storage directories exist."""
        self.settings.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.settings.pending_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_embedding_path(self, user_id: str) -> Path:
        """Get the file path for a user's finalized embedding."""
        return self.settings.embeddings_dir / f"{user_id}.pt"
    
    def _get_pending_dir(self, user_id: str) -> Path:
        """Get the directory path for a user's pending enrollment."""
        return self.settings.pending_dir / user_id
    
    def _get_pending_metadata_path(self, user_id: str) -> Path:
        """Get the metadata file path for pending enrollment."""
        return self._get_pending_dir(user_id) / "metadata.json"
    
    def _has_pending_enrollment(self, user_id: str) -> bool:
        """Check if user has a pending (incomplete) enrollment."""
        return self._get_pending_metadata_path(user_id).exists()
    
    def _load_pending_metadata(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Load pending enrollment metadata."""
        metadata_path = self._get_pending_metadata_path(user_id)
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def _save_pending_metadata(self, user_id: str, metadata: Dict[str, Any]) -> None:
        """Save pending enrollment metadata."""
        metadata_path = self._get_pending_metadata_path(user_id)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _create_pending_enrollment(self, user_id: str) -> Dict[str, Any]:
        """Create a new pending enrollment for a user."""
        pending_dir = self._get_pending_dir(user_id)
        pending_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            "user_id": user_id,
            "started_at": datetime.utcnow().isoformat(),
            "samples_collected": 0,
            "samples_required": self.settings.min_enrollment_samples,
            "sample_files": []
        }
        self._save_pending_metadata(user_id, metadata)
        return metadata
    
    def _add_pending_sample(
        self, 
        user_id: str, 
        embedding: torch.Tensor
    ) -> Tuple[int, int]:
        """
        Add a sample to pending enrollment.
        
        Returns:
            Tuple of (samples_collected, samples_required)
        """
        metadata = self._load_pending_metadata(user_id)
        if metadata is None:
            metadata = self._create_pending_enrollment(user_id)
        
        # Increment sample count
        sample_index = metadata["samples_collected"] + 1
        sample_filename = f"sample_{sample_index}.pt"
        sample_path = self._get_pending_dir(user_id) / sample_filename
        
        # Save the sample embedding
        torch.save({
            "embedding": embedding,
            "collected_at": datetime.utcnow().isoformat(),
            "sample_index": sample_index
        }, sample_path)
        
        # Update metadata
        metadata["samples_collected"] = sample_index
        metadata["sample_files"].append(sample_filename)
        self._save_pending_metadata(user_id, metadata)
        
        return sample_index, metadata["samples_required"]
    
    def _load_pending_embeddings(self, user_id: str) -> List[torch.Tensor]:
        """Load all pending embeddings for a user."""
        metadata = self._load_pending_metadata(user_id)
        if metadata is None:
            return []
        
        embeddings = []
        pending_dir = self._get_pending_dir(user_id)
        
        for sample_file in metadata["sample_files"]:
            sample_path = pending_dir / sample_file
            if sample_path.exists():
                data = torch.load(sample_path, weights_only=False)
                embeddings.append(data["embedding"])
        
        return embeddings
    
    def _finalize_enrollment(self, user_id: str) -> Tuple[bool, str]:
        """
        Finalize enrollment by combining pending samples into a centroid.
        
        Returns:
            Tuple of (success, message)
        """
        metadata = self._load_pending_metadata(user_id)
        if metadata is None:
            return False, "No pending enrollment found"
        
        # Load all pending embeddings
        embeddings = self._load_pending_embeddings(user_id)
        if len(embeddings) < self.settings.min_enrollment_samples:
            return False, f"Not enough samples: {len(embeddings)}/{self.settings.min_enrollment_samples}"
        
        try:
            # Combine embeddings into centroid
            centroid = combine_embeddings(embeddings)
            
            # Get sample timestamps
            sample_timestamps = []
            pending_dir = self._get_pending_dir(user_id)
            for sample_file in metadata["sample_files"]:
                sample_path = pending_dir / sample_file
                if sample_path.exists():
                    data = torch.load(sample_path, weights_only=False)
                    sample_timestamps.append(data.get("collected_at", ""))
            
            # Save the finalized profile
            embedding_path = self._get_embedding_path(user_id)
            torch.save({
                "embedding": centroid,
                "created_at": datetime.utcnow().isoformat(),
                "version": self.SCHEMA_VERSION,
                "sample_count": len(embeddings),
                "sample_timestamps": sample_timestamps
            }, embedding_path)
            
            # Cleanup pending directory
            shutil.rmtree(self._get_pending_dir(user_id), ignore_errors=True)
            
            return True, f"Enrollment complete with {len(embeddings)} samples"
            
        except Exception as e:
            return False, f"Failed to finalize enrollment: {str(e)}"
    
    def _cancel_pending_enrollment(self, user_id: str) -> Tuple[bool, int]:
        """
        Cancel a pending enrollment.
        
        Returns:
            Tuple of (success, samples_discarded)
        """
        metadata = self._load_pending_metadata(user_id)
        if metadata is None:
            return False, 0
        
        samples_discarded = metadata.get("samples_collected", 0)
        
        # Remove pending directory
        shutil.rmtree(self._get_pending_dir(user_id), ignore_errors=True)
        
        return True, samples_discarded
    
    def enroll_user(
        self, 
        user_id: str, 
        audio_bytes: bytes
    ) -> Tuple[bool, str, bool, int, int]:
        """
        Enroll a user by collecting voice samples.
        
        Multi-sample enrollment: collects samples until minimum is reached,
        then automatically finalizes the speaker profile.
        
        Args:
            user_id: Unique identifier for the user
            audio_bytes: Raw audio file bytes
            
        Returns:
            Tuple of (success, message, enrollment_complete, samples_collected, samples_required)
        """
        try:
            # Check if user is already fully enrolled
            if self._get_embedding_path(user_id).exists():
                return False, f"User '{user_id}' is already enrolled. Delete existing enrollment first.", True, 0, 0
            
            # Validate audio quality and speech presence
            validation_result = audio_validator.validate_audio_bytes(audio_bytes)
            if not validation_result.is_valid:
                return False, validation_result.error_message, False, 0, 0
            
            # Extract embedding from audio
            embedding = speaker_model.encode_audio(audio_bytes)
            
            if not validate_embedding(embedding):
                return False, "Failed to extract valid embedding from audio", False, 0, 0
            
            # Add to pending samples
            samples_collected, samples_required = self._add_pending_sample(user_id, embedding)
            
            # Check if we have enough samples to finalize
            if samples_collected >= samples_required:
                success, message = self._finalize_enrollment(user_id)
                return success, message, True, samples_collected, samples_required
            
            # Still collecting samples
            return (
                True, 
                f"Sample {samples_collected} of {samples_required} collected successfully",
                False,
                samples_collected,
                samples_required
            )
            
        except Exception as e:
            return False, f"Enrollment failed: {str(e)}", False, 0, 0
    
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
                # Check for pending enrollment
                if self._has_pending_enrollment(user_id):
                    metadata = self._load_pending_metadata(user_id)
                    collected = metadata.get("samples_collected", 0)
                    required = metadata.get("samples_required", self.settings.min_enrollment_samples)
                    return False, 0.0, f"User '{user_id}' enrollment is incomplete ({collected}/{required} samples)"
                return False, 0.0, f"User '{user_id}' is not enrolled"
            
            # Validate audio quality and speech presence
            validation_result = audio_validator.validate_audio_bytes(audio_bytes)
            if not validation_result.is_valid:
                return False, 0.0, validation_result.error_message
            
            # Load enrolled embedding (works for both legacy and multi-sample format)
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
    
    def get_enrollment_status(
        self, 
        user_id: str
    ) -> Tuple[bool, Optional[datetime], bool, Optional[int], Optional[int]]:
        """
        Check if a user is enrolled and enrollment status.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Tuple of (enrolled, created_at, enrollment_complete, samples_collected, samples_required)
        """
        embedding_path = self._get_embedding_path(user_id)
        
        # Check for finalized enrollment
        if embedding_path.exists():
            try:
                data = torch.load(embedding_path, weights_only=False)
                created_at = datetime.fromisoformat(data.get("created_at", ""))
                sample_count = data.get("sample_count", 1)  # Legacy has 1 sample
                return True, created_at, True, sample_count, sample_count
            except Exception:
                return True, None, True, None, None
        
        # Check for pending enrollment
        if self._has_pending_enrollment(user_id):
            metadata = self._load_pending_metadata(user_id)
            if metadata:
                return (
                    False, 
                    None, 
                    False, 
                    metadata.get("samples_collected", 0),
                    metadata.get("samples_required", self.settings.min_enrollment_samples)
                )
        
        return False, None, True, None, None  # Not enrolled at all
    
    def delete_enrollment(self, user_id: str) -> Tuple[bool, str]:
        """
        Delete a user's voice enrollment.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Tuple of (success, message)
        """
        embedding_path = self._get_embedding_path(user_id)
        has_pending = self._has_pending_enrollment(user_id)
        
        if not embedding_path.exists() and not has_pending:
            return False, f"User '{user_id}' is not enrolled"
        
        try:
            # Delete finalized enrollment if exists
            if embedding_path.exists():
                embedding_path.unlink()
            
            # Delete pending enrollment if exists
            if has_pending:
                shutil.rmtree(self._get_pending_dir(user_id), ignore_errors=True)
            
            return True, f"Enrollment for user '{user_id}' deleted successfully"
        except Exception as e:
            return False, f"Failed to delete enrollment: {str(e)}"
    
    def cancel_pending_enrollment(self, user_id: str) -> Tuple[bool, str, int]:
        """
        Cancel a pending (incomplete) enrollment.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Tuple of (success, message, samples_discarded)
        """
        if not self._has_pending_enrollment(user_id):
            return False, f"No pending enrollment found for user '{user_id}'", 0
        
        success, samples_discarded = self._cancel_pending_enrollment(user_id)
        
        if success:
            return True, f"Pending enrollment cancelled, {samples_discarded} samples discarded", samples_discarded
        else:
            return False, "Failed to cancel pending enrollment", 0


# Global instance
voice_service = VoiceService()
