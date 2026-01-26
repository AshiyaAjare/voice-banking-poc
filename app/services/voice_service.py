"""
Voice service for enrollment and verification operations.

Supports multi-sample enrollment where users provide multiple audio samples
that are combined into a robust speaker profile (centroid).

Uses Qdrant vector database for embedding storage and similarity search.
"""
import json
import shutil
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

from app.config import get_settings
from app.services.speaker_models.speaker_model_provider import get_speaker_model
from app.services.embedding_utils import combine_embeddings, validate_embedding
from app.services.audio_validator import audio_validator
from app.services.vectordb.voice_vector_store import voice_vector_store
from app.services.vectordb.collections import ensure_voice_collection


class VoiceService:
    """Service for voice enrollment and verification operations."""
    
    # Schema version for multi-sample enrollment with dual models
    SCHEMA_VERSION = 4  # Bumped for Qdrant migration
    
    def __init__(self):
        self.settings = get_settings()
        self.speaker_model = get_speaker_model()
        self._ensure_storage_dirs()
        # Ensure Qdrant collection exists on startup
        try:
            ensure_voice_collection()
        except Exception as e:
            print(f"Warning: Could not initialize Qdrant collection: {e}")
    
    def _ensure_storage_dirs(self) -> None:
        """Ensure the pending samples directory exists."""
        self.settings.pending_dir.mkdir(parents=True, exist_ok=True)
    
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
        embedding: torch.Tensor,
        wespeaker_embedding: Optional[torch.Tensor] = None
    ) -> Tuple[int, int]:
        """
        Add a voice sample to a user's pending enrollment.
        
        Args:
            user_id: User identifier
            embedding: ECAPA-TDNN embedding
            wespeaker_embedding: Optional WeSpeaker embedding for comparison
        
        Returns:
            Tuple of (samples_collected, samples_required)
        """
        # Load or create pending enrollment
        metadata = self._load_pending_metadata(user_id)
        if metadata is None:
            metadata = self._create_pending_enrollment(user_id)
            
        sample_index = metadata["samples_collected"] + 1
        sample_filename = f"sample_{sample_index}.pt"
        sample_path = self._get_pending_dir(user_id) / sample_filename
        
        # Save sample data with both embeddings
        sample_data = {
            "embedding": embedding,
            "wespeaker_embedding": wespeaker_embedding,
            "collected_at": datetime.utcnow().isoformat()
        }
        torch.save(sample_data, sample_path)
        
        # Update metadata
        metadata["samples_collected"] = sample_index
        metadata["sample_files"].append(sample_filename)
        self._save_pending_metadata(user_id, metadata)
        
        return sample_index, metadata["samples_required"]
    
    def _load_pending_embeddings(
        self, 
        user_id: str
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Load all pending embeddings for a user.
        
        Returns:
            Tuple of (ECAPA embeddings list, WeSpeaker embeddings list)
        """
        metadata = self._load_pending_metadata(user_id)
        if metadata is None:
            return [], []
        
        primary_embeddings = []
        wespeaker_embeddings = []
        pending_dir = self._get_pending_dir(user_id)
        
        for sample_file in metadata["sample_files"]:
            sample_path = pending_dir / sample_file
            if sample_path.exists():
                data = torch.load(sample_path, weights_only=False)
                primary_embeddings.append(data["embedding"])
                if data.get("wespeaker_embedding") is not None:
                    wespeaker_embeddings.append(data["wespeaker_embedding"])
        
        return primary_embeddings, wespeaker_embeddings
    
    def _finalize_enrollment(self, user_id: str) -> Tuple[bool, str]:
        """
        Finalize enrollment by combining pending samples into centroids
        and storing them in Qdrant.
        
        Returns:
            Tuple of (success, message)
        """
        metadata = self._load_pending_metadata(user_id)
        if metadata is None:
            return False, "No pending enrollment found"
        
        # Load all pending embeddings
        primary_embeddings, wespeaker_embeddings = self._load_pending_embeddings(user_id)
        if len(primary_embeddings) < self.settings.min_enrollment_samples:
            return False, f"Not enough samples: {len(primary_embeddings)}/{self.settings.min_enrollment_samples}"
        
        try:
            # Combine ECAPA embeddings into centroid
            centroid = combine_embeddings(primary_embeddings)
            
            # Combine WeSpeaker embeddings into centroid if available
            wespeaker_centroid = None
            if wespeaker_embeddings and len(wespeaker_embeddings) == len(primary_embeddings):
                wespeaker_centroid = combine_embeddings(wespeaker_embeddings)
            
            # Get sample timestamps
            sample_timestamps = []
            pending_dir = self._get_pending_dir(user_id)
            for sample_file in metadata["sample_files"]:
                sample_path = pending_dir / sample_file
                if sample_path.exists():
                    data = torch.load(sample_path, weights_only=False)
                    sample_timestamps.append(data.get("collected_at", ""))
            
            # Convert centroid to list for Qdrant storage
            centroid_list = centroid.squeeze().tolist()
            
            # Store in Qdrant
            embedding_metadata = {
                "version": self.SCHEMA_VERSION,
                "sample_count": len(primary_embeddings),
                "sample_timestamps": sample_timestamps,
                "model": "ECAPA-TDNN",
            }
            
            # Store WeSpeaker centroid as additional metadata if available
            if wespeaker_centroid is not None:
                embedding_metadata["wespeaker_centroid"] = wespeaker_centroid.squeeze().tolist()
            
            # Store in Qdrant
            voice_vector_store.store_embedding(
                user_id=user_id,
                embedding=centroid_list,
                metadata=embedding_metadata
            )
            
            # Cleanup pending directory
            shutil.rmtree(self._get_pending_dir(user_id), ignore_errors=True)
            
            return True, f"Enrollment complete with {len(primary_embeddings)} samples"
            
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
    
    def _is_enrolled(self, user_id: str) -> bool:
        """Check if user is enrolled in Qdrant."""
        return voice_vector_store.user_exists(user_id)
    
    def enroll_user(
        self, 
        user_id: str, 
        audio_bytes: bytes
    ) -> Tuple[bool, str, bool, int, int]:
        """
        Enroll a user by collecting voice samples.
        
        Multi-sample enrollment: collects samples until minimum is reached,
        then automatically finalizes the speaker profile in Qdrant.
        
        Args:
            user_id: Unique identifier for the user
            audio_bytes: Raw audio file bytes
            
        Returns:
            Tuple of (success, message, enrollment_complete, samples_collected, samples_required)
        """
        try:
            # Check if user is already fully enrolled in Qdrant
            if self._is_enrolled(user_id):
                return False, f"User '{user_id}' is already enrolled. Delete existing enrollment first.", True, 0, 0
            
            # Validate audio quality and speech presence
            validation_result = audio_validator.validate_audio_bytes(audio_bytes)
            if not validation_result.is_valid:
                return False, validation_result.error_message, False, 0, 0
            
            # Extract embeddings from both models (parallel computation)
            embedding, wespeaker_embedding = self.speaker_model.encode_audio_dual(audio_bytes)
            
            if not validate_embedding(embedding):
                return False, "Failed to extract valid embedding from audio", False, 0, 0
            
            # Add to pending samples
            samples_collected, samples_required = self._add_pending_sample(
                user_id, embedding, wespeaker_embedding
            )
            
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
    ) -> Tuple[bool, float, str, Dict[str, Any]]:
        """
        Verify a user's voice against their enrollment in Qdrant.
        
        Args:
            user_id: Unique identifier for the user
            audio_bytes: Raw audio file bytes to verify
            
        Returns:
            Tuple of (matched, score, message, dual_model_scores)
            dual_model_scores contains primary_model, primary_score, wespeaker_model, wespeaker_score
        """
        dual_scores = {
            "primary_model": "ECAPA-TDNN",
            "primary_score": 0.0,
            "wespeaker_model": None,
            "wespeaker_score": None
        }
        
        try:
            # Check if user is enrolled in Qdrant
            if not self._is_enrolled(user_id):
                # Check for pending enrollment
                if self._has_pending_enrollment(user_id):
                    metadata = self._load_pending_metadata(user_id)
                    collected = metadata.get("samples_collected", 0)
                    required = metadata.get("samples_required", self.settings.min_enrollment_samples)
                    return False, 0.0, f"User '{user_id}' enrollment is incomplete ({collected}/{required} samples)", dual_scores
                return False, 0.0, f"User '{user_id}' is not enrolled", dual_scores
            
            # Validate audio quality and speech presence
            validation_result = audio_validator.validate_audio_bytes(audio_bytes)
            if not validation_result.is_valid:
                return False, 0.0, validation_result.error_message, dual_scores
            
            # Extract test embeddings from both models (parallel computation)
            test_primary, test_wespeaker = self.speaker_model.encode_audio_dual(audio_bytes)
            
            # Convert to list for Qdrant search
            test_primary_list = test_primary.squeeze().tolist()
            
            # Search in Qdrant - cosine similarity is computed automatically
            results = voice_vector_store.search_by_user(
                user_id=user_id,
                query_embedding=test_primary_list,
                top_k=1
            )
            
            if not results:
                return False, 0.0, f"User '{user_id}' enrollment data not found in vector store", dual_scores
            
            # Qdrant returns cosine similarity score directly
            primary_score = results[0].score
            dual_scores["primary_score"] = primary_score
            
            # Compute WeSpeaker similarity if available
            enrolled_record = results[0]
            wespeaker_centroid_list = enrolled_record.payload.get("wespeaker_centroid")
            
            if wespeaker_centroid_list is not None and test_wespeaker is not None:
                wespeaker_centroid = torch.tensor(wespeaker_centroid_list).unsqueeze(0)
                wespeaker_score = self.speaker_model.compute_similarity_wespeaker(
                    wespeaker_centroid, test_wespeaker
                )
                if wespeaker_score is not None:
                    dual_scores["wespeaker_model"] = "WeSpeaker"
                    dual_scores["wespeaker_score"] = wespeaker_score
            
            # Make decision based on primary model score only
            score = primary_score
            matched = score >= self.settings.similarity_threshold
            
            if matched:
                message = "Voice matched - authentication successful"
            else:
                message = "Voice not matched - authentication failed"
            
            return matched, score, message, dual_scores
            
        except Exception as e:
            return False, 0.0, f"Verification failed: {str(e)}", dual_scores
    
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
        # Check for finalized enrollment in Qdrant
        if self._is_enrolled(user_id):
            try:
                results = voice_vector_store.get_by_user_id(user_id)
                if results:
                    payload = results[0].payload
                    created_at = datetime.fromisoformat(payload.get("created_at", ""))
                    sample_count = payload.get("sample_count", 1)
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
        Delete a user's voice enrollment from Qdrant.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Tuple of (success, message)
        """
        is_enrolled = self._is_enrolled(user_id)
        has_pending = self._has_pending_enrollment(user_id)
        
        if not is_enrolled and not has_pending:
            return False, f"User '{user_id}' is not enrolled"
        
        try:
            # Delete from Qdrant if enrolled
            if is_enrolled:
                voice_vector_store.delete_by_user_id(user_id)
            
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
