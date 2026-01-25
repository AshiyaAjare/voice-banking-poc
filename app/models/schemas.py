from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class EnrollmentResponse(BaseModel):
    """Response for voice enrollment."""
    success: bool
    user_id: str
    message: str
    # Multi-sample enrollment fields
    enrollment_complete: bool = Field(
        True, 
        description="True if enrollment is finalized with all required samples"
    )
    samples_collected: int = Field(
        1, 
        description="Number of samples collected so far"
    )
    samples_required: int = Field(
        1, 
        description="Total samples required for enrollment"
    )


class VerificationResponse(BaseModel):
    """Response for voice verification with dual model comparison."""
    matched: bool
    score: float = Field(..., description="Primary model similarity score")
    threshold: float = Field(..., description="Threshold used for matching")
    user_id: str
    message: str
    # Dual model comparison fields
    primary_model: str = Field("ECAPA-TDNN", description="Primary model name")
    primary_score: float = Field(..., description="ECAPA-TDNN score")
    secondary_model: Optional[str] = Field(None, description="Deprecated")
    secondary_score: Optional[float] = Field(None, description="Deprecated")
    # WeSpeaker comparison fields
    wespeaker_model: Optional[str] = Field(None, description="WeSpeaker model name for A/B comparison")
    wespeaker_score: Optional[float] = Field(None, description="WeSpeaker similarity score for comparison")


class UserEnrollmentStatus(BaseModel):
    """Status of a user's enrollment."""
    enrolled: bool
    user_id: str
    created_at: Optional[datetime] = None
    # Multi-sample enrollment fields
    enrollment_complete: bool = Field(
        True, 
        description="False if enrollment is in progress"
    )
    samples_collected: Optional[int] = Field(
        None, 
        description="Number of samples collected (if enrollment in progress)"
    )
    samples_required: Optional[int] = Field(
        None, 
        description="Total samples required (if enrollment in progress)"
    )


class DeleteEnrollmentResponse(BaseModel):
    """Response for enrollment deletion."""
    success: bool
    user_id: str
    message: str


class CancelEnrollmentResponse(BaseModel):
    """Response for canceling a pending enrollment."""
    success: bool
    user_id: str
    message: str
    samples_discarded: int = Field(
        0,
        description="Number of pending samples that were discarded"
    )


class ConfigResponse(BaseModel):
    """Configuration settings exposed to frontend."""
    min_enrollment_samples: int = Field(..., description="Minimum samples required for enrollment")
    max_enrollment_samples: int = Field(..., description="Maximum samples allowed for enrollment")
    enable_secondary_model: bool = Field(False, description="Deprecated")
    primary_model_name: str = Field("ECAPA-TDNN", description="Primary model name")
    secondary_model_name: Optional[str] = Field(None, description="Deprecated")
    similarity_threshold: float = Field(..., description="Similarity threshold for verification")
    enable_wespeaker_comparison: bool = Field(False, description="WeSpeaker A/B comparison enabled")


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str


# ---------------------------------------------------------
# Accent-Aware Enrollment & Verification Schemas
# ---------------------------------------------------------

class StartEnrollmentRequest(BaseModel):
    """Request to start a multi-language enrollment session."""
    user_id: str = Field(..., description="Unique identifier for the user")
    primary_language: str = Field(
        ..., 
        description="BCP-47 language tag (e.g., hi-IN)"
    )
    secondary_language: Optional[str] = Field(
        default=None, 
        description="BCP-47 language tag (e.g., en-IN) - optional"
    )
    optional_languages: Optional[List[str]] = Field(
        default=None,
        max_length=1,
        description="Optional additional languages (max 1)"
    )


class StartEnrollmentResponse(BaseModel):
    """Response for enrollment session initialization."""
    success: bool
    user_id: str
    message: str
    primary_language: str
    secondary_language: Optional[str] = None
    optional_languages: Optional[List[str]] = None
    samples_required_per_language: int


class LanguageEnrollmentProgress(BaseModel):
    """Enrollment progress for a single language."""
    language_code: str
    role: str = Field(..., description="primary, secondary, or optional")
    samples_collected: int
    samples_required: int
    is_complete: bool


class AccentEnrollmentStatusResponse(BaseModel):
    """Multi-language enrollment status."""
    user_id: str
    primary_language: str
    secondary_language: Optional[str] = None
    optional_languages: Optional[List[str]] = None
    languages: List[LanguageEnrollmentProgress]
    is_fully_enrolled: bool = Field(
        ..., 
        description="True if primary (and secondary if declared) complete"
    )
    can_finalize: bool = Field(
        ..., 
        description="True if enrollment can be finalized"
    )


class AccentEnrollmentSampleResponse(BaseModel):
    """Response for adding an enrollment sample."""
    success: bool
    user_id: str
    message: str
    language: str
    samples_collected: int
    samples_required: int
    language_complete: bool
    detected_language: Optional[str] = None
    detection_confidence: Optional[float] = None


class AccentVerificationResponse(BaseModel):
    """Response for accent-aware verification."""
    matched: bool
    score: float = Field(..., description="Final fused score")
    threshold: float
    user_id: str
    message: str
    strategy_used: str
    detected_language: Optional[str] = None
    matched_language: Optional[str] = None
    confidence_level: Optional[str] = None
    # Dual model scores
    primary_model: str = Field(default="ECAPA-TDNN", description="Primary model name")
    primary_score: float = Field(..., description="ECAPA-TDNN score")
    secondary_model: Optional[str] = Field(None, description="Deprecated")
    secondary_score: Optional[float] = Field(None, description="Deprecated")
    # WeSpeaker comparison
    wespeaker_model: Optional[str] = Field(None, description="WeSpeaker model name for A/B comparison")
    wespeaker_score: Optional[float] = Field(None, description="WeSpeaker similarity score for comparison")
