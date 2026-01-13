from datetime import datetime
from typing import Optional
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
    """Response for voice verification."""
    matched: bool
    score: float = Field(..., description="Similarity score between embeddings")
    threshold: float = Field(..., description="Threshold used for matching")
    user_id: str
    message: str


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


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str

