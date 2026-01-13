from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class EnrollmentResponse(BaseModel):
    """Response for voice enrollment."""
    success: bool
    user_id: str
    message: str


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


class DeleteEnrollmentResponse(BaseModel):
    """Response for enrollment deletion."""
    success: bool
    user_id: str
    message: str


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str
