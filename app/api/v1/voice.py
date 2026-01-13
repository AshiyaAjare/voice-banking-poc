"""
Voice authentication API endpoints.

Provides endpoints for:
- Voice enrollment (multi-sample)
- Voice verification
- Enrollment status checking
- Enrollment deletion/cancellation
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status

from app.config import get_settings
from app.models.schemas import (
    EnrollmentResponse,
    VerificationResponse,
    UserEnrollmentStatus,
    DeleteEnrollmentResponse,
    CancelEnrollmentResponse,
)
from app.services.voice_service import voice_service


router = APIRouter(prefix="/voice", tags=["Voice Authentication"])


@router.post(
    "/enroll",
    response_model=EnrollmentResponse,
    summary="Enroll a user's voice",
    description="Upload an audio file to create a voice enrollment for a user. "
                "Multi-sample enrollment: submit 5 samples to complete enrollment."
)
async def enroll_voice(
    user_id: str = Form(..., description="Unique identifier for the user"),
    audio: UploadFile = File(..., description="Audio file for enrollment")
):
    """
    Enroll a user by extracting and storing their voice embedding.
    
    Multi-sample enrollment requires 5 audio samples to create a robust
    speaker profile. Each call to this endpoint adds one sample.
    
    The audio file should contain clear speech from the user being enrolled.
    Supported formats: WAV, MP3, FLAC, OGG.
    """
    # Read audio bytes
    audio_bytes = await audio.read()
    
    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty audio file provided"
        )
    
    # Perform enrollment
    success, message, enrollment_complete, samples_collected, samples_required = \
        voice_service.enroll_user(user_id, audio_bytes)
    
    if not success and not enrollment_complete:
        # Enrollment in progress but failed for this sample
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=message
        )
    
    if not success and "already enrolled" in message.lower():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=message
        )
    
    return EnrollmentResponse(
        success=success,
        user_id=user_id,
        message=message,
        enrollment_complete=enrollment_complete,
        samples_collected=samples_collected,
        samples_required=samples_required
    )


@router.post(
    "/verify",
    response_model=VerificationResponse,
    summary="Verify a user's voice",
    description="Upload an audio file to verify against a user's enrolled voice."
)
async def verify_voice(
    user_id: str = Form(..., description="Unique identifier for the user"),
    audio: UploadFile = File(..., description="Audio file for verification")
):
    """
    Verify a user's voice against their enrolled embedding.
    
    Returns whether the voice matches and the similarity score.
    """
    settings = get_settings()
    
    # Read audio bytes
    audio_bytes = await audio.read()
    
    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty audio file provided"
        )
    
    # Perform verification
    matched, score, message = voice_service.verify_user(user_id, audio_bytes)
    
    # Check if user wasn't enrolled
    if "not enrolled" in message.lower():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=message
        )
    
    # Check if enrollment is incomplete
    if "incomplete" in message.lower():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    return VerificationResponse(
        matched=matched,
        score=score,
        threshold=settings.similarity_threshold,
        user_id=user_id,
        message=message
    )


@router.get(
    "/enrollment/{user_id}",
    response_model=UserEnrollmentStatus,
    summary="Check enrollment status",
    description="Check if a user has an enrolled voice profile and enrollment progress."
)
async def get_enrollment_status(user_id: str):
    """
    Check if a user is enrolled and when they were enrolled.
    
    Also returns enrollment progress for multi-sample enrollment.
    """
    enrolled, created_at, enrollment_complete, samples_collected, samples_required = \
        voice_service.get_enrollment_status(user_id)
    
    return UserEnrollmentStatus(
        enrolled=enrolled,
        user_id=user_id,
        created_at=created_at,
        enrollment_complete=enrollment_complete,
        samples_collected=samples_collected,
        samples_required=samples_required
    )


@router.delete(
    "/enrollment/{user_id}",
    response_model=DeleteEnrollmentResponse,
    summary="Delete enrollment",
    description="Remove a user's voice enrollment (both pending and finalized)."
)
async def delete_enrollment(user_id: str):
    """
    Delete a user's voice enrollment from the system.
    """
    success, message = voice_service.delete_enrollment(user_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=message
        )
    
    return DeleteEnrollmentResponse(
        success=success,
        user_id=user_id,
        message=message
    )


@router.delete(
    "/enrollment/{user_id}/cancel",
    response_model=CancelEnrollmentResponse,
    summary="Cancel pending enrollment",
    description="Cancel an in-progress enrollment and discard collected samples."
)
async def cancel_pending_enrollment(user_id: str):
    """
    Cancel a pending (incomplete) enrollment.
    
    This discards all collected samples without affecting any existing
    finalized enrollment.
    """
    success, message, samples_discarded = voice_service.cancel_pending_enrollment(user_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=message
        )
    
    return CancelEnrollmentResponse(
        success=success,
        user_id=user_id,
        message=message,
        samples_discarded=samples_discarded
    )
