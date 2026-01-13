from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status

from app.config import get_settings
from app.models.schemas import (
    EnrollmentResponse,
    VerificationResponse,
    UserEnrollmentStatus,
    DeleteEnrollmentResponse,
)
from app.services.voice_service import voice_service


router = APIRouter(prefix="/voice", tags=["Voice Authentication"])


@router.post(
    "/enroll",
    response_model=EnrollmentResponse,
    summary="Enroll a user's voice",
    description="Upload an audio file to create a voice enrollment for a user."
)
async def enroll_voice(
    user_id: str = Form(..., description="Unique identifier for the user"),
    audio: UploadFile = File(..., description="Audio file for enrollment")
):
    """
    Enroll a user by extracting and storing their voice embedding.
    
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
    success, message = voice_service.enroll_user(user_id, audio_bytes)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=message
        )
    
    return EnrollmentResponse(
        success=success,
        user_id=user_id,
        message=message
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
    description="Check if a user has an enrolled voice profile."
)
async def get_enrollment_status(user_id: str):
    """
    Check if a user is enrolled and when they were enrolled.
    """
    enrolled, created_at = voice_service.get_enrollment_status(user_id)
    
    return UserEnrollmentStatus(
        enrolled=enrolled,
        user_id=user_id,
        created_at=created_at
    )


@router.delete(
    "/enrollment/{user_id}",
    response_model=DeleteEnrollmentResponse,
    summary="Delete enrollment",
    description="Remove a user's voice enrollment."
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
