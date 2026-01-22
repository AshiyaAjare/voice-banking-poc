"""
Voice authentication API endpoints.

Provides endpoints for:
- Voice enrollment (multi-sample, multi-language)
- Voice verification (with dual model comparison and accent-awareness)
- Enrollment status checking
- Enrollment deletion/cancellation
- Configuration endpoint for frontend
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status

from app.config import get_settings
from app.models.schemas import (
    EnrollmentResponse,
    VerificationResponse,
    UserEnrollmentStatus,
    DeleteEnrollmentResponse,
    CancelEnrollmentResponse,
    ConfigResponse,
    # New accent-aware schemas
    StartEnrollmentRequest,
    StartEnrollmentResponse,
    AccentEnrollmentStatusResponse,
    AccentEnrollmentSampleResponse,
    AccentVerificationResponse,
    LanguageEnrollmentProgress,
)
from app.models.accent import VerificationStrategy
from app.services.voice_service import voice_service
from app.services.accent.accent_profile_service import accent_profile_service
from app.core.crypto.xor_cipher import XORCipher



router = APIRouter(prefix="/voice", tags=["Voice Authentication"])

# Helper method for decryption
def decrypt_audio_bytes(encrypted_bytes: bytes) -> bytes:
    """
    Decrypt XOR-encrypted audio bytes at API boundary.
    """
    settings = get_settings()
    cipher = XORCipher(settings.xor_audio_key)
    return cipher.apply(encrypted_bytes)


# ===========================================================
# Configuration
# ===========================================================

@router.get(
    "/config",
    response_model=ConfigResponse,
    summary="Get configuration",
    description="Get enrollment and verification configuration for the frontend."
)
async def get_config():
    """
    Get configuration settings for the frontend.
    
    Returns enrollment sample requirements and model configuration.
    """
    settings = get_settings()
    
    return ConfigResponse(
        min_enrollment_samples=settings.min_enrollment_samples,
        max_enrollment_samples=settings.max_enrollment_samples,
        enable_secondary_model=False,
        primary_model_name="ECAPA-TDNN",
        secondary_model_name=None,
        similarity_threshold=settings.similarity_threshold
    )


# ===========================================================
# Accent-Aware Enrollment (NEW)
# ===========================================================

@router.post(
    "/enroll/start",
    response_model=StartEnrollmentResponse,
    summary="Start multi-language enrollment",
    description="Initialize a multi-language enrollment session with language preferences."
)
async def start_enrollment(request: StartEnrollmentRequest):
    """
    Start a multi-language enrollment session.
    
    User must declare primary and secondary languages.
    Optional third language can be added.
    """
    try:
        result = accent_profile_service.start_enrollment(
            user_id=request.user_id,
            primary_language=request.primary_language,
            secondary_language=request.secondary_language,
            optional_languages=request.optional_languages,
        )
        
        return StartEnrollmentResponse(
            success=result["success"],
            user_id=result["user_id"],
            message=result["message"],
            primary_language=result["primary_language"],
            secondary_language=result["secondary_language"],
            optional_languages=result.get("optional_languages"),
            samples_required_per_language=result["samples_required_per_language"],
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post(
    "/enroll/sample",
    response_model=AccentEnrollmentSampleResponse,
    summary="Add enrollment sample",
    description="Add a voice sample for a specific language during enrollment."
)
async def add_enrollment_sample(
    user_id: str = Form(..., description="Unique identifier for the user"),
    language: str = Form(..., description="BCP-47 language tag (e.g., hi-IN)"),
    audio: UploadFile = File(..., description="Audio file for enrollment")
):
    """
    Add a voice sample for a specific language bucket.
    
    Must call /enroll/start first to initialize enrollment session.
    """
    # Read audio bytes
    audio_bytes = await audio.read()
    audio_bytes = decrypt_audio_bytes(audio_bytes)
    
    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty audio file provided"
        )
    
    result = accent_profile_service.enroll_user(
        user_id=user_id,
        audio_bytes=audio_bytes,
        target_language=language,
    )
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"]
        )
    
    return AccentEnrollmentSampleResponse(
        success=result["success"],
        user_id=user_id,
        message=result["message"],
        language=result["language"],
        samples_collected=result["samples_collected"],
        samples_required=result["samples_required"],
        language_complete=result["completed"],
        detected_language=result.get("detected_language"),
        detection_confidence=result.get("detection_confidence"),
    )


@router.get(
    "/enroll/status/{user_id}",
    response_model=AccentEnrollmentStatusResponse,
    summary="Get multi-language enrollment status",
    description="Get enrollment progress across all declared languages."
)
async def get_accent_enrollment_status(user_id: str):
    """
    Get enrollment status for all declared languages.
    
    Returns progress for each language bucket.
    """
    result = accent_profile_service.get_enrollment_status(user_id)
    
    if not result.get("success", False):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=result.get("message", "Enrollment not found")
        )
    
    # Build language progress list
    languages = [
        LanguageEnrollmentProgress(
            language_code=lang["language_code"],
            role=lang["role"],
            samples_collected=lang["samples_collected"],
            samples_required=lang["samples_required"],
            is_complete=lang["is_complete"],
        )
        for lang in result.get("languages", [])
    ]
    
    return AccentEnrollmentStatusResponse(
        user_id=user_id,
        primary_language=result["primary_language"],
        secondary_language=result["secondary_language"],
        optional_languages=result.get("optional_languages"),
        languages=languages,
        is_fully_enrolled=result["is_fully_enrolled"],
        can_finalize=result["can_finalize"],
    )


# ===========================================================
# Accent-Aware Verification (NEW)
# ===========================================================

@router.post(
    "/accent/verify",
    response_model=AccentVerificationResponse,
    summary="Accent-aware voice verification",
    description="Verify a user's voice with accent-aware strategies."
)
async def accent_verify(
    user_id: str = Form(..., description="Unique identifier for the user"),
    audio: UploadFile = File(..., description="Audio file for verification"),
    strategy: str = Form(
        default="best_of_all",
        description="Verification strategy: accent_matched | best_of_all | declared_language_fallback"
    )
):
    """
    Accent-aware voice verification.
    
    Strategies:
    - accent_matched: Match against detected language first
    - best_of_all: Try all enrolled languages, pick best score
    - declared_language_fallback: Try in order (primary → secondary → optional)
    """
    settings = get_settings()
    
    # Read audio bytes
    audio_bytes = await audio.read()
    audio_bytes = decrypt_audio_bytes(audio_bytes)
    
    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty audio file provided"
        )
    
    # Parse strategy
    try:
        verification_strategy = VerificationStrategy(strategy)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid strategy: {strategy}. Must be one of: accent_matched, best_of_all, declared_language_fallback"
        )
    
    # Perform verification
    matched, score, details = accent_profile_service.verify_user(
        user_id=user_id,
        audio_bytes=audio_bytes,
        strategy=verification_strategy,
    )
    
    if not details.get("success", True):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=details.get("message", "Verification failed")
        )
    
    decision = details.get("decision")
    dual_scores = details.get("dual_scores", {})
    
    return AccentVerificationResponse(
        matched=matched,
        score=score,
        threshold=settings.similarity_threshold,
        user_id=user_id,
        message=decision.reason if decision else "Verification complete",
        strategy_used=details.get("strategy_used", strategy),
        detected_language=details.get("detected_language"),
        matched_language=details.get("matched_language"),
        confidence_level=decision.confidence_level if decision else None,
        primary_model="ECAPA-TDNN",
        primary_score=dual_scores.get("primary_score", score),
        secondary_model=None,
        secondary_score=dual_scores.get("secondary_score"),
    )


# ===========================================================
# Legacy Endpoints (Backward Compatibility)
# ===========================================================

@router.post(
    "/enroll",
    response_model=EnrollmentResponse,
    summary="Enroll a user's voice (legacy)",
    description="DEPRECATED: Use /enroll/start + /enroll/sample for multi-language enrollment. "
                "This endpoint maintains backward compatibility for single-language enrollment.",
    deprecated=True,
)
async def enroll_voice(
    user_id: str = Form(..., description="Unique identifier for the user"),
    audio: UploadFile = File(..., description="Audio file for enrollment")
):
    """
    Enroll a user by extracting and storing their voice embedding.
    
    DEPRECATED: This endpoint is for backward compatibility only.
    For multi-language enrollment, use /enroll/start + /enroll/sample.
    
    This legacy endpoint uses the default language (en-IN) for enrollment.
    """
    settings = get_settings()
    default_language = getattr(settings, 'default_language', 'en-IN')
    
    # Read audio bytes
    audio_bytes = await audio.read()
    audio_bytes = decrypt_audio_bytes(audio_bytes)
    
    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty audio file provided"
        )
    
    # Check if enrollment was started with new flow
    existing_status = accent_profile_service.get_enrollment_status(user_id)
    
    if existing_status.get("success"):
        # User has a profile, route through AccentProfileService
        # Use primary language from profile
        target_language = existing_status.get("primary_language", default_language)
        
        result = accent_profile_service.enroll_user(
            user_id=user_id,
            audio_bytes=audio_bytes,
            target_language=target_language,
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["message"]
            )
        
        return EnrollmentResponse(
            success=result["success"],
            user_id=user_id,
            message=result["message"],
            enrollment_complete=result["completed"],
            samples_collected=result["samples_collected"],
            samples_required=result["samples_required"]
        )
    
    # No profile - use legacy VoiceService directly
    success, message, enrollment_complete, samples_collected, samples_required = \
        voice_service.enroll_user(user_id, audio_bytes)
    
    if not success and not enrollment_complete:
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
    summary="Verify a user's voice (legacy)",
    description="DEPRECATED: Use /accent/verify with strategy selection. "
                "This endpoint uses best_of_all strategy for backward compatibility.",
    deprecated=True,
)
async def verify_voice(
    user_id: str = Form(..., description="Unique identifier for the user"),
    audio: UploadFile = File(..., description="Audio file for verification")
):
    """
    Verify a user's voice against their enrolled embedding.
    
    DEPRECATED: This endpoint is for backward compatibility only.
    For accent-aware verification, use /accent/verify.
    
    Returns whether the voice matches and similarity scores from both models.
    """
    settings = get_settings()
    
    # Read audio bytes
    audio_bytes = await audio.read()
    audio_bytes = decrypt_audio_bytes(audio_bytes)
    
    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty audio file provided"
        )
    
    # Check if user has accent profile
    existing_status = accent_profile_service.get_enrollment_status(user_id)
    
    if existing_status.get("success") and existing_status.get("is_fully_enrolled"):
        # User has a complete profile, route through AccentProfileService
        matched, score, details = accent_profile_service.verify_user(
            user_id=user_id,
            audio_bytes=audio_bytes,
            strategy=VerificationStrategy.BEST_OF_ALL,
        )
        
        if not details.get("success", True):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=details.get("message", "Verification failed")
            )
        
        dual_scores = details.get("dual_scores", {})
        decision = details.get("decision")
        
        return VerificationResponse(
            matched=matched,
            score=score,
            threshold=settings.similarity_threshold,
            user_id=user_id,
            message=decision.reason if decision else "Verification complete",
            primary_model=dual_scores.get("primary_model", "ECAPA-TDNN"),
            primary_score=dual_scores.get("primary_score", score),
            secondary_model=dual_scores.get("secondary_model"),
            secondary_score=dual_scores.get("secondary_score")
        )
    
    # No accent profile - use legacy VoiceService directly
    matched, score, message, dual_scores = voice_service.verify_user(user_id, audio_bytes)
    
    if "not enrolled" in message.lower():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=message
        )
    
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
        message=message,
        primary_model=dual_scores.get("primary_model", "ECAPA-TDNN"),
        primary_score=dual_scores.get("primary_score", score),
        secondary_model=dual_scores.get("secondary_model"),
        secondary_score=dual_scores.get("secondary_score")
    )


# ===========================================================
# Enrollment Management
# ===========================================================

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
