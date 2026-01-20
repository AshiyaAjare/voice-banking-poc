from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import json

from app.config import get_settings
from app.services.voice_service import voice_service
from app.services.accent.accent_detection_service import accent_detection_service
from app.services.accent.enrollment_policy import enrollment_policy
from app.services.accent.verification_policy import verification_policy

from app.models.accent import (
    LanguageEnrollmentRequest,
    AccentVerificationAttempt,
    AccentEnrollmentStatus,
    VerificationStrategy,
    LanguageRole,
)


class AccentProfileService:
    """
    Orchestrates accent / language-aware enrollment and verification.

    This service:
    - routes audio to language buckets
    - applies enrollment & verification policies
    - delegates actual work to VoiceService
    - manages enrollment profiles (language preferences)
    """

    PROFILE_SCHEMA_VERSION = 4

    def __init__(self):
        self.settings = get_settings()

    # -------------------------------------------------
    # Profile Storage Helpers
    # -------------------------------------------------

    def _get_user_dir(self, user_id: str) -> Path:
        """Get user's base directory in embeddings storage."""
        return self.settings.embeddings_dir / user_id

    def _get_profile_path(self, user_id: str) -> Path:
        """Get path to user's profile.json."""
        return self._get_user_dir(user_id) / "profile.json"

    def _load_profile(self, user_id: str) -> Optional[Dict]:
        """Load user's enrollment profile if it exists."""
        profile_path = self._get_profile_path(user_id)
        if not profile_path.exists():
            return None
        try:
            with open(profile_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def _save_profile(self, user_id: str, profile: Dict) -> None:
        """Save user's enrollment profile (creates directories lazily)."""
        user_dir = self._get_user_dir(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        profile_path = self._get_profile_path(user_id)
        with open(profile_path, "w") as f:
            json.dump(profile, f, indent=2, default=str)

    # -------------------------------------------------
    # Enrollment Lifecycle
    # -------------------------------------------------

    def start_enrollment(
        self,
        user_id: str,
        primary_language: str,
        secondary_language: Optional[str] = None,
        optional_languages: Optional[List[str]] = None,
    ) -> Dict:
        """
        Initialize a multi-language enrollment session.

        Stores enrollment intent in profile.json. Does NOT start
        collecting samples - that happens via enroll_user().

        Args:
            user_id: user identifier
            primary_language: BCP-47 tag (e.g., hi-IN)
            secondary_language: BCP-47 tag (e.g., en-IN) - optional
            optional_languages: list of optional languages (max 1)

        Returns:
            dict with session initialization status
        """
        # 1. Create a LanguageEnrollmentRequest for validation
        enrollment_request = LanguageEnrollmentRequest(
            primary_language=primary_language,
            secondary_language=secondary_language,
            optional_languages=optional_languages,
        )

        # 2. Validate language selection via policy
        enrollment_policy.validate_language_selection(enrollment_request)

        # 3. Check if user already has a profile
        existing_profile = self._load_profile(user_id)
        if existing_profile:
            # Allow re-initialization (user can restart enrollment)
            pass

        # 4. Create enrollment profile
        profile = {
            "user_id": user_id,
            "primary_language": primary_language,
            "secondary_language": secondary_language,
            "optional_languages": optional_languages or [],
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "schema_version": self.PROFILE_SCHEMA_VERSION,
        }

        # 5. Save profile (creates directory lazily)
        self._save_profile(user_id, profile)

        return {
            "success": True,
            "user_id": user_id,
            "message": "Enrollment session initialized",
            "primary_language": primary_language,
            "secondary_language": secondary_language,
            "optional_languages": optional_languages or [],
            "samples_required_per_language": enrollment_policy.min_samples_per_language,
        }

    def get_enrollment_status(self, user_id: str) -> Dict:
        """
        Get enrollment status across all declared languages.

        Returns progress for each language bucket by querying
        VoiceService for each scoped user ID.

        Args:
            user_id: user identifier

        Returns:
            dict with multi-language enrollment status
        """
        # 1. Load profile
        profile = self._load_profile(user_id)
        if not profile:
            return {
                "success": False,
                "user_id": user_id,
                "message": "No enrollment profile found. Call start_enrollment first.",
                "languages": [],
                "is_fully_enrolled": False,
                "can_finalize": False,
            }

        primary_language = profile.get("primary_language")
        secondary_language = profile.get("secondary_language")
        optional_languages = profile.get("optional_languages", [])

        # 2. Build list of all languages to check
        all_languages = []
        if primary_language:
            all_languages.append((primary_language, LanguageRole.PRIMARY.value))
        if secondary_language:
            all_languages.append((secondary_language, LanguageRole.SECONDARY.value))
        for lang in optional_languages:
            all_languages.append((lang, LanguageRole.OPTIONAL.value))

        # 3. Get status for each language from VoiceService
        language_statuses = []
        enrollment_statuses_for_policy = []

        for lang_code, role in all_languages:
            scoped_user_id = f"{user_id}:{lang_code}"

            enrolled, _, enrollment_complete, samples_collected, samples_required = \
                voice_service.get_enrollment_status(scoped_user_id)

            # For pending enrollments, samples_collected may be None
            collected = samples_collected or 0
            required = samples_required or enrollment_policy.min_samples_per_language

            is_complete = enrolled and enrollment_complete

            language_statuses.append({
                "language_code": lang_code,
                "role": role,
                "samples_collected": collected,
                "samples_required": required,
                "is_complete": is_complete,
            })

            # Build policy-compatible status objects
            enrollment_statuses_for_policy.append(
                AccentEnrollmentStatus(
                    language_code=lang_code,
                    samples_collected=collected,
                    samples_required=required,
                )
            )

        # 4. Check if user enrollment can be finalized
        can_finalize = enrollment_policy.can_finalize_user_enrollment(
            enrollment_statuses=enrollment_statuses_for_policy,
            primary_language=primary_language,
            secondary_language=secondary_language,  # Can be None
        )

        # 5. Check if fully enrolled (primary + secondary if declared complete)
        required_languages = [primary_language]
        if secondary_language:
            required_languages.append(secondary_language)
        
        is_fully_enrolled = enrollment_policy.is_user_enrollment_complete(
            enrollment_statuses=enrollment_statuses_for_policy,
            required_languages=required_languages,
        )

        return {
            "success": True,
            "user_id": user_id,
            "primary_language": primary_language,
            "secondary_language": secondary_language,
            "optional_languages": optional_languages,
            "languages": language_statuses,
            "is_fully_enrolled": is_fully_enrolled,
            "can_finalize": can_finalize,
        }

    # -------------------------------------------------
    # Enrollment (Sample Collection)
    # -------------------------------------------------

    def enroll_user(
        self,
        user_id: str,
        audio_bytes: bytes,
        target_language: str,
    ) -> Dict:
        """
        Enroll a user for a specific language bucket.

        Args:
            user_id: user identifier
            audio_bytes: raw audio
            target_language: language user is currently enrolling for

        Returns:
            dict with enrollment status
        """
        # 1. Load profile to verify enrollment was started
        profile = self._load_profile(user_id)
        if not profile:
            return {
                "success": False,
                "message": "Enrollment not started. Call start_enrollment first.",
                "language": target_language,
                "samples_collected": 0,
                "samples_required": enrollment_policy.min_samples_per_language,
                "completed": False,
                "detected_language": None,
                "detection_confidence": None,
            }

        # 2. Verify target_language is in the declared languages
        allowed_languages = [
            profile.get("primary_language"),
            profile.get("secondary_language"),
        ] + profile.get("optional_languages", [])

        if target_language not in allowed_languages:
            return {
                "success": False,
                "message": f"Language '{target_language}' not in declared languages: {allowed_languages}",
                "language": target_language,
                "samples_collected": 0,
                "samples_required": enrollment_policy.min_samples_per_language,
                "completed": False,
                "detected_language": None,
                "detection_confidence": None,
            }

        # 3. Advisory accent detection (non-blocking)
        detection_result = accent_detection_service.detect_accent(audio_bytes)

        # NOTE:
        # User-declared language is authoritative.
        # Detection is logged/returned only for observability.

        # 4. Delegate to VoiceService using language-scoped user_id
        scoped_user_id = f"{user_id}:{target_language}"

        success, message, completed, collected, required = voice_service.enroll_user(
            scoped_user_id,
            audio_bytes,
        )

        # 5. Update profile timestamp
        if success:
            profile["updated_at"] = datetime.utcnow().isoformat()
            self._save_profile(user_id, profile)

        return {
            "success": success,
            "message": message,
            "language": target_language,
            "samples_collected": collected,
            "samples_required": required,
            "completed": completed,
            "detected_language": detection_result.detected_language,
            "detection_confidence": detection_result.confidence,
        }

    # -------------------------------------------------
    # Verification
    # -------------------------------------------------

    def verify_user(
        self,
        user_id: str,
        audio_bytes: bytes,
        strategy: VerificationStrategy = VerificationStrategy.BEST_OF_ALL,
    ) -> Tuple[bool, float, Dict]:
        """
        Verify a user's voice using accent-aware strategies.

        Args:
            user_id: user identifier
            audio_bytes: raw audio
            strategy: verification strategy

        Returns:
            matched, score, details
        """
        # 1. Load profile to get enrolled languages
        profile = self._load_profile(user_id)
        if not profile:
            return False, 0.0, {
                "success": False,
                "message": "No enrollment profile found",
                "detected_language": None,
            }

        # 2. Build enrolled_languages dict
        enrolled_languages: Dict[str, str] = {}
        
        primary_language = profile.get("primary_language")
        secondary_language = profile.get("secondary_language")
        optional_languages = profile.get("optional_languages", [])

        # Check which languages are actually enrolled (have finalized embeddings)
        for lang, role in [
            (primary_language, LanguageRole.PRIMARY.value),
            (secondary_language, LanguageRole.SECONDARY.value),
        ] + [(l, LanguageRole.OPTIONAL.value) for l in optional_languages]:
            if lang:
                scoped_user_id = f"{user_id}:{lang}"
                enrolled, _, enrollment_complete, _, _ = voice_service.get_enrollment_status(scoped_user_id)
                if enrolled and enrollment_complete:
                    enrolled_languages[lang] = role

        if not enrolled_languages:
            return False, 0.0, {
                "success": False,
                "message": "No languages enrolled. Complete enrollment first.",
                "detected_language": None,
            }

        # 3. Detect accent (advisory)
        detection = accent_detection_service.detect_accent(audio_bytes)
        detected_language = detection.detected_language

        language_scores: Dict[str, Tuple[float, Optional[float]]] = {}
        best_dual_scores: Dict = {}

        # 4. Strategy: Accent-matched first
        if (
            strategy == VerificationStrategy.ACCENT_MATCHED
            and detected_language in enrolled_languages
        ):
            scoped_user_id = f"{user_id}:{detected_language}"

            matched, score, message, dual_scores = voice_service.verify_user(
                scoped_user_id,
                audio_bytes,
            )

            decision = verification_policy.decide_accent_matched(
                language_code=detected_language,
                primary_score=dual_scores["primary_score"],
                secondary_score=dual_scores.get("secondary_score"),
            )

            return decision.matched, score, {
                "success": True,
                "decision": decision,
                "detected_language": detected_language,
                "matched_language": detected_language if decision.matched else None,
                "strategy_used": strategy.value,
                "dual_scores": dual_scores,
            }

        # 5. Fallback: try all enrolled languages
        for language in enrolled_languages.keys():
            scoped_user_id = f"{user_id}:{language}"

            try:
                matched, score, _, dual_scores = voice_service.verify_user(
                    scoped_user_id,
                    audio_bytes,
                )

                language_scores[language] = (
                    dual_scores["primary_score"],
                    dual_scores.get("secondary_score"),
                )
                
                # Track best scores for response
                if not best_dual_scores or dual_scores["primary_score"] > best_dual_scores.get("primary_score", 0):
                    best_dual_scores = dual_scores
            except Exception:
                # Skip languages that fail verification
                continue

        if not language_scores:
            return False, 0.0, {
                "success": False,
                "message": "Verification failed for all enrolled languages",
                "detected_language": detected_language,
            }

        # 6. Apply policy decision
        if strategy == VerificationStrategy.BEST_OF_ALL:
            decision = verification_policy.decide_best_of_all(language_scores)
        else:
            decision = verification_policy.decide_declared_language_fallback(
                language_scores
            )

        # Calculate final score from decision
        final_score = 0.0
        if decision.language_used and decision.language_used in language_scores:
            primary, secondary = language_scores[decision.language_used]
            final_score = verification_policy.fuse_scores(primary, secondary)

        return decision.matched, final_score, {
            "success": True,
            "decision": decision,
            "detected_language": detected_language,
            "matched_language": decision.language_used if decision.matched else None,
            "strategy_used": strategy.value,
            "dual_scores": best_dual_scores,
        }


# Global singleton
accent_profile_service = AccentProfileService()
