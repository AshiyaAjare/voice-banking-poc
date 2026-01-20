from typing import List, Dict, Optional

from app.models.accent import (
    LanguageEnrollmentRequest,
    AccentEnrollmentStatus,
)


class EnrollmentPolicy:
    """
    Defines rules for accent / language-based voice enrollment.

    This class contains NO side effects.
    It only evaluates rules and returns decisions.
    """

    def __init__(
        self,
        min_samples_per_language: int = 3,
        max_optional_languages: int = 1,
    ):
        self.min_samples_per_language = min_samples_per_language
        self.max_optional_languages = max_optional_languages

    # -------------------------------------------------
    # Language selection validation
    # -------------------------------------------------

    def validate_language_selection(
        self,
        request: LanguageEnrollmentRequest,
    ) -> None:
        """
        Validate user-declared language selection.

        Raises:
            ValueError if selection is invalid
        """
        # Only validate secondary language uniqueness if it's provided
        if request.secondary_language and request.primary_language == request.secondary_language:
            raise ValueError("Primary and secondary languages must be different")

        optional = request.optional_languages or []

        if len(optional) > self.max_optional_languages:
            raise ValueError(
                f"At most {self.max_optional_languages} optional language(s) allowed"
            )

        if request.primary_language in optional:
            raise ValueError("Primary language cannot be optional")

        if request.secondary_language and request.secondary_language in optional:
            raise ValueError("Secondary language cannot be optional")

    # -------------------------------------------------
    # Enrollment progress evaluation
    # -------------------------------------------------

    def get_language_enrollment_status(
        self,
        language_code: str,
        samples_collected: int,
    ) -> AccentEnrollmentStatus:
        """
        Evaluate enrollment status for a single language.
        """
        return AccentEnrollmentStatus(
            language_code=language_code,
            samples_collected=samples_collected,
            samples_required=self.min_samples_per_language,
        )

    def is_language_enrollment_complete(
        self,
        samples_collected: int,
    ) -> bool:
        """
        Check if a language has enough samples to finalize enrollment.
        """
        return samples_collected >= self.min_samples_per_language

    # -------------------------------------------------
    # User-level enrollment evaluation
    # -------------------------------------------------

    def is_user_enrollment_complete(
        self,
        enrollment_statuses: List[AccentEnrollmentStatus],
        required_languages: List[str],
    ) -> bool:
        """
        Check if user enrollment is complete across required languages.

        Required languages are typically:
        - primary
        - secondary
        Optional languages do not block completion.
        """
        status_by_language: Dict[str, AccentEnrollmentStatus] = {
            status.language_code: status
            for status in enrollment_statuses
        }

        for lang in required_languages:
            status = status_by_language.get(lang)
            if status is None or not status.is_complete:
                return False

        return True

    # -------------------------------------------------
    # Finalization rules
    # -------------------------------------------------

    def can_finalize_language_enrollment(
        self,
        samples_collected: int,
    ) -> bool:
        """
        Determine whether a language enrollment can be finalized.
        """
        return samples_collected >= self.min_samples_per_language

    def can_finalize_user_enrollment(
        self,
        enrollment_statuses: List[AccentEnrollmentStatus],
        primary_language: str,
        secondary_language: Optional[str] = None,
    ) -> bool:
        """
        Determine whether the user enrollment can be finalized.

        User enrollment is considered complete when:
        - Primary language is complete
        - Secondary language is complete (if declared)
        """
        required_languages = [primary_language]
        if secondary_language:
            required_languages.append(secondary_language)
        
        return self.is_user_enrollment_complete(
            enrollment_statuses=enrollment_statuses,
            required_languages=required_languages,
        )


# Default singleton (matches your existing service pattern)
enrollment_policy = EnrollmentPolicy()
