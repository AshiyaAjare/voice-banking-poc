from typing import Dict, List, Optional, Tuple

from app.models.accent import (
    VerificationStrategy,
    AccentVerificationDecision,
)


class VerificationPolicy:
    """
    Defines decision strategies for accent-aware voice verification.

    This class contains NO side effects.
    It evaluates scores and returns verification decisions.
    """

    def __init__(
        self,
        primary_threshold: float = 0.70,
        secondary_threshold: float = 0.65,
        fusion_weight_primary: float = 0.7,
        fusion_weight_secondary: float = 0.3,
    ):
        self.primary_threshold = primary_threshold
        self.secondary_threshold = secondary_threshold
        self.fusion_weight_primary = fusion_weight_primary
        self.fusion_weight_secondary = fusion_weight_secondary

    # -------------------------------------------------
    # Score fusion
    # -------------------------------------------------

    def fuse_scores(
        self,
        primary_score: float,
        secondary_score: Optional[float] = None,
    ) -> float:
        """
        Fuse ECAPA and X-Vector scores into a final score.
        """
        if secondary_score is None:
            return primary_score

        return (
            self.fusion_weight_primary * primary_score
            + self.fusion_weight_secondary * secondary_score
        )

    # -------------------------------------------------
    # Decision rules
    # -------------------------------------------------

    def is_match(
        self,
        final_score: float,
        strategy: VerificationStrategy,
    ) -> bool:
        """
        Determine whether a final score constitutes a match.
        """
        if strategy == VerificationStrategy.DUAL_MODEL_FUSION:
            return final_score >= self.secondary_threshold

        return final_score >= self.primary_threshold

    # -------------------------------------------------
    # Strategy-based decisions
    # -------------------------------------------------

    def decide_accent_matched(
        self,
        language_code: str,
        primary_score: float,
        secondary_score: Optional[float] = None,
    ) -> AccentVerificationDecision:
        """
        Accent-matched verification decision.
        """
        final_score = self.fuse_scores(primary_score, secondary_score)
        matched = self.is_match(final_score, VerificationStrategy.ACCENT_MATCHED)

        return AccentVerificationDecision(
            matched=matched,
            reason="Accent-matched verification",
            language_used=language_code,
            confidence_level=self._confidence_label(final_score),
        )

    def decide_declared_language_fallback(
        self,
        language_scores: Dict[str, Tuple[float, Optional[float]]],
    ) -> AccentVerificationDecision:
        """
        Try declared languages (primary → secondary → optional).
        """
        for language, (primary, secondary) in language_scores.items():
            final_score = self.fuse_scores(primary, secondary)
            if self.is_match(final_score, VerificationStrategy.DECLARED_LANGUAGE_FALLBACK):
                return AccentVerificationDecision(
                    matched=True,
                    reason="Declared language fallback",
                    language_used=language,
                    confidence_level=self._confidence_label(final_score),
                )

        return AccentVerificationDecision(
            matched=False,
            reason="Declared language fallback failed",
            language_used=None,
            confidence_level=None,
        )

    def decide_best_of_all(
        self,
        language_scores: Dict[str, Tuple[float, Optional[float]]],
    ) -> AccentVerificationDecision:
        """
        Best-of-all strategy across all enrolled language profiles.
        """
        best_language: Optional[str] = None
        best_score: float = -1.0

        for language, (primary, secondary) in language_scores.items():
            final_score = self.fuse_scores(primary, secondary)
            if final_score > best_score:
                best_score = final_score
                best_language = language

        if best_language is None:
            return AccentVerificationDecision(
                matched=False,
                reason="No language profiles available",
                language_used=None,
                confidence_level=None,
            )

        matched = self.is_match(best_score, VerificationStrategy.BEST_OF_ALL)

        return AccentVerificationDecision(
            matched=matched,
            reason="Best-of-all language match",
            language_used=best_language,
            confidence_level=self._confidence_label(best_score),
        )

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------

    def _confidence_label(self, score: float) -> str:
        """
        Convert a numeric score into a human-readable confidence label.
        """
        if score >= 0.85:
            return "very_high"
        if score >= 0.75:
            return "high"
        if score >= 0.65:
            return "medium"
        return "low"


# Default singleton
verification_policy = VerificationPolicy()
