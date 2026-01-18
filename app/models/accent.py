from datetime import datetime
from typing import List, Optional, Dict
from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------
# Enums
# ---------------------------------------------------------

class LanguageRole(str, Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    OPTIONAL = "optional"


class VerificationStrategy(str, Enum):
    ACCENT_MATCHED = "accent_matched"
    DECLARED_LANGUAGE_FALLBACK = "declared_language_fallback"
    BEST_OF_ALL = "best_of_all"
    DUAL_MODEL_FUSION = "dual_model_fusion"


# ---------------------------------------------------------
# Core Accent / Language Models
# ---------------------------------------------------------

class AccentProfile(BaseModel):
    """
    Represents a single language/accent-specific voice profile
    for a user.
    """
    user_id: str
    language_code: str = Field(
        ...,
        description="BCP-47 language tag (e.g., en-IN, hi-IN, mr-IN)"
    )
    role: LanguageRole

    sample_count: int = 0
    embedding_version: int

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    # Optional analytics / future use
    confidence_stats: Optional[Dict[str, float]] = None


# ---------------------------------------------------------
# Enrollment Models
# ---------------------------------------------------------

class LanguageEnrollmentRequest(BaseModel):
    """
    Captures user-declared language preferences during enrollment.
    User intent is authoritative.
    """
    primary_language: str
    secondary_language: str
    optional_languages: Optional[List[str]] = Field(
        default=None,
        max_items=1,
        description="Optional additional languages (max 1 for now)"
    )


class AccentEnrollmentStatus(BaseModel):
    """
    Enrollment progress for a specific language.
    """
    language_code: str
    samples_collected: int
    samples_required: int

    @property
    def is_complete(self) -> bool:
        return self.samples_collected >= self.samples_required


class UserAccentEnrollmentSummary(BaseModel):
    """
    Overall enrollment status of a user across languages.
    """
    user_id: str
    primary_language: str
    secondary_language: str

    enrolled_languages: List[AccentEnrollmentStatus]

    @property
    def is_fully_enrolled(self) -> bool:
        return all(lang.is_complete for lang in self.enrolled_languages)


# ---------------------------------------------------------
# Accent Detection (Advisory)
# ---------------------------------------------------------

class AccentDetectionResult(BaseModel):
    """
    Result of accent/language detection.
    Advisory only — never overrides user intent.
    """
    detected_language: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_name: str

    raw_scores: Optional[Dict[str, float]] = None

    @property
    def is_confident(self) -> bool:
        return self.confidence >= 0.75


# ---------------------------------------------------------
# Verification Models
# ---------------------------------------------------------

class AccentVerificationAttempt(BaseModel):
    """
    Represents a single verification attempt.
    Useful for auditing, debugging, and analytics.
    """
    user_id: str

    detected_language: Optional[str]
    matched_language: Optional[str]

    strategy_used: VerificationStrategy

    primary_score: float
    secondary_score: Optional[float] = None
    final_score: float

    matched: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AccentVerificationDecision(BaseModel):
    """
    Final verification decision returned to the API layer.
    """
    matched: bool
    reason: str
    language_used: Optional[str]
    confidence_level: Optional[str] = None
