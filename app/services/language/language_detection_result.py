"""
Language Detection Result Model

Purpose:
- Carry language detection metadata
- Never contains audio or embeddings
- Used only for routing & policy decisions
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LanguageDetectionResult:
    """
    Result returned by LanguageDetectionService.

    Attributes:
        language: BCP-47 language tag (e.g. 'hi-IN', 'en-IN')
        confidence: Confidence score between 0.0 and 1.0
        model_name: Name of LID model used
    """
    language: str
    confidence: float
    model_name: str

    @property
    def is_confident(self) -> bool:
        """
        Convenience helper for policy decisions.
        """
        return self.confidence >= 0.80
