"""
Enrollment Prompt Service

Provides sample sentences for users to read during voice enrollment.
Prompts are loaded from a configurable JSON file.
"""
import json
from pathlib import Path
from typing import Optional, List
from functools import lru_cache


@lru_cache(maxsize=1)
def _load_prompts() -> dict:
    """Load prompts from JSON file (cached)."""
    path = Path(__file__).parent.parent / "data" / "enrollment_prompts.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def get_enrollment_prompt(language_code: str, sample_index: int) -> Optional[str]:
    """
    Get the prompt sentence for a specific language and sample number.
    
    Args:
        language_code: BCP-47 language tag (e.g., "hi-IN")
        sample_index: 0-based index of the current sample
        
    Returns:
        Prompt sentence or None if not available
    """
    prompts = _load_prompts()
    sentences = prompts.get(language_code, [])
    if 0 <= sample_index < len(sentences):
        return sentences[sample_index]
    return None


def get_all_prompts(language_code: str) -> List[str]:
    """
    Get all prompt sentences for a language.
    
    Args:
        language_code: BCP-47 language tag (e.g., "hi-IN")
        
    Returns:
        List of prompt sentences (empty list if not available)
    """
    prompts = _load_prompts()
    return prompts.get(language_code, [])
