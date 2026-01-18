from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Settings
    app_name: str = "Voice Banking API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Primary Model Settings (ECAPA-TDNN)
    model_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    model_savedir: Path = Path("pretrained_models/spkrec")
    
    # Secondary Model Settings (X-Vector) for comparison
    enable_secondary_model: bool = True
    secondary_model_source: str = "speechbrain/spkrec-xvect-voxceleb"
    secondary_model_savedir: Path = Path("pretrained_models/xvect")
    
    # Storage Settings
    embeddings_dir: Path = Path("storage/embeddings")
    pending_dir: Path = Path("storage/pending")
    
    # Multi-sample Enrollment Settings
    min_enrollment_samples: int = 3
    max_enrollment_samples: int = 10
    enrollment_timeout_hours: int = 24  # Auto-expire pending enrollments
    
    # Verification Settings
    similarity_threshold: float = 0.80
    
    # VAD (Voice Activity Detection) Settings
    vad_min_speech_duration_ms: int = 500     # Min speech required (ms)
    vad_min_speech_percentage: float = 0.10   # Min 10% of audio must be speech
    vad_threshold: float = 0.5                # Silero VAD probability threshold
    
    # Audio Quality Settings
    min_audio_rms: float = 0.01               # Min RMS energy (quiet audio rejection)
    
    # Accent / Multi-language Settings
    accent_schema_version: int = 4
    default_language: str = "en-IN"
    supported_languages: list = [
        "en-IN", "hi-IN", "mr-IN", "ta-IN", "te-IN", 
        "kn-IN", "ml-IN", "gu-IN", "pa-IN", "bn-IN"
    ]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
