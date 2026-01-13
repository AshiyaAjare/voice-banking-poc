from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Settings
    app_name: str = "Voice Banking API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Model Settings
    model_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    model_savedir: Path = Path("pretrained_models/spkrec")
    
    # Storage Settings
    embeddings_dir: Path = Path("storage/embeddings")
    
    # Verification Settings
    similarity_threshold: float = 0.25
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
