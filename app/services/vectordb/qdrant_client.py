from qdrant_client import QdrantClient
from app.config import get_settings

def get_qdrant_client():
    settings = get_settings()
    return QdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT
    )
