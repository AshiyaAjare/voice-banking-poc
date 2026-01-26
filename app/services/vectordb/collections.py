from qdrant_client.http import models
from app.services.vectordb.qdrant_client import get_qdrant_client
from app.config import get_settings

def ensure_voice_collection(vector_dim: int = None):
    """Ensure the voice embeddings collection exists with proper configuration."""
    settings = get_settings()
    client = get_qdrant_client()
    
    vector_dim = vector_dim or settings.QDRANT_VECTOR_DIM

    collections = client.get_collections().collections
    if settings.QDRANT_COLLECTION in [c.name for c in collections]:
        return

    client.create_collection(
        collection_name=settings.QDRANT_COLLECTION,
        vectors_config=models.VectorParams(
            size=vector_dim,
            distance=models.Distance.COSINE
        )
    )
    
    # Create payload index for efficient user_id filtering
    client.create_payload_index(
        collection_name=settings.QDRANT_COLLECTION,
        field_name="user_id",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
