"""
Voice Vector Store - Qdrant-based storage for voice embeddings.

Provides methods for storing, searching, and managing voice embeddings
with user-based filtering for enrollment and verification operations.
"""
from uuid import uuid4
from datetime import datetime
from typing import Optional, List, Dict, Any

from qdrant_client.http import models

from app.services.vectordb.qdrant_client import get_qdrant_client
from app.config import get_settings


class VoiceVectorStore:
    """Qdrant-based voice embedding storage."""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = get_qdrant_client()
        self.collection = self.settings.QDRANT_COLLECTION

    def store_embedding(
        self,
        user_id: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a voice embedding in Qdrant.
        
        Args:
            user_id: Unique identifier for the user
            embedding: Voice embedding vector (list of floats)
            metadata: Additional payload data (language, timestamps, etc.)
            
        Returns:
            Point ID of the stored embedding
        """
        point_id = str(uuid4())
        
        payload = {
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            **(metadata or {})
        }
        
        self.client.upsert(
            collection_name=self.collection,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
            ]
        )
        
        return point_id

    def search_by_user(
        self,
        user_id: str,
        query_embedding: List[float],
        top_k: int = 1
    ) -> List[models.ScoredPoint]:
        """
        Search for similar embeddings filtered by user_id.
        
        Args:
            user_id: User to search within
            query_embedding: Query vector
            top_k: Number of results to return
            
        Returns:
            List of scored points with similarity scores
        """
        return self.client.search(
            collection_name=self.collection,
            query_vector=query_embedding,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id)
                    )
                ]
            ),
            limit=top_k
        )

    def get_by_user_id(self, user_id: str) -> List[models.Record]:
        """
        Retrieve all embeddings for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of records for the user
        """
        results, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id)
                    )
                ]
            ),
            limit=100,
            with_vectors=True
        )
        return results

    def delete_by_user_id(self, user_id: str) -> int:
        """
        Delete all embeddings for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Number of points deleted
        """
        # Get points to delete first (to return count)
        existing = self.get_by_user_id(user_id)
        count = len(existing)
        
        if count > 0:
            self.client.delete(
                collection_name=self.collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="user_id",
                                match=models.MatchValue(value=user_id)
                            )
                        ]
                    )
                )
            )
        
        return count

    def user_exists(self, user_id: str) -> bool:
        """
        Check if a user has any stored embeddings.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if user has embeddings stored
        """
        results = self.get_by_user_id(user_id)
        return len(results) > 0

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[models.Filter] = None
    ) -> List[models.ScoredPoint]:
        """
        General search without user filtering.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            filters: Optional Qdrant filter
            
        Returns:
            List of scored points
        """
        return self.client.search(
            collection_name=self.collection,
            query_vector=query_embedding,
            query_filter=filters,
            limit=top_k
        )


# Global singleton
voice_vector_store = VoiceVectorStore()
