"""
Embedding manipulation utilities for multi-sample enrollment.

This module provides utilities for combining multiple speaker embeddings
into a single robust speaker profile (centroid).
"""
import torch
from typing import List, Tuple, Optional


def combine_embeddings(embeddings: List[torch.Tensor]) -> torch.Tensor:
    """
    Combine multiple speaker embeddings into a single centroid profile.
    
    Strategy:
    1. Stack all embeddings
    2. Compute mean across samples (centroid)
    3. L2 normalize the result for consistent similarity scoring
    
    Args:
        embeddings: List of tensors, each shape [1, 192]
    
    Returns:
        Normalized centroid embedding, shape [1, 192]
    
    Raises:
        ValueError: If embeddings list is empty or tensors have inconsistent shapes
    """
    if not embeddings:
        raise ValueError("Cannot combine empty list of embeddings")
    
    # Validate shapes
    expected_shape = embeddings[0].shape
    for i, emb in enumerate(embeddings):
        if emb.shape != expected_shape:
            raise ValueError(
                f"Embedding {i} has shape {emb.shape}, expected {expected_shape}"
            )
    
    # Stack: [N, 1, 192] or handle different shapes
    if len(embeddings[0].shape) == 3:
        # Shape is [1, 1, 192] - squeeze first
        stacked = torch.cat([e.squeeze(0) for e in embeddings], dim=0)
    elif len(embeddings[0].shape) == 2:
        # Shape is [1, 192]
        stacked = torch.cat(embeddings, dim=0)
    else:
        raise ValueError(f"Unexpected embedding shape: {embeddings[0].shape}")
    
    # Compute centroid (mean across samples)
    centroid = torch.mean(stacked, dim=0, keepdim=True)
    
    # L2 normalize for consistent cosine similarity
    centroid = torch.nn.functional.normalize(centroid, p=2, dim=1)
    
    return centroid


def validate_embedding(embedding: torch.Tensor) -> bool:
    """
    Validate that an embedding tensor has the expected format.
    
    Args:
        embedding: Tensor to validate
        
    Returns:
        True if valid, False otherwise
    """
    if embedding is None:
        return False
    
    # Check it's a tensor
    if not isinstance(embedding, torch.Tensor):
        return False
    
    # Check shape - should be [1, 192] or [1, 1, 192]
    if len(embedding.shape) not in (2, 3):
        return False
    
    # Check for NaN or Inf
    if torch.isnan(embedding).any() or torch.isinf(embedding).any():
        return False
    
    return True


