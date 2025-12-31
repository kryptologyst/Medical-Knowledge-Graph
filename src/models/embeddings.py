"""Knowledge graph embedding models for medical reasoning."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.core import get_device

logger = logging.getLogger("medical_kg")


class TransE(nn.Module):
    """TransE (Translating Embeddings) model for knowledge graph completion."""
    
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,
        margin: float = 1.0,
        p_norm: int = 2
    ):
        """Initialize TransE model.
        
        Args:
            num_entities: Number of entities
            num_relations: Number of relations
            embedding_dim: Embedding dimension
            margin: Margin for margin ranking loss
            p_norm: P-norm for distance calculation
        """
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.p_norm = p_norm
        
        # Entity embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        # Relation embeddings
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _init_embeddings(self) -> None:
        """Initialize embeddings with Xavier uniform."""
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        
        # Normalize entity embeddings
        with torch.no_grad():
            self.entity_embeddings.weight.data = F.normalize(
                self.entity_embeddings.weight.data, p=2, dim=1
            )
    
    def forward(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            head: Head entity indices
            relation: Relation indices
            tail: Tail entity indices
            
        Returns:
            Distance scores
        """
        head_emb = self.entity_embeddings(head)
        relation_emb = self.relation_embeddings(relation)
        tail_emb = self.entity_embeddings(tail)
        
        # TransE: h + r ≈ t
        score = head_emb + relation_emb - tail_emb
        
        # Calculate distance
        distance = torch.norm(score, p=self.p_norm, dim=1)
        
        return distance
    
    def predict(self, head: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        """Predict tail entities for given head and relation.
        
        Args:
            head: Head entity indices
            relation: Relation indices
            
        Returns:
            Predicted tail entity scores
        """
        head_emb = self.entity_embeddings(head)
        relation_emb = self.relation_embeddings(relation)
        
        # Calculate h + r
        predicted_emb = head_emb + relation_emb
        
        # Calculate distances to all entities
        all_entity_emb = self.entity_embeddings.weight
        distances = torch.cdist(predicted_emb, all_entity_emb, p=self.p_norm)
        
        return -distances  # Negative distance as score


class RotatE(nn.Module):
    """RotatE (Rotational Embeddings) model for knowledge graph completion."""
    
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,
        margin: float = 1.0,
        gamma: float = 1.0
    ):
        """Initialize RotatE model.
        
        Args:
            num_entities: Number of entities
            num_relations: Number of relations
            embedding_dim: Embedding dimension (must be even)
            margin: Margin for margin ranking loss
            gamma: Scaling factor
        """
        super().__init__()
        assert embedding_dim % 2 == 0, "Embedding dimension must be even for RotatE"
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.gamma = gamma
        
        # Entity embeddings (complex)
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        # Relation embeddings (complex, but only real part for rotation)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim // 2)
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _init_embeddings(self) -> None:
        """Initialize embeddings."""
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        
        # Normalize entity embeddings
        with torch.no_grad():
            self.entity_embeddings.weight.data = F.normalize(
                self.entity_embeddings.weight.data, p=2, dim=1
            )
    
    def forward(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            head: Head entity indices
            relation: Relation indices
            tail: Tail entity indices
            
        Returns:
            Distance scores
        """
        head_emb = self.entity_embeddings(head)
        relation_emb = self.relation_embeddings(relation)
        tail_emb = self.entity_embeddings(tail)
        
        # Convert to complex
        head_real, head_imag = head_emb.chunk(2, dim=1)
        tail_real, tail_imag = tail_emb.chunk(2, dim=1)
        
        # Rotate head embedding
        relation_angle = relation_emb
        cos_r = torch.cos(relation_angle)
        sin_r = torch.sin(relation_angle)
        
        rotated_head_real = head_real * cos_r - head_imag * sin_r
        rotated_head_imag = head_real * sin_r + head_imag * cos_r
        
        # Calculate distance
        distance_real = rotated_head_real - tail_real
        distance_imag = rotated_head_imag - tail_imag
        
        distance = torch.sqrt(distance_real**2 + distance_imag**2 + 1e-8)
        distance = distance.sum(dim=1)
        
        return distance
    
    def predict(self, head: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        """Predict tail entities for given head and relation.
        
        Args:
            head: Head entity indices
            relation: Relation indices
            
        Returns:
            Predicted tail entity scores
        """
        head_emb = self.entity_embeddings(head)
        relation_emb = self.relation_embeddings(relation)
        
        # Convert to complex
        head_real, head_imag = head_emb.chunk(2, dim=1)
        
        # Rotate head embedding
        relation_angle = relation_emb
        cos_r = torch.cos(relation_angle)
        sin_r = torch.sin(relation_angle)
        
        rotated_head_real = head_real * cos_r - head_imag * sin_r
        rotated_head_imag = head_real * sin_r + head_imag * cos_r
        
        # Calculate distances to all entities
        all_entity_emb = self.entity_embeddings.weight
        all_real, all_imag = all_entity_emb.chunk(2, dim=1)
        
        # Broadcast for efficient computation
        rotated_head_real = rotated_head_real.unsqueeze(1)  # [batch, 1, dim]
        rotated_head_imag = rotated_head_imag.unsqueeze(1)  # [batch, 1, dim]
        
        distance_real = rotated_head_real - all_real.unsqueeze(0)  # [batch, num_entities, dim]
        distance_imag = rotated_head_imag - all_imag.unsqueeze(0)  # [batch, num_entities, dim]
        
        distances = torch.sqrt(distance_real**2 + distance_imag**2 + 1e-8)
        distances = distances.sum(dim=2)  # [batch, num_entities]
        
        return -distances  # Negative distance as score


class ComplEx(nn.Module):
    """ComplEx (Complex Embeddings) model for knowledge graph completion."""
    
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,
        margin: float = 1.0
    ):
        """Initialize ComplEx model.
        
        Args:
            num_entities: Number of entities
            num_relations: Number of relations
            embedding_dim: Embedding dimension
            margin: Margin for margin ranking loss
        """
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        
        # Entity embeddings (complex)
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        # Relation embeddings (complex)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _init_embeddings(self) -> None:
        """Initialize embeddings."""
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
    
    def forward(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            head: Head entity indices
            relation: Relation indices
            tail: Tail entity indices
            
        Returns:
            Distance scores
        """
        head_emb = self.entity_embeddings(head)
        relation_emb = self.relation_embeddings(relation)
        tail_emb = self.entity_embeddings(tail)
        
        # Convert to complex
        head_real, head_imag = head_emb.chunk(2, dim=1)
        relation_real, relation_imag = relation_emb.chunk(2, dim=1)
        tail_real, tail_imag = tail_emb.chunk(2, dim=1)
        
        # ComplEx scoring function: Re(<h, r, conj(t)>)
        score_real = head_real * relation_real * tail_real + head_imag * relation_imag * tail_real + \
                    head_real * relation_imag * tail_imag - head_imag * relation_real * tail_imag
        
        score = score_real.sum(dim=1)
        
        return -score  # Negative score as distance
    
    def predict(self, head: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        """Predict tail entities for given head and relation.
        
        Args:
            head: Head entity indices
            relation: Relation indices
            
        Returns:
            Predicted tail entity scores
        """
        head_emb = self.entity_embeddings(head)
        relation_emb = self.relation_embeddings(relation)
        
        # Convert to complex
        head_real, head_imag = head_emb.chunk(2, dim=1)
        relation_real, relation_imag = relation_emb.chunk(2, dim=1)
        
        # Calculate scores for all entities
        all_entity_emb = self.entity_embeddings.weight
        all_real, all_imag = all_entity_emb.chunk(2, dim=1)
        
        # Broadcast for efficient computation
        head_real = head_real.unsqueeze(1)  # [batch, 1, dim]
        head_imag = head_imag.unsqueeze(1)  # [batch, 1, dim]
        relation_real = relation_real.unsqueeze(1)  # [batch, 1, dim]
        relation_imag = relation_imag.unsqueeze(1)  # [batch, 1, dim]
        
        # ComplEx scoring
        score_real = head_real * relation_real * all_real.unsqueeze(0) + \
                    head_imag * relation_imag * all_real.unsqueeze(0) + \
                    head_real * relation_imag * all_imag.unsqueeze(0) - \
                    head_imag * relation_real * all_imag.unsqueeze(0)
        
        scores = score_real.sum(dim=2)  # [batch, num_entities]
        
        return scores


class KnowledgeGraphModel(nn.Module):
    """Wrapper class for knowledge graph embedding models."""
    
    def __init__(
        self,
        model_name: str,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,
        margin: float = 1.0,
        **kwargs
    ):
        """Initialize knowledge graph model.
        
        Args:
            model_name: Name of the model ('TransE', 'RotatE', 'ComplEx')
            num_entities: Number of entities
            num_relations: Number of relations
            embedding_dim: Embedding dimension
            margin: Margin for margin ranking loss
            **kwargs: Additional model-specific parameters
        """
        super().__init__()
        self.model_name = model_name
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        
        # Initialize the specific model
        if model_name.lower() == 'transe':
            self.model = TransE(
                num_entities, num_relations, embedding_dim, margin, **kwargs
            )
        elif model_name.lower() == 'rotate':
            self.model = RotatE(
                num_entities, num_relations, embedding_dim, margin, **kwargs
            )
        elif model_name.lower() == 'complex':
            self.model = ComplEx(
                num_entities, num_relations, embedding_dim, margin, **kwargs
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def forward(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(head, relation, tail)
    
    def predict(self, head: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        """Predict tail entities."""
        return self.model.predict(head, relation)
    
    def get_entity_embeddings(self) -> torch.Tensor:
        """Get entity embeddings."""
        return self.model.entity_embeddings.weight
    
    def get_relation_embeddings(self) -> torch.Tensor:
        """Get relation embeddings."""
        return self.model.relation_embeddings.weight


def create_model(
    model_name: str,
    num_entities: int,
    num_relations: int,
    config: Dict
) -> KnowledgeGraphModel:
    """Create a knowledge graph model.
    
    Args:
        model_name: Name of the model
        num_entities: Number of entities
        num_relations: Number of relations
        config: Model configuration
        
    Returns:
        Initialized model
    """
    return KnowledgeGraphModel(
        model_name=model_name,
        num_entities=num_entities,
        num_relations=num_relations,
        **config
    )
