"""Tests for medical knowledge graph system."""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.data.dataset import MedicalKGDataset, create_synthetic_medical_kg
from src.models.embeddings import TransE, RotatE, ComplEx
from src.losses.metrics import MarginRankingLoss, KnowledgeGraphLoss
from src.utils.core import set_seed, get_device, validate_triple


class TestDataUtils:
    """Test data utility functions."""
    
    def test_validate_triple(self):
        """Test triple validation."""
        assert validate_triple("hypertension", "treated_by", "amlodipine") == True
        assert validate_triple("", "treated_by", "amlodipine") == False
        assert validate_triple("hypertension", "", "amlodipine") == False
        assert validate_triple("hypertension", "treated_by", "") == False
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        triples = create_synthetic_medical_kg(
            num_entities=100,
            num_relations=10,
            num_triples=500
        )
        
        assert len(triples) > 0
        assert all(len(triple) == 3 for triple in triples)
        assert all(validate_triple(*triple) for triple in triples)


class TestModels:
    """Test knowledge graph models."""
    
    def setup_method(self):
        """Setup test data."""
        set_seed(42)
        self.num_entities = 100
        self.num_relations = 10
        self.embedding_dim = 50
        self.device = get_device()
    
    def test_transe_model(self):
        """Test TransE model."""
        model = TransE(
            self.num_entities,
            self.num_relations,
            self.embedding_dim
        ).to(self.device)
        
        # Test forward pass
        head = torch.randint(0, self.num_entities, (10,)).to(self.device)
        relation = torch.randint(0, self.num_relations, (10,)).to(self.device)
        tail = torch.randint(0, self.num_entities, (10,)).to(self.device)
        
        scores = model(head, relation, tail)
        assert scores.shape == (10,)
        assert torch.all(scores >= 0)  # Distances should be non-negative
    
    def test_rotate_model(self):
        """Test RotatE model."""
        model = RotatE(
            self.num_entities,
            self.num_relations,
            self.embedding_dim
        ).to(self.device)
        
        # Test forward pass
        head = torch.randint(0, self.num_entities, (10,)).to(self.device)
        relation = torch.randint(0, self.num_relations, (10,)).to(self.device)
        tail = torch.randint(0, self.num_entities, (10,)).to(self.device)
        
        scores = model(head, relation, tail)
        assert scores.shape == (10,)
        assert torch.all(scores >= 0)  # Distances should be non-negative
    
    def test_complex_model(self):
        """Test ComplEx model."""
        model = ComplEx(
            self.num_entities,
            self.num_relations,
            self.embedding_dim
        ).to(self.device)
        
        # Test forward pass
        head = torch.randint(0, self.num_entities, (10,)).to(self.device)
        relation = torch.randint(0, self.num_relations, (10,)).to(self.device)
        tail = torch.randint(0, self.num_entities, (10,)).to(self.device)
        
        scores = model(head, relation, tail)
        assert scores.shape == (10,)


class TestLosses:
    """Test loss functions."""
    
    def setup_method(self):
        """Setup test data."""
        set_seed(42)
        self.device = get_device()
    
    def test_margin_ranking_loss(self):
        """Test margin ranking loss."""
        loss_fn = MarginRankingLoss(margin=1.0)
        
        positive_scores = torch.tensor([0.5, 0.3, 0.8]).to(self.device)
        negative_scores = torch.tensor([0.2, 0.6, 0.1]).to(self.device)
        
        loss = loss_fn(positive_scores, negative_scores)
        assert loss.item() >= 0
    
    def test_knowledge_graph_loss(self):
        """Test knowledge graph loss."""
        loss_fn = KnowledgeGraphLoss(use_negative_sampling=False)
        
        predictions = torch.randn(10).to(self.device)
        labels = torch.randint(0, 2, (10,)).float().to(self.device)
        
        loss = loss_fn(predictions, labels)
        assert loss.item() >= 0


class TestDataset:
    """Test dataset functionality."""
    
    def setup_method(self):
        """Setup test data."""
        set_seed(42)
        self.triples = [
            ("hypertension", "treated_by", "amlodipine"),
            ("diabetes", "treated_by", "metformin"),
            ("metformin", "side_effect", "nausea"),
        ]
        
        self.entity_to_id = {
            "hypertension": 0, "diabetes": 1, "metformin": 2,
            "amlodipine": 3, "nausea": 4
        }
        
        self.relation_to_id = {
            "treated_by": 0, "side_effect": 1
        }
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        dataset = MedicalKGDataset(
            self.triples,
            self.entity_to_id,
            self.relation_to_id,
            negative_sampling=False
        )
        
        assert len(dataset) == len(self.triples)
        
        # Test getting an item
        item = dataset[0]
        assert 'head' in item
        assert 'relation' in item
        assert 'tail' in item
        assert 'label' in item


if __name__ == "__main__":
    pytest.main([__file__])
