"""Loss functions and metrics for knowledge graph embedding models."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

logger = logging.getLogger("medical_kg")


class MarginRankingLoss(nn.Module):
    """Margin ranking loss for knowledge graph embedding."""
    
    def __init__(self, margin: float = 1.0):
        """Initialize margin ranking loss.
        
        Args:
            margin: Margin value
        """
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor
    ) -> torch.Tensor:
        """Calculate margin ranking loss.
        
        Args:
            positive_scores: Scores for positive triples
            negative_scores: Scores for negative triples
            
        Returns:
            Loss value
        """
        loss = F.relu(self.margin - positive_scores + negative_scores)
        return loss.mean()


class KnowledgeGraphLoss(nn.Module):
    """Combined loss function for knowledge graph training."""
    
    def __init__(
        self,
        margin: float = 1.0,
        use_negative_sampling: bool = True,
        negative_weight: float = 1.0
    ):
        """Initialize knowledge graph loss.
        
        Args:
            margin: Margin for ranking loss
            use_negative_sampling: Whether to use negative sampling
            negative_weight: Weight for negative samples
        """
        super().__init__()
        self.margin = margin
        self.use_negative_sampling = use_negative_sampling
        self.negative_weight = negative_weight
        
        self.margin_loss = MarginRankingLoss(margin)
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        positive_scores: Optional[torch.Tensor] = None,
        negative_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculate combined loss.
        
        Args:
            predictions: Model predictions
            labels: Ground truth labels
            positive_scores: Scores for positive triples
            negative_scores: Scores for negative triples
            
        Returns:
            Loss value
        """
        if self.use_negative_sampling and positive_scores is not None and negative_scores is not None:
            # Use margin ranking loss
            margin_loss = self.margin_loss(positive_scores, negative_scores)
            return margin_loss
        else:
            # Use binary cross entropy loss
            return self.bce_loss(predictions, labels)


class KnowledgeGraphMetrics:
    """Metrics for knowledge graph evaluation."""
    
    def __init__(self, k_values: List[int] = [1, 3, 10]):
        """Initialize metrics.
        
        Args:
            k_values: K values for Hits@K calculation
        """
        self.k_values = k_values
        self.reset()
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.predictions = []
        self.labels = []
        self.ranks = []
        self.mrr_scores = []
    
    def update(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        ranks: Optional[torch.Tensor] = None
    ) -> None:
        """Update metrics with new predictions.
        
        Args:
            predictions: Model predictions
            labels: Ground truth labels
            ranks: Optional ranks for MRR calculation
        """
        self.predictions.extend(predictions.cpu().numpy())
        self.labels.extend(labels.cpu().numpy())
        
        if ranks is not None:
            self.ranks.extend(ranks.cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics.
        
        Returns:
            Dictionary of metric values
        """
        metrics = {}
        
        # Convert to numpy arrays
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        
        # AUC metrics
        if len(np.unique(labels)) > 1:
            metrics['auc'] = roc_auc_score(labels, predictions)
            metrics['auprc'] = average_precision_score(labels, predictions)
        
        # MRR and Hits@K
        if self.ranks:
            ranks = np.array(self.ranks)
            metrics['mrr'] = np.mean(1.0 / ranks)
            
            for k in self.k_values:
                hits_at_k = np.mean(ranks <= k)
                metrics[f'hits@{k}'] = hits_at_k
        
        return metrics


def calculate_ranks(
    scores: torch.Tensor,
    true_indices: torch.Tensor,
    filter_indices: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Calculate ranks for link prediction evaluation.
    
    Args:
        scores: Predicted scores for all entities
        true_indices: True entity indices
        filter_indices: Indices to filter out (for filtered metrics)
        
    Returns:
        Ranks of true entities
    """
    batch_size = scores.size(0)
    ranks = torch.zeros(batch_size, device=scores.device)
    
    for i in range(batch_size):
        score = scores[i]
        true_idx = true_indices[i].item()
        
        # Filter out known triples if provided
        if filter_indices is not None:
            filter_mask = filter_indices[i]
            score = score.clone()
            score[filter_mask] = float('-inf')
        
        # Calculate rank
        sorted_indices = torch.argsort(score, descending=True)
        rank = (sorted_indices == true_idx).nonzero(as_tuple=True)[0].item() + 1
        ranks[i] = rank
    
    return ranks


def evaluate_model(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    entity_to_id: Dict[str, int],
    relation_to_id: Dict[str, int],
    id_to_entity: Dict[int, str],
    id_to_relation: Dict[int, str],
    filter_triples: Optional[List[Tuple[str, str, str]]] = None
) -> Dict[str, float]:
    """Evaluate model on test data.
    
    Args:
        model: Trained model
        data_loader: Test data loader
        device: Device to run evaluation on
        entity_to_id: Entity to ID mapping
        relation_to_id: Relation to ID mapping
        id_to_entity: ID to entity mapping
        id_to_relation: ID to relation mapping
        filter_triples: Known triples to filter out
        
    Returns:
        Evaluation metrics
    """
    model.eval()
    metrics = KnowledgeGraphMetrics()
    
    # Create filter indices if provided
    filter_indices = None
    if filter_triples:
        filter_indices = create_filter_indices(
            filter_triples, entity_to_id, relation_to_id
        )
    
    with torch.no_grad():
        for batch in data_loader:
            head = batch['head'].to(device)
            relation = batch['relation'].to(device)
            tail = batch['tail'].to(device)
            labels = batch['label'].to(device)
            
            # Get predictions
            predictions = model.predict(head, relation)
            
            # Calculate ranks
            true_indices = tail
            batch_filter_indices = None
            if filter_indices is not None:
                batch_filter_indices = filter_indices[head.cpu().numpy(), relation.cpu().numpy()]
            
            ranks = calculate_ranks(predictions, true_indices, batch_filter_indices)
            
            # Update metrics
            metrics.update(predictions, labels, ranks)
    
    return metrics.compute()


def create_filter_indices(
    filter_triples: List[Tuple[str, str, str]],
    entity_to_id: Dict[str, int],
    relation_to_id: Dict[str, int]
) -> Dict[Tuple[int, int], List[int]]:
    """Create filter indices for filtered evaluation.
    
    Args:
        filter_triples: Known triples to filter out
        entity_to_id: Entity to ID mapping
        relation_to_id: Relation to ID mapping
        
    Returns:
        Dictionary mapping (head_id, relation_id) to list of tail_ids to filter
    """
    filter_indices = {}
    
    for head, relation, tail in filter_triples:
        if head in entity_to_id and relation in relation_to_id and tail in entity_to_id:
            head_id = entity_to_id[head]
            relation_id = relation_to_id[relation]
            tail_id = entity_to_id[tail]
            
            key = (head_id, relation_id)
            if key not in filter_indices:
                filter_indices[key] = []
            filter_indices[key].append(tail_id)
    
    return filter_indices


def calculate_filtered_metrics(
    model: nn.Module,
    test_triples: List[Tuple[str, str, str]],
    entity_to_id: Dict[str, int],
    relation_to_id: Dict[str, int],
    id_to_entity: Dict[int, str],
    id_to_relation: Dict[int, str],
    device: torch.device
) -> Dict[str, float]:
    """Calculate filtered metrics for link prediction.
    
    Args:
        model: Trained model
        test_triples: Test triples
        entity_to_id: Entity to ID mapping
        relation_to_id: Relation to ID mapping
        id_to_entity: ID to entity mapping
        id_to_relation: ID to relation mapping
        device: Device to run evaluation on
        
    Returns:
        Filtered evaluation metrics
    """
    model.eval()
    
    # Create filter indices
    filter_indices = create_filter_indices(test_triples, entity_to_id, relation_to_id)
    
    all_ranks = []
    
    with torch.no_grad():
        for head, relation, tail in test_triples:
            if head not in entity_to_id or relation not in relation_to_id or tail not in entity_to_id:
                continue
            
            head_id = entity_to_id[head]
            relation_id = relation_to_id[relation]
            tail_id = entity_to_id[tail]
            
            # Get predictions for this head-relation pair
            head_tensor = torch.tensor([head_id], device=device)
            relation_tensor = torch.tensor([relation_id], device=device)
            
            predictions = model.predict(head_tensor, relation_tensor)
            
            # Get filter indices for this pair
            key = (head_id, relation_id)
            if key in filter_indices:
                filter_mask = torch.zeros(len(entity_to_id), dtype=torch.bool, device=device)
                filter_mask[filter_indices[key]] = True
            else:
                filter_mask = None
            
            # Calculate rank
            ranks = calculate_ranks(predictions, torch.tensor([tail_id], device=device), filter_mask)
            all_ranks.append(ranks[0].item())
    
    # Calculate metrics
    all_ranks = np.array(all_ranks)
    mrr = np.mean(1.0 / all_ranks)
    
    metrics = {'mrr': mrr}
    for k in [1, 3, 10]:
        hits_at_k = np.mean(all_ranks <= k)
        metrics[f'hits@{k}'] = hits_at_k
    
    return metrics


def calculate_calibration_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 10
) -> Dict[str, float]:
    """Calculate calibration metrics.
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
        num_bins: Number of bins for calibration
        
    Returns:
        Calibration metrics
    """
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # Convert predictions to probabilities
    probabilities = torch.sigmoid(torch.tensor(predictions)).numpy()
    
    # Calculate calibration error
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = probabilities[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    # Calculate Brier score
    brier_score = np.mean((probabilities - labels) ** 2)
    
    return {
        'ece': ece,
        'brier_score': brier_score
    }
