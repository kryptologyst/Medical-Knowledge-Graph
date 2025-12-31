"""Evaluation script for medical knowledge graph models."""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

import torch
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import (
    MedicalKGDataset, load_triples_from_csv, split_dataset,
    create_entity_mapping, create_relation_mapping
)
from src.models.embeddings import create_model
from src.losses.metrics import calculate_filtered_metrics
from src.utils.core import get_device, setup_logging, load_config


def evaluate_model_performance(
    model_path: Path,
    config_path: Path,
    test_data_path: Path,
    output_path: Path
) -> Dict[str, Any]:
    """Evaluate model performance on test data.
    
    Args:
        model_path: Path to trained model checkpoint
        config_path: Path to model configuration
        test_data_path: Path to test data
        output_path: Path to save evaluation results
        
    Returns:
        Evaluation results
    """
    # Setup
    device = get_device()
    logger = setup_logging()
    
    # Load config
    config = load_config(config_path)
    
    # Load test data
    test_triples = load_triples_from_csv(test_data_path)
    
    # Create mappings (in practice, these would be saved during training)
    all_entities = set()
    all_relations = set()
    for head, relation, tail in test_triples:
        all_entities.add(head)
        all_entities.add(tail)
        all_relations.add(relation)
    
    entity_to_id = create_entity_mapping(list(all_entities))
    relation_to_id = create_relation_mapping(list(all_relations))
    
    # Create model
    model = create_model(
        config.model.name,
        len(entity_to_id),
        len(relation_to_id),
        config.model
    )
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Evaluating {config.model.name} model on {len(test_triples)} test triples")
    
    # Calculate filtered metrics
    metrics = calculate_filtered_metrics(
        model, test_triples, entity_to_id, relation_to_id,
        {v: k for k, v in entity_to_id.items()},
        {v: k for k, v in relation_to_id.items()},
        device
    )
    
    # Add additional evaluation metrics
    results = {
        'model_name': config.model.name,
        'test_triples': len(test_triples),
        'num_entities': len(entity_to_id),
        'num_relations': len(relation_to_id),
        'metrics': metrics,
        'config': OmegaConf.to_container(config, resolve=True)
    }
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation completed. Results saved to {output_path}")
    logger.info(f"MRR: {metrics['mrr']:.4f}")
    logger.info(f"Hits@1: {metrics['hits@1']:.4f}")
    logger.info(f"Hits@3: {metrics['hits@3']:.4f}")
    logger.info(f"Hits@10: {metrics['hits@10']:.4f}")
    
    return results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate Medical Knowledge Graph Model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config_path', type=str, required=True, help='Path to config file')
    parser.add_argument('--test_data_path', type=str, required=True, help='Path to test data')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save results')
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_model_performance(
        Path(args.model_path),
        Path(args.config_path),
        Path(args.test_data_path),
        Path(args.output_path)
    )
    
    print("Evaluation Results:")
    print(f"Model: {results['model_name']}")
    print(f"Test Triples: {results['test_triples']}")
    print(f"MRR: {results['metrics']['mrr']:.4f}")
    print(f"Hits@1: {results['metrics']['hits@1']:.4f}")
    print(f"Hits@3: {results['metrics']['hits@3']:.4f}")
    print(f"Hits@10: {results['metrics']['hits@10']:.4f}")


if __name__ == '__main__':
    main()
