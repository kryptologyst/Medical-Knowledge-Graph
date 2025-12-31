"""Training script for medical knowledge graph models."""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import (
    MedicalKGDataset, create_synthetic_medical_kg, split_dataset,
    create_data_loaders, create_entity_mapping, create_relation_mapping
)
from src.models.embeddings import create_model
from src.losses.metrics import KnowledgeGraphLoss, evaluate_model, calculate_filtered_metrics
from src.utils.core import (
    set_seed, get_device, setup_logging, load_config, save_config,
    count_parameters, get_model_size_mb, EarlyStopping
)


def train_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Train model for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to run on
        epoch: Current epoch number
        
    Returns:
        Training metrics
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        
        head = batch['head'].to(device)
        relation = batch['relation'].to(device)
        tail = batch['tail'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        predictions = model(head, relation, tail)
        
        # Calculate loss
        if hasattr(criterion, 'use_negative_sampling') and criterion.use_negative_sampling:
            # For margin ranking loss, we need positive and negative scores
            batch_size = head.size(0)
            positive_mask = labels == 1
            negative_mask = labels == 0
            
            if positive_mask.sum() > 0 and negative_mask.sum() > 0:
                positive_scores = predictions[positive_mask]
                negative_scores = predictions[negative_mask]
                loss = criterion(None, None, positive_scores, negative_scores)
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            loss = criterion(predictions, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 100 == 0:
            logging.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / num_batches
    return {'train_loss': avg_loss}


def validate_epoch(
    model: torch.nn.Module,
    valid_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Validate model for one epoch.
    
    Args:
        model: Model to validate
        valid_loader: Validation data loader
        criterion: Loss function
        device: Device to run on
        
    Returns:
        Validation metrics
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in valid_loader:
            head = batch['head'].to(device)
            relation = batch['relation'].to(device)
            tail = batch['tail'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            predictions = model(head, relation, tail)
            
            # Calculate loss
            if hasattr(criterion, 'use_negative_sampling') and criterion.use_negative_sampling:
                batch_size = head.size(0)
                positive_mask = labels == 1
                negative_mask = labels == 0
                
                if positive_mask.sum() > 0 and negative_mask.sum() > 0:
                    positive_scores = predictions[positive_mask]
                    negative_scores = predictions[negative_mask]
                    loss = criterion(None, None, positive_scores, negative_scores)
                else:
                    loss = torch.tensor(0.0, device=device)
            else:
                loss = criterion(predictions, labels)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return {'valid_loss': avg_loss}


def train_model(
    config: DictConfig,
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: Path
) -> Dict[str, Any]:
    """Train the knowledge graph model.
    
    Args:
        config: Training configuration
        model: Model to train
        train_loader: Training data loader
        valid_loader: Validation data loader
        test_loader: Test data loader
        device: Device to run on
        output_dir: Output directory for checkpoints
        
    Returns:
        Training results
    """
    # Setup optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=config.training.patience,
        verbose=True
    )
    
    # Setup loss function
    criterion = KnowledgeGraphLoss(
        margin=config.model.margin,
        use_negative_sampling=config.training.use_negative_sampling,
        negative_weight=config.training.negative_weight
    )
    
    # Setup early stopping
    early_stopping = EarlyStopping(
        patience=config.training.early_stopping_patience,
        min_delta=config.training.min_delta,
        restore_best_weights=True
    )
    
    # Setup tensorboard
    writer = SummaryWriter(output_dir / 'tensorboard')
    
    # Training loop
    best_valid_loss = float('inf')
    train_history = []
    valid_history = []
    
    for epoch in range(config.training.epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        train_history.append(train_metrics)
        
        # Validate
        valid_metrics = validate_epoch(model, valid_loader, criterion, device)
        valid_history.append(valid_metrics)
        
        # Update learning rate
        scheduler.step(valid_metrics['valid_loss'])
        
        # Log metrics
        logging.info(f'Epoch {epoch}: Train Loss: {train_metrics["train_loss"]:.4f}, '
                    f'Valid Loss: {valid_metrics["valid_loss"]:.4f}')
        
        # Tensorboard logging
        writer.add_scalar('Loss/Train', train_metrics['train_loss'], epoch)
        writer.add_scalar('Loss/Valid', valid_metrics['valid_loss'], epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Save best model
        if valid_metrics['valid_loss'] < best_valid_loss:
            best_valid_loss = valid_metrics['valid_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss': valid_metrics['valid_loss'],
                'config': config
            }, output_dir / 'best_model.pt')
        
        # Early stopping
        if early_stopping(valid_metrics['valid_loss'], model):
            logging.info(f'Early stopping at epoch {epoch}')
            break
    
    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    logging.info('Evaluating on test set...')
    test_metrics = evaluate_model(
        model, test_loader, device,
        {}, {}, {}, {}  # These would be the mappings in a real implementation
    )
    
    writer.close()
    
    return {
        'best_valid_loss': best_valid_loss,
        'test_metrics': test_metrics,
        'train_history': train_history,
        'valid_history': valid_history,
        'final_epoch': epoch
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Medical Knowledge Graph Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    setup_logging()
    
    # Load config
    config = load_config(args.config)
    
    # Setup device
    if args.device == 'auto':
        device = get_device()
    else:
        device = torch.device(args.device)
    
    logging.info(f'Using device: {device}')
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    save_config(config, output_dir / 'config.yaml')
    
    # Generate synthetic data if needed
    if not Path(config.data.train_path).exists():
        logging.info('Generating synthetic medical knowledge graph...')
        create_synthetic_medical_kg(
            num_entities=config.data.num_entities,
            num_relations=config.data.num_relations,
            num_triples=config.data.num_triples,
            save_path=Path(config.data.train_path)
        )
    
    # Load data
    from src.data.dataset import load_triples_from_csv
    triples = load_triples_from_csv(Path(config.data.train_path))
    
    # Split data
    train_triples, valid_triples, test_triples = split_dataset(
        triples,
        train_ratio=config.data.train_ratio,
        valid_ratio=config.data.valid_ratio,
        test_ratio=config.data.test_ratio,
        random_seed=args.seed
    )
    
    # Create entity and relation mappings
    all_entities = []
    all_relations = []
    for head, relation, tail in triples:
        all_entities.extend([head, tail])
        all_relations.append(relation)
    
    entity_to_id = create_entity_mapping(all_entities)
    relation_to_id = create_relation_mapping(all_relations)
    
    # Create datasets
    train_dataset = MedicalKGDataset(
        train_triples, entity_to_id, relation_to_id,
        negative_sampling=config.training.use_negative_sampling,
        num_negatives=config.training.num_negatives
    )
    
    valid_dataset = MedicalKGDataset(
        valid_triples, entity_to_id, relation_to_id,
        negative_sampling=False
    )
    
    test_dataset = MedicalKGDataset(
        test_triples, entity_to_id, relation_to_id,
        negative_sampling=False
    )
    
    # Create data loaders
    train_loader, valid_loader, test_loader = create_data_loaders(
        train_dataset, valid_dataset, test_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers
    )
    
    # Create model
    model = create_model(
        config.model.name,
        len(entity_to_id),
        len(relation_to_id),
        config.model
    )
    
    model = model.to(device)
    
    # Log model info
    num_params = count_parameters(model)
    model_size = get_model_size_mb(model)
    logging.info(f'Model: {config.model.name}')
    logging.info(f'Parameters: {num_params:,}')
    logging.info(f'Size: {model_size:.2f} MB')
    
    # Train model
    results = train_model(
        config, model, train_loader, valid_loader, test_loader,
        device, output_dir
    )
    
    # Save results
    import json
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info('Training completed!')
    logging.info(f'Best validation loss: {results["best_valid_loss"]:.4f}')
    logging.info(f'Test metrics: {results["test_metrics"]}')


if __name__ == '__main__':
    main()
