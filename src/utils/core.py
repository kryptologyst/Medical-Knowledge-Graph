"""Core utilities for medical knowledge graph system."""

import random
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("medical_kg")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        OmegaConf configuration object
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, save_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object
        save_path: Path to save configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, save_path)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in megabytes.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        restore_best_weights: bool = True
    ):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score: float, model: nn.Module) -> bool:
        """Check if training should stop.
        
        Args:
            score: Current validation score
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        
        return False
    
    def save_checkpoint(self, model: nn.Module) -> None:
        """Save model checkpoint.
        
        Args:
            model: Model to save
        """
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()


def deidentify_text(text: str) -> str:
    """Basic text de-identification for demo purposes.
    
    Args:
        text: Input text
        
    Returns:
        De-identified text
    """
    # Simple regex-based de-identification
    import re
    
    # Replace common PHI patterns
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)  # SSN
    text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', text)  # Phone
    text = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]', text)  # Names
    text = re.sub(r'\b\d{1,2}/\d{1,2}/\d{4}\b', '[DATE]', text)  # Dates
    
    return text


def validate_triple(head: str, relation: str, tail: str) -> bool:
    """Validate a knowledge graph triple.
    
    Args:
        head: Head entity
        relation: Relation type
        tail: Tail entity
        
    Returns:
        True if triple is valid
    """
    if not all([head, relation, tail]):
        return False
    
    if len(head.strip()) == 0 or len(tail.strip()) == 0:
        return False
    
    return True


def normalize_entity(entity: str) -> str:
    """Normalize entity names for consistency.
    
    Args:
        entity: Entity name
        
    Returns:
        Normalized entity name
    """
    return entity.strip().lower().replace(" ", "_")


def create_entity_mapping(entities: List[str]) -> Dict[str, int]:
    """Create entity to ID mapping.
    
    Args:
        entities: List of entity names
        
    Returns:
        Dictionary mapping entities to IDs
    """
    unique_entities = sorted(list(set(entities)))
    return {entity: idx for idx, entity in enumerate(unique_entities)}


def create_relation_mapping(relations: List[str]) -> Dict[str, int]:
    """Create relation to ID mapping.
    
    Args:
        relations: List of relation types
        
    Returns:
        Dictionary mapping relations to IDs
    """
    unique_relations = sorted(list(set(relations)))
    return {relation: idx for idx, relation in enumerate(unique_relations)}
