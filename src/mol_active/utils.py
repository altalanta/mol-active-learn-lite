"""Utility functions for molecular active learning."""

import logging
import random
from typing import Any, Dict, Optional
import numpy as np
import torch
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Set random seed to {seed}")


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def save_config(config: Dict[str, Any], config_path: Path) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save YAML config file
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, indent=2, default_flow_style=False)
    
    logger.info(f"Saved configuration to {config_path}")


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration dictionary
    """
    merged = {}
    
    for config in configs:
        _deep_update(merged, config)
    
    return merged


def _deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
    """Recursively update nested dictionary.
    
    Args:
        base_dict: Base dictionary to update (modified in-place)
        update_dict: Dictionary with updates
    """
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> None:
    """Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to write logs to
        format_string: Optional custom format string
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    
    handlers = [logging.StreamHandler()]
    
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level_map.get(level.upper(), logging.INFO),
        format=format_string,
        handlers=handlers,
        force=True
    )
    
    logger.info(f"Logging configured at {level} level")


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Get the best available device for PyTorch.
    
    Args:
        prefer_gpu: Whether to prefer GPU if available
        
    Returns:
        PyTorch device
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format time duration in human-readable format.
    
    Args:
        seconds: Time duration in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def create_experiment_dir(
    base_dir: Path,
    experiment_name: Optional[str] = None,
    timestamp: bool = True
) -> Path:
    """Create experiment directory with timestamp.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Optional experiment name
        timestamp: Whether to add timestamp to directory name
        
    Returns:
        Path to created experiment directory
    """
    from datetime import datetime
    
    if experiment_name is None:
        experiment_name = "experiment"
    
    if timestamp:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{experiment_name}_{timestamp_str}"
    else:
        dir_name = experiment_name
    
    experiment_dir = base_dir / dir_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created experiment directory: {experiment_dir}")
    return experiment_dir


def save_results(
    results: Dict[str, Any],
    save_path: Path,
    format: str = "yaml"
) -> None:
    """Save results to file.
    
    Args:
        results: Results dictionary
        save_path: Path to save results
        format: File format ("yaml" or "json")
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "yaml":
        with open(save_path, 'w') as f:
            yaml.dump(results, f, indent=2, default_flow_style=False)
    elif format == "json":
        import json
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Saved results to {save_path}")


def load_results(file_path: Path) -> Dict[str, Any]:
    """Load results from file.
    
    Args:
        file_path: Path to results file
        
    Returns:
        Results dictionary
    """
    suffix = file_path.suffix.lower()
    
    if suffix == ".yaml" or suffix == ".yml":
        with open(file_path, 'r') as f:
            results = yaml.safe_load(f)
    elif suffix == ".json":
        import json
        with open(file_path, 'r') as f:
            results = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
    
    logger.info(f"Loaded results from {file_path}")
    return results


class EarlyStopping:
    """Early stopping utility for training loops."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best_weights: bool = True
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
        if mode == "min":
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
            
    def __call__(self, score: float, model: Optional[torch.nn.Module] = None) -> bool:
        """Check if training should stop.
        
        Args:
            score: Current validation score
            model: Optional model to save best weights
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            if model is not None and self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif self.monitor_op(score - self.min_delta, self.best_score):
            self.best_score = score
            self.counter = 0
            if model is not None and self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            logger.info(f"Early stopping after {self.counter} epochs without improvement")
            if model is not None and self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict({k: v.to(next(model.parameters()).device) 
                                     for k, v in self.best_weights.items()})
                logger.info("Restored best model weights")
            return True
            
        return False


def compute_class_weights(targets: np.ndarray, method: str = "balanced") -> np.ndarray:
    """Compute class weights for imbalanced datasets.
    
    Args:
        targets: Target values
        method: Weighting method ("balanced" or "inverse")
        
    Returns:
        Class weights array
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    unique_classes = np.unique(targets)
    
    if method == "balanced":
        weights = compute_class_weight(
            class_weight="balanced",
            classes=unique_classes,
            y=targets
        )
    elif method == "inverse":
        class_counts = np.bincount(targets.astype(int))
        weights = 1.0 / class_counts[unique_classes]
        weights = weights / np.sum(weights) * len(unique_classes)
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    return weights


def stratified_split(
    data: np.ndarray,
    targets: np.ndarray,
    test_size: float = 0.2,
    random_state: Optional[int] = None
) -> tuple:
    """Perform stratified split for continuous targets.
    
    Args:
        data: Input data
        targets: Target values
        test_size: Fraction of data for test set
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (train_data, test_data, train_targets, test_targets)
    """
    from sklearn.model_selection import train_test_split
    import pandas as pd
    
    # Create quantile bins for stratification
    n_bins = min(10, len(targets) // 20)
    target_bins = pd.qcut(targets, q=n_bins, duplicates='drop')
    
    return train_test_split(
        data, targets,
        test_size=test_size,
        stratify=target_bins,
        random_state=random_state
    )


def moving_average(values: list, window: int) -> list:
    """Compute moving average of values.
    
    Args:
        values: List of values
        window: Window size for moving average
        
    Returns:
        List of moving averages
    """
    if len(values) < window:
        return values
        
    moving_avg = []
    for i in range(len(values)):
        if i < window - 1:
            moving_avg.append(np.mean(values[:i+1]))
        else:
            moving_avg.append(np.mean(values[i-window+1:i+1]))
            
    return moving_avg