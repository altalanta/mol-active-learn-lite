"""Lightning module for molecular property prediction."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """Multi-layer perceptron with batch normalization and dropout."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Activation function
        if activation == "relu":
            self.activation_fn = nn.ReLU()
        elif activation == "gelu":
            self.activation_fn = nn.GELU()
        elif activation == "swish":
            self.activation_fn = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build network layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(self.activation_fn)
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Final output layer (no activation for regression)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Store penultimate layer dimension for embeddings
        self.embedding_dim = hidden_dims[-1] if hidden_dims else input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)

    def forward_with_embeddings(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both predictions and penultimate layer embeddings."""
        # Forward through all layers except the last
        embeddings = x
        for layer in self.network[:-1]:
            embeddings = layer(embeddings)
        
        # Final prediction
        predictions = self.network[-1](embeddings)
        
        return predictions, embeddings


class MolecularPropertyPredictor(L.LightningModule):
    """Lightning module for molecular property prediction."""

    def __init__(
        self,
        model_config: Dict,
        train_config: Dict,
        input_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.model_config = model_config
        self.train_config = train_config
        
        # Model architecture
        if input_dim is not None:
            model_config["input_dim"] = input_dim
            
        self.model = MLP(
            input_dim=model_config["input_dim"],
            hidden_dims=model_config["hidden_dims"],
            output_dim=model_config["output_dim"],
            dropout_rate=model_config["dropout_rate"],
            use_batch_norm=model_config["use_batch_norm"],
            activation=model_config["activation"],
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Target normalization
        self.normalize_targets = model_config.get("normalize_targets", True)
        self.target_mean: Optional[torch.Tensor] = None
        self.target_std: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

    def forward_with_embeddings(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning predictions and embeddings."""
        return self.model.forward_with_embeddings(x)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        features = batch["features"]
        targets = batch["targets"]
        
        # Normalize targets if needed
        if self.normalize_targets:
            targets = self._normalize_targets(targets)
        
        # Forward pass
        predictions = self(features)
        loss = self.criterion(predictions.squeeze(), targets)
        
        # Logging
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        features = batch["features"]
        targets = batch["targets"]
        
        # Normalize targets if needed
        if self.normalize_targets:
            targets = self._normalize_targets(targets)
        
        # Forward pass
        predictions = self(features)
        loss = self.criterion(predictions.squeeze(), targets)
        
        # Calculate metrics
        with torch.no_grad():
            # Denormalize for metric calculation
            if self.normalize_targets:
                pred_denorm = self._denormalize_targets(predictions.squeeze())
                target_denorm = self._denormalize_targets(targets)
            else:
                pred_denorm = predictions.squeeze()
                target_denorm = targets
            
            mae = F.l1_loss(pred_denorm, target_denorm)
            rmse = torch.sqrt(F.mse_loss(pred_denorm, target_denorm))
        
        # Logging
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mae", mae, on_step=False, on_epoch=True)
        self.log("val_rmse", rmse, on_step=False, on_epoch=True)
        
        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step."""
        features = batch["features"]
        targets = batch["targets"]
        
        # Forward pass
        predictions = self(features)
        
        # Denormalize if needed
        if self.normalize_targets:
            predictions = self._denormalize_targets(predictions.squeeze())
        else:
            predictions = predictions.squeeze()
        
        # Calculate metrics
        mae = F.l1_loss(predictions, targets)
        rmse = torch.sqrt(F.mse_loss(predictions, targets))
        
        # Logging
        self.log("test_mae", mae, on_step=False, on_epoch=True)
        self.log("test_rmse", rmse, on_step=False, on_epoch=True)
        
        return {
            "predictions": predictions,
            "targets": targets,
            "mae": mae,
            "rmse": rmse,
        }

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Prediction step with embeddings."""
        features = batch["features"]
        
        # Forward pass with embeddings
        predictions, embeddings = self.forward_with_embeddings(features)
        
        # Denormalize if needed
        if self.normalize_targets:
            predictions = self._denormalize_targets(predictions.squeeze())
        else:
            predictions = predictions.squeeze()
        
        return {
            "predictions": predictions,
            "embeddings": embeddings,
            "indices": batch.get("index", torch.arange(len(features))),
        }

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and schedulers."""
        optimizer_config = self.train_config["optimizer"]
        
        if optimizer_config["name"] == "adam":
            optimizer = Adam(
                self.parameters(),
                lr=optimizer_config["lr"],
                weight_decay=optimizer_config.get("weight_decay", 0),
                betas=optimizer_config.get("betas", [0.9, 0.999]),
            )
        elif optimizer_config["name"] == "adamw":
            optimizer = AdamW(
                self.parameters(),
                lr=optimizer_config["lr"],
                weight_decay=optimizer_config.get("weight_decay", 0),
                betas=optimizer_config.get("betas", [0.9, 0.999]),
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_config['name']}")

        # Learning rate scheduler
        scheduler_config = self.train_config.get("scheduler")
        if scheduler_config is None:
            return optimizer
        
        if scheduler_config["name"] == "reduce_on_plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=scheduler_config.get("mode", "min"),
                factor=scheduler_config.get("factor", 0.5),
                patience=scheduler_config.get("patience", 10),
                min_lr=scheduler_config.get("min_lr", 1e-6),
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "frequency": 1,
                },
            }
        
        return optimizer

    def _normalize_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Normalize targets using training statistics."""
        if self.target_mean is None or self.target_std is None:
            return targets
        return (targets - self.target_mean) / self.target_std

    def _denormalize_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Denormalize targets using training statistics."""
        if self.target_mean is None or self.target_std is None:
            return targets
        return targets * self.target_std + self.target_mean

    def set_target_normalization(self, mean: float, std: float) -> None:
        """Set target normalization parameters."""
        self.target_mean = torch.tensor(mean, dtype=torch.float32)
        self.target_std = torch.tensor(std, dtype=torch.float32)
        logger.info(f"Set target normalization: mean={mean:.3f}, std={std:.3f}")

    def get_embedding_dim(self) -> int:
        """Get the dimension of penultimate layer embeddings."""
        return self.model.embedding_dim