"""Deep ensemble implementation for uncertainty estimation."""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from .lightning_module import MolecularPropertyPredictor
from ..utils import set_seed

logger = logging.getLogger(__name__)


class DeepEnsemble:
    """Deep ensemble for uncertainty estimation."""

    def __init__(
        self,
        model_config: Dict,
        train_config: Dict,
        ensemble_size: int = 5,
        input_dim: Optional[int] = None,
    ) -> None:
        self.model_config = model_config
        self.train_config = train_config
        self.ensemble_size = ensemble_size
        self.input_dim = input_dim
        
        self.models: List[MolecularPropertyPredictor] = []
        self.trainers: List[L.Trainer] = []
        self.is_fitted = False

    def fit(
        self,
        datamodule: L.LightningDataModule,
        save_dir: Optional[Path] = None,
    ) -> None:
        """Train the ensemble."""
        logger.info(f"Training ensemble of {self.ensemble_size} models...")
        
        # Get input dimension from data if not provided
        if self.input_dim is None:
            datamodule.setup()
            sample_batch = next(iter(datamodule.train_dataloader()))
            self.input_dim = sample_batch["features"].shape[1]
            logger.info(f"Detected input dimension: {self.input_dim}")

        # Calculate target normalization if needed
        target_mean, target_std = None, None
        if self.model_config.get("normalize_targets", True):
            target_mean, target_std = self._calculate_target_stats(datamodule)

        for i in range(self.ensemble_size):
            logger.info(f"Training model {i+1}/{self.ensemble_size}")
            
            # Set different seed for each model
            base_seed = self.train_config.get("seed", 42)
            set_seed(base_seed + i)
            
            # Create model
            model = MolecularPropertyPredictor(
                model_config=self.model_config,
                train_config=self.train_config,
                input_dim=self.input_dim,
            )
            
            # Set target normalization
            if target_mean is not None and target_std is not None:
                model.set_target_normalization(target_mean, target_std)

            # Setup callbacks
            callbacks = []
            
            # Early stopping
            if self.train_config.get("early_stopping_patience"):
                early_stopping = EarlyStopping(
                    monitor=self.train_config.get("early_stopping_monitor", "val_loss"),
                    mode=self.train_config.get("early_stopping_mode", "min"),
                    patience=self.train_config["early_stopping_patience"],
                    verbose=False,
                )
                callbacks.append(early_stopping)

            # Model checkpointing
            if save_dir is not None:
                checkpoint_dir = save_dir / f"model_{i}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                checkpoint_callback = ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    filename="best",
                    monitor=self.train_config.get("monitor", "val_loss"),
                    mode=self.train_config.get("mode", "min"),
                    save_top_k=1,
                    save_last=True,
                )
                callbacks.append(checkpoint_callback)

            # Create trainer
            trainer = L.Trainer(
                max_epochs=self.train_config.get("num_epochs", 100),
                accelerator=self.train_config.get("accelerator", "cpu"),
                devices=self.train_config.get("devices", 1),
                precision=self.train_config.get("precision", 32),
                callbacks=callbacks,
                enable_progress_bar=self.train_config.get("enable_progress_bar", True),
                log_every_n_steps=self.train_config.get("log_every_n_steps", 10),
                deterministic=self.train_config.get("deterministic", True),
                logger=False,  # Disable lightning logging for ensemble members
            )

            # Train model
            trainer.fit(model, datamodule)
            
            # Load best checkpoint if available
            if save_dir is not None and checkpoint_callback.best_model_path:
                model = MolecularPropertyPredictor.load_from_checkpoint(
                    checkpoint_callback.best_model_path,
                    model_config=self.model_config,
                    train_config=self.train_config,
                    input_dim=self.input_dim,
                )
                if target_mean is not None and target_std is not None:
                    model.set_target_normalization(target_mean, target_std)

            self.models.append(model)
            self.trainers.append(trainer)

        self.is_fitted = True
        logger.info("Ensemble training completed")

    def predict(
        self, 
        dataloader: torch.utils.data.DataLoader,
        return_embeddings: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Make predictions with the ensemble."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        all_predictions = []
        all_embeddings = [] if return_embeddings else None
        indices = None

        for i, model in enumerate(self.models):
            model.eval()
            predictions = []
            embeddings = [] if return_embeddings else None
            
            with torch.no_grad():
                for batch in dataloader:
                    if return_embeddings:
                        pred, emb = model.forward_with_embeddings(batch["features"])
                        embeddings.append(emb.cpu().numpy())
                    else:
                        pred = model(batch["features"])
                    
                    # Denormalize if needed
                    if model.normalize_targets:
                        pred = model._denormalize_targets(pred.squeeze())
                    else:
                        pred = pred.squeeze()
                    
                    predictions.append(pred.cpu().numpy())
                    
                    # Store indices from first model only
                    if i == 0 and "index" in batch:
                        if indices is None:
                            indices = []
                        indices.append(batch["index"].cpu().numpy())

            predictions = np.concatenate(predictions)
            all_predictions.append(predictions)
            
            if return_embeddings:
                embeddings = np.concatenate(embeddings)
                all_embeddings.append(embeddings)

        # Stack predictions from all models
        all_predictions = np.stack(all_predictions, axis=0)  # Shape: (ensemble_size, n_samples)
        
        # Calculate ensemble statistics
        mean_predictions = np.mean(all_predictions, axis=0)
        std_predictions = np.std(all_predictions, axis=0)
        
        results = {
            "mean": mean_predictions,
            "std": std_predictions,
            "individual": all_predictions,
        }
        
        if return_embeddings:
            # Average embeddings across ensemble
            all_embeddings = np.stack(all_embeddings, axis=0)
            mean_embeddings = np.mean(all_embeddings, axis=0)
            results["embeddings"] = mean_embeddings
        
        if indices is not None:
            results["indices"] = np.concatenate(indices)

        return results

    def predict_uncertainty(
        self, 
        dataloader: torch.utils.data.DataLoader,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty estimation."""
        results = self.predict(dataloader, return_embeddings=False)
        return results["mean"], results["std"]

    def save(self, save_dir: Path) -> None:
        """Save the ensemble."""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save ensemble configuration
        config = {
            "model_config": self.model_config,
            "train_config": self.train_config,
            "ensemble_size": self.ensemble_size,
            "input_dim": self.input_dim,
        }
        
        import yaml
        with open(save_dir / "ensemble_config.yaml", "w") as f:
            yaml.dump(config, f)

        # Save individual models
        for i, model in enumerate(self.models):
            model_path = save_dir / f"model_{i}.ckpt"
            trainer = self.trainers[i]
            trainer.save_checkpoint(model_path)

        logger.info(f"Saved ensemble to {save_dir}")

    @classmethod
    def load(cls, save_dir: Path) -> "DeepEnsemble":
        """Load a saved ensemble."""
        # Load configuration
        import yaml
        with open(save_dir / "ensemble_config.yaml", "r") as f:
            config = yaml.safe_load(f)

        ensemble = cls(
            model_config=config["model_config"],
            train_config=config["train_config"],
            ensemble_size=config["ensemble_size"],
            input_dim=config["input_dim"],
        )

        # Load individual models
        models = []
        for i in range(config["ensemble_size"]):
            model_path = save_dir / f"model_{i}.ckpt"
            model = MolecularPropertyPredictor.load_from_checkpoint(
                model_path,
                model_config=config["model_config"],
                train_config=config["train_config"],
                input_dim=config["input_dim"],
            )
            models.append(model)

        ensemble.models = models
        ensemble.trainers = [None] * len(models)  # Trainers not needed for inference
        ensemble.is_fitted = True

        logger.info(f"Loaded ensemble from {save_dir}")
        return ensemble

    def _calculate_target_stats(self, datamodule: L.LightningDataModule) -> Tuple[float, float]:
        """Calculate target normalization statistics from training data."""
        datamodule.setup()
        train_dataloader = datamodule.train_dataloader()
        
        all_targets = []
        for batch in train_dataloader:
            all_targets.append(batch["targets"].numpy())
        
        all_targets = np.concatenate(all_targets)
        mean = float(np.mean(all_targets))
        std = float(np.std(all_targets))
        
        logger.info(f"Target statistics: mean={mean:.3f}, std={std:.3f}")
        return mean, std