"""Monte Carlo Dropout implementation for uncertainty estimation."""

import logging
from typing import Dict, Optional, Tuple
from pathlib import Path
import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from .lightning_module import MolecularPropertyPredictor
from ..utils import set_seed

logger = logging.getLogger(__name__)


class MCDropoutPredictor:
    """Monte Carlo Dropout for uncertainty estimation."""

    def __init__(
        self,
        model_config: Dict,
        train_config: Dict,
        n_samples: int = 100,
        input_dim: Optional[int] = None,
    ) -> None:
        self.model_config = model_config
        self.train_config = train_config
        self.n_samples = n_samples
        self.input_dim = input_dim
        
        self.model: Optional[MolecularPropertyPredictor] = None
        self.trainer: Optional[L.Trainer] = None
        self.is_fitted = False

    def fit(
        self,
        datamodule: L.LightningDataModule,
        save_dir: Optional[Path] = None,
    ) -> None:
        """Train the MC-Dropout model."""
        logger.info("Training MC-Dropout model...")
        
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

        # Set seed
        set_seed(self.train_config.get("seed", 42))
        
        # Create model
        self.model = MolecularPropertyPredictor(
            model_config=self.model_config,
            train_config=self.train_config,
            input_dim=self.input_dim,
        )
        
        # Set target normalization
        if target_mean is not None and target_std is not None:
            self.model.set_target_normalization(target_mean, target_std)

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
        checkpoint_callback = None
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_callback = ModelCheckpoint(
                dirpath=save_dir,
                filename="best",
                monitor=self.train_config.get("monitor", "val_loss"),
                mode=self.train_config.get("mode", "min"),
                save_top_k=1,
                save_last=True,
            )
            callbacks.append(checkpoint_callback)

        # Create trainer
        self.trainer = L.Trainer(
            max_epochs=self.train_config.get("num_epochs", 100),
            accelerator=self.train_config.get("accelerator", "cpu"),
            devices=self.train_config.get("devices", 1),
            precision=self.train_config.get("precision", 32),
            callbacks=callbacks,
            enable_progress_bar=self.train_config.get("enable_progress_bar", True),
            log_every_n_steps=self.train_config.get("log_every_n_steps", 10),
            deterministic=self.train_config.get("deterministic", True),
            logger=False,  # Disable lightning logging
        )

        # Train model
        self.trainer.fit(self.model, datamodule)
        
        # Load best checkpoint if available
        if save_dir is not None and checkpoint_callback and checkpoint_callback.best_model_path:
            self.model = MolecularPropertyPredictor.load_from_checkpoint(
                checkpoint_callback.best_model_path,
                model_config=self.model_config,
                train_config=self.train_config,
                input_dim=self.input_dim,
            )
            if target_mean is not None and target_std is not None:
                self.model.set_target_normalization(target_mean, target_std)

        self.is_fitted = True
        logger.info("MC-Dropout training completed")

    def predict(
        self, 
        dataloader: torch.utils.data.DataLoader,
        return_embeddings: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Make predictions with MC-Dropout uncertainty estimation."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Enable training mode to keep dropout active
        self.model.train()
        
        all_predictions = []
        all_embeddings = [] if return_embeddings else None
        indices = None

        # Perform multiple forward passes with dropout
        for i in range(self.n_samples):
            predictions = []
            embeddings = [] if return_embeddings else None
            
            with torch.no_grad():
                for batch in dataloader:
                    if return_embeddings:
                        pred, emb = self.model.forward_with_embeddings(batch["features"])
                        embeddings.append(emb.cpu().numpy())
                    else:
                        pred = self.model(batch["features"])
                    
                    # Denormalize if needed
                    if self.model.normalize_targets:
                        pred = self.model._denormalize_targets(pred.squeeze())
                    else:
                        pred = pred.squeeze()
                    
                    predictions.append(pred.cpu().numpy())
                    
                    # Store indices from first sample only
                    if i == 0 and "index" in batch:
                        if indices is None:
                            indices = []
                        indices.append(batch["index"].cpu().numpy())

            predictions = np.concatenate(predictions)
            all_predictions.append(predictions)
            
            if return_embeddings:
                embeddings = np.concatenate(embeddings)
                all_embeddings.append(embeddings)

        # Stack predictions from all MC samples
        all_predictions = np.stack(all_predictions, axis=0)  # Shape: (n_samples, n_data)
        
        # Calculate MC-Dropout statistics
        mean_predictions = np.mean(all_predictions, axis=0)
        std_predictions = np.std(all_predictions, axis=0)
        
        results = {
            "mean": mean_predictions,
            "std": std_predictions,
            "individual": all_predictions,
        }
        
        if return_embeddings:
            # Average embeddings across MC samples
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
        """Save the MC-Dropout model."""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model configuration
        config = {
            "model_config": self.model_config,
            "train_config": self.train_config,
            "n_samples": self.n_samples,
            "input_dim": self.input_dim,
        }
        
        import yaml
        with open(save_dir / "mc_dropout_config.yaml", "w") as f:
            yaml.dump(config, f)

        # Save model checkpoint
        model_path = save_dir / "model.ckpt"
        if self.trainer is not None:
            self.trainer.save_checkpoint(model_path)
        else:
            # Fallback: save using torch
            torch.save({
                "state_dict": self.model.state_dict(),
                "hyper_parameters": self.model.hparams,
            }, model_path)

        logger.info(f"Saved MC-Dropout model to {save_dir}")

    @classmethod
    def load(cls, save_dir: Path) -> "MCDropoutPredictor":
        """Load a saved MC-Dropout model."""
        # Load configuration
        import yaml
        with open(save_dir / "mc_dropout_config.yaml", "r") as f:
            config = yaml.safe_load(f)

        predictor = cls(
            model_config=config["model_config"],
            train_config=config["train_config"],
            n_samples=config["n_samples"],
            input_dim=config["input_dim"],
        )

        # Load model
        model_path = save_dir / "model.ckpt"
        predictor.model = MolecularPropertyPredictor.load_from_checkpoint(
            model_path,
            model_config=config["model_config"],
            train_config=config["train_config"],
            input_dim=config["input_dim"],
        )
        predictor.is_fitted = True

        logger.info(f"Loaded MC-Dropout model from {save_dir}")
        return predictor

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