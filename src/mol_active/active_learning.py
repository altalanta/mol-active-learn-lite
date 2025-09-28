"""Active learning with Gaussian Process surrogates and acquisition functions."""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from pathlib import Path

import gpytorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.preprocessing import StandardScaler

from .models.ensemble import DeepEnsemble
from .models.mc_dropout import MCDropoutPredictor

logger = logging.getLogger(__name__)


class GPSurrogate:
    """Gaussian Process surrogate model for active learning."""
    
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.model: Optional[SingleTaskGP] = None
        self.mll: Optional[ExactMarginalLogLikelihood] = None
        self.input_scaler: Optional[StandardScaler] = None
        self.output_scaler: Optional[StandardScaler] = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the GP surrogate model.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples,)
        """
        logger.info(f"Fitting GP surrogate on {len(X)} samples...")
        
        # Standardize inputs and outputs
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
        
        X_scaled = self.input_scaler.fit_transform(X)
        y_scaled = self.output_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Convert to torch tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y_scaled)
        
        # Create GP model
        self.model = SingleTaskGP(
            X_tensor, 
            y_tensor.unsqueeze(-1),
            # Use automatic relevance determination (ARD) kernel
            covar_module=gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    ard_num_dims=X_tensor.shape[-1],
                    lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0)
                ),
                outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15)
            ),
            likelihood=gpytorch.likelihoods.GaussianLikelihood(
                noise_prior=gpytorch.priors.GammaPrior(1.1, 0.05)
            )
        )
        
        # Marginal log likelihood
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        
        # Fit hyperparameters
        self.model.train()
        self.mll.train()
        
        try:
            fit_gpytorch_model(self.mll)
            logger.info("GP hyperparameter optimization completed")
        except Exception as e:
            logger.warning(f"GP hyperparameter optimization failed: {e}")
            logger.warning("Using default hyperparameters")
        
        self.model.eval()
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with the GP model.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Tuple of (mean_predictions, std_predictions)
        """
        if not self.is_fitted:
            raise ValueError("GP model must be fitted before prediction")
            
        # Standardize inputs
        X_scaled = self.input_scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(X_tensor)
            mean = posterior.mean
            variance = posterior.variance
            
        # Convert back to numpy and rescale
        mean_scaled = mean.squeeze().numpy()
        std_scaled = torch.sqrt(variance).squeeze().numpy()
        
        # Rescale outputs
        mean_pred = self.output_scaler.inverse_transform(mean_scaled.reshape(-1, 1)).flatten()
        
        # For standard deviation, we need to scale by the output standard deviation
        output_std = np.sqrt(self.output_scaler.var_[0])
        std_pred = std_scaled * output_std
        
        return mean_pred, std_pred


class AcquisitionFunction:
    """Wrapper for BoTorch acquisition functions."""
    
    def __init__(self, name: str, config: Dict) -> None:
        self.name = name
        self.config = config
        self.gp_model: Optional[SingleTaskGP] = None
        
    def set_gp_model(self, gp_model: SingleTaskGP) -> None:
        """Set the GP model for the acquisition function."""
        self.gp_model = gp_model
        
    def __call__(self, X: np.ndarray, y_best: Optional[float] = None) -> np.ndarray:
        """Evaluate acquisition function.
        
        Args:
            X: Candidate points (n_candidates, n_features)
            y_best: Best observed value (for EI)
            
        Returns:
            Acquisition values (n_candidates,)
        """
        if self.gp_model is None:
            raise ValueError("GP model must be set before calling acquisition function")
            
        X_tensor = torch.FloatTensor(X)
        
        if self.name == "ei":
            if y_best is None:
                raise ValueError("y_best must be provided for Expected Improvement")
            acq_func = ExpectedImprovement(
                model=self.gp_model,
                best_f=torch.tensor(y_best).float()
            )
        elif self.name == "ucb":
            beta = self.config.get("ucb_beta", 2.0)
            acq_func = UpperConfidenceBound(
                model=self.gp_model,
                beta=beta
            )
        else:
            raise ValueError(f"Unknown acquisition function: {self.name}")
            
        # Evaluate acquisition function
        with torch.no_grad():
            acq_values = acq_func(X_tensor.unsqueeze(-2)).squeeze().numpy()
            
        return acq_values


class ActiveLearner:
    """Active learning coordinator."""
    
    def __init__(
        self,
        uncertainty_model: Union[DeepEnsemble, MCDropoutPredictor],
        config: Dict
    ) -> None:
        self.uncertainty_model = uncertainty_model
        self.config = config
        
        # Initialize GP surrogate
        self.gp_surrogate = GPSurrogate(config.get("gp_config", {}))
        
        # Initialize acquisition function
        acq_name = config.get("acquisition_function", "ei")
        acq_config = config.get("acquisition_config", {})
        self.acquisition_func = AcquisitionFunction(acq_name, acq_config)
        
        # Track learning history
        self.history: List[Dict] = []
        
    def select_candidates(
        self,
        pool_features: np.ndarray,
        pool_embeddings: np.ndarray,
        labeled_embeddings: np.ndarray,
        labeled_targets: np.ndarray,
        batch_size: int,
        y_best: Optional[float] = None
    ) -> np.ndarray:
        """Select candidates for labeling using active learning.
        
        Args:
            pool_features: Features for unlabeled pool (n_pool, n_features)
            pool_embeddings: Penultimate layer embeddings for pool (n_pool, embed_dim)
            labeled_embeddings: Embeddings for labeled data (n_labeled, embed_dim)
            labeled_targets: Target values for labeled data (n_labeled,)
            batch_size: Number of candidates to select
            y_best: Best observed target value (for EI acquisition)
            
        Returns:
            Indices of selected candidates in the pool
        """
        logger.info(f"Selecting {batch_size} candidates from pool of {len(pool_features)}")
        
        # Fit GP surrogate on embeddings
        self.gp_surrogate.fit(labeled_embeddings, labeled_targets)
        self.acquisition_func.set_gp_model(self.gp_surrogate.model)
        
        # Get predictions and uncertainties from uncertainty model
        logger.info("Getting uncertainty estimates...")
        uncertainty_results = self._get_uncertainty_estimates(pool_features)
        
        # Evaluate acquisition function on pool embeddings
        logger.info("Evaluating acquisition function...")
        
        # Convert embeddings to standardized format for GP
        pool_embeddings_scaled = self.gp_surrogate.input_scaler.transform(pool_embeddings)
        
        # Calculate acquisition values
        acq_values = self.acquisition_func(pool_embeddings_scaled, y_best)
        
        # Combine with uncertainty estimates (optional weighting)
        alpha = self.config.get("uncertainty_weight", 0.1)
        combined_scores = acq_values + alpha * uncertainty_results["std"]
        
        # Select top candidates
        selected_indices = np.argsort(combined_scores)[-batch_size:]
        
        # Log selection statistics
        logger.info(f"Selected candidates with acquisition values: "
                   f"mean={np.mean(acq_values[selected_indices]):.3f}, "
                   f"max={np.max(acq_values[selected_indices]):.3f}")
        
        # Store in history
        self.history.append({
            "num_labeled": len(labeled_targets),
            "pool_size": len(pool_features),
            "batch_size": batch_size,
            "acquisition_values": acq_values,
            "selected_indices": selected_indices,
            "uncertainty_mean": np.mean(uncertainty_results["std"]),
            "uncertainty_max": np.max(uncertainty_results["std"]),
        })
        
        return selected_indices
        
    def _get_uncertainty_estimates(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """Get uncertainty estimates from the uncertainty model."""
        # Create a simple dataloader for prediction
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        
        feature_tensor = torch.FloatTensor(features)
        dataset = TensorDataset(feature_tensor)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
        
        # Convert dataloader to expected format
        formatted_dataloader = []
        for batch in dataloader:
            formatted_batch = {"features": batch[0]}
            formatted_dataloader.append(formatted_batch)
        
        # Get predictions
        results = self.uncertainty_model.predict(formatted_dataloader, return_embeddings=True)
        
        return results
        
    def get_history(self) -> List[Dict]:
        """Get the active learning history."""
        return self.history.copy()
        
    def save_history(self, path: Path) -> None:
        """Save active learning history."""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        history_serializable = []
        for entry in self.history:
            entry_copy = entry.copy()
            for key, value in entry_copy.items():
                if isinstance(value, np.ndarray):
                    entry_copy[key] = value.tolist()
            history_serializable.append(entry_copy)
        
        with open(path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
            
        logger.info(f"Saved active learning history to {path}")


def random_selection(pool_size: int, batch_size: int, seed: Optional[int] = None) -> np.ndarray:
    """Random candidate selection baseline.
    
    Args:
        pool_size: Size of the candidate pool
        batch_size: Number of candidates to select
        seed: Random seed for reproducibility
        
    Returns:
        Randomly selected indices
    """
    if seed is not None:
        np.random.seed(seed)
    
    return np.random.choice(pool_size, size=batch_size, replace=False)


def uncertainty_sampling(
    uncertainty_estimates: np.ndarray,
    batch_size: int
) -> np.ndarray:
    """Uncertainty sampling baseline (select highest uncertainty).
    
    Args:
        uncertainty_estimates: Uncertainty estimates for candidates
        batch_size: Number of candidates to select
        
    Returns:
        Indices of candidates with highest uncertainty
    """
    return np.argsort(uncertainty_estimates)[-batch_size:]


def diversity_sampling(
    embeddings: np.ndarray,
    batch_size: int,
    metric: str = "euclidean"
) -> np.ndarray:
    """Diversity-based sampling using k-means clustering.
    
    Args:
        embeddings: Feature embeddings for candidates
        batch_size: Number of candidates to select
        metric: Distance metric for clustering
        
    Returns:
        Indices of diverse candidates
    """
    from sklearn.cluster import KMeans
    
    # Use k-means to find diverse representatives
    kmeans = KMeans(n_clusters=batch_size, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Select point closest to each cluster center
    selected_indices = []
    for i in range(batch_size):
        cluster_mask = (cluster_labels == i)
        cluster_points = embeddings[cluster_mask]
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_points) > 0:
            # Find point closest to cluster center
            center = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(cluster_points - center, axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]
            selected_indices.append(closest_idx)
    
    return np.array(selected_indices)