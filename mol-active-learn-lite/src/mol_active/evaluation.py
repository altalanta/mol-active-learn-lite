"""Uncertainty evaluation metrics and calibration analysis."""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)


def negative_log_likelihood(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_std: np.ndarray
) -> float:
    """Calculate negative log likelihood for Gaussian predictions.
    
    Args:
        y_true: True target values
        y_pred: Predicted mean values
        y_std: Predicted standard deviations
        
    Returns:
        Negative log likelihood
    """
    # Avoid numerical issues with very small standard deviations
    y_std = np.maximum(y_std, 1e-6)
    
    # Gaussian NLL: 0.5 * log(2π) + log(σ) + (y - μ)² / (2σ²)
    nll = 0.5 * np.log(2 * np.pi) + np.log(y_std) + 0.5 * ((y_true - y_pred) / y_std) ** 2
    return float(np.mean(nll))


def calibration_error(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_std: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, float]:
    """Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    
    Args:
        y_true: True target values
        y_pred: Predicted mean values
        y_std: Predicted standard deviations
        n_bins: Number of confidence bins
        
    Returns:
        Tuple of (ECE, MCE)
    """
    # Calculate Z-scores (how many standard deviations away from prediction)
    z_scores = np.abs(y_true - y_pred) / np.maximum(y_std, 1e-6)
    
    # Convert to confidence levels (probability that true value is within prediction interval)
    # For Gaussian: P(|Z| < z) = 2 * Φ(z) - 1
    from scipy.stats import norm
    confidence_levels = 2 * norm.cdf(z_scores) - 1
    
    # Create bins for confidence levels
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    ece = 0.0
    mce = 0.0
    
    for i in range(n_bins):
        # Find samples in this confidence bin
        in_bin = (confidence_levels >= bin_boundaries[i]) & (confidence_levels < bin_boundaries[i + 1])
        
        if np.sum(in_bin) > 0:
            # Empirical accuracy: fraction of predictions that are actually correct
            # (i.e., true value is within the predicted confidence interval)
            bin_accuracy = np.mean(confidence_levels[in_bin])
            
            # Expected confidence for this bin
            bin_confidence = bin_centers[i]
            
            # Calibration error for this bin
            bin_error = abs(bin_accuracy - bin_confidence)
            
            # Weight by number of samples in bin
            bin_weight = np.sum(in_bin) / len(confidence_levels)
            ece += bin_weight * bin_error
            
            # Track maximum calibration error
            mce = max(mce, bin_error)
    
    return float(ece), float(mce)


def reliability_diagram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[Path] = None
) -> Dict[str, np.ndarray]:
    """Generate reliability diagram data and optionally plot it.
    
    Args:
        y_true: True target values
        y_pred: Predicted mean values
        y_std: Predicted standard deviations
        n_bins: Number of confidence bins
        save_path: Optional path to save the plot
        
    Returns:
        Dictionary with reliability diagram data
    """
    # Calculate confidence levels
    z_scores = np.abs(y_true - y_pred) / np.maximum(y_std, 1e-6)
    from scipy.stats import norm
    confidence_levels = 2 * norm.cdf(z_scores) - 1
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(n_bins):
        in_bin = (confidence_levels >= bin_boundaries[i]) & (confidence_levels < bin_boundaries[i + 1])
        
        if np.sum(in_bin) > 0:
            bin_accuracy = np.mean(confidence_levels[in_bin])
            bin_confidence = bin_centers[i]
            bin_count = np.sum(in_bin)
            
            bin_accuracies.append(bin_accuracy)
            bin_confidences.append(bin_confidence)
            bin_counts.append(bin_count)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(bin_centers[i])
            bin_counts.append(0)
    
    bin_accuracies = np.array(bin_accuracies)
    bin_confidences = np.array(bin_confidences)
    bin_counts = np.array(bin_counts)
    
    # Create plot if requested
    if save_path is not None:
        plt.figure(figsize=(8, 6))
        
        # Plot reliability curve
        plt.plot(bin_confidences, bin_accuracies, 'o-', label='Model', linewidth=2, markersize=8)
        
        # Plot perfect calibration line
        plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
        
        # Add bin count information as bar chart
        plt.bar(bin_confidences, bin_counts / np.max(bin_counts) * 0.1, 
                alpha=0.3, width=0.08, label='Frequency')
        
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title('Reliability Diagram')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved reliability diagram to {save_path}")
    
    return {
        "bin_confidences": bin_confidences,
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts,
        "confidence_levels": confidence_levels,
    }


def uncertainty_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """Calculate comprehensive uncertainty evaluation metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted mean values  
        y_std: Predicted standard deviations
        n_bins: Number of bins for calibration metrics
        
    Returns:
        Dictionary of uncertainty metrics
    """
    # Basic regression metrics
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = float(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    
    # Uncertainty-specific metrics
    nll = negative_log_likelihood(y_true, y_pred, y_std)
    ece, mce = calibration_error(y_true, y_pred, y_std, n_bins)
    
    # Prediction interval coverage
    # Calculate coverage for different confidence levels
    z_scores = np.abs(y_true - y_pred) / np.maximum(y_std, 1e-6)
    coverage_68 = float(np.mean(z_scores <= 1.0))  # ~68% for Gaussian
    coverage_95 = float(np.mean(z_scores <= 1.96))  # ~95% for Gaussian
    coverage_99 = float(np.mean(z_scores <= 2.58))  # ~99% for Gaussian
    
    # Uncertainty quality metrics
    # Spearman correlation between prediction errors and uncertainties
    abs_errors = np.abs(y_true - y_pred)
    from scipy.stats import spearmanr
    error_uncertainty_corr = float(spearmanr(abs_errors, y_std)[0])
    
    # Average uncertainty
    mean_uncertainty = float(np.mean(y_std))
    std_uncertainty = float(np.std(y_std))
    
    return {
        # Regression metrics
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        
        # Uncertainty metrics
        "nll": nll,
        "ece": ece,
        "mce": mce,
        
        # Coverage metrics
        "coverage_68": coverage_68,
        "coverage_95": coverage_95,
        "coverage_99": coverage_99,
        
        # Uncertainty quality
        "error_uncertainty_correlation": error_uncertainty_corr,
        "mean_uncertainty": mean_uncertainty,
        "std_uncertainty": std_uncertainty,
    }


def plot_predictions_vs_uncertainty(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    save_path: Optional[Path] = None,
    max_points: int = 1000
) -> None:
    """Plot predictions vs true values with uncertainty bars.
    
    Args:
        y_true: True target values
        y_pred: Predicted mean values
        y_std: Predicted standard deviations
        save_path: Optional path to save the plot
        max_points: Maximum number of points to plot (for performance)
    """
    # Subsample if too many points
    if len(y_true) > max_points:
        indices = np.random.choice(len(y_true), max_points, replace=False)
        y_true = y_true[indices]
        y_pred = y_pred[indices]
        y_std = y_std[indices]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Predictions vs True with error bars
    ax1.errorbar(y_true, y_pred, yerr=y_std, fmt='o', alpha=0.6, capsize=2)
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title('Predictions vs True Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Prediction errors vs uncertainties
    abs_errors = np.abs(y_true - y_pred)
    ax2.scatter(y_std, abs_errors, alpha=0.6)
    
    # Add correlation line
    from scipy.stats import pearsonr
    corr, _ = pearsonr(y_std, abs_errors)
    ax2.set_xlabel('Predicted Uncertainty (σ)')
    ax2.set_ylabel('Absolute Error |y_true - y_pred|')
    ax2.set_title(f'Uncertainty vs Error (r={corr:.3f})')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved prediction plot to {save_path}")
    else:
        plt.show()


def compare_uncertainty_methods(
    results: Dict[str, Dict[str, np.ndarray]],
    y_true: np.ndarray,
    save_dir: Optional[Path] = None
) -> Dict[str, Dict[str, float]]:
    """Compare multiple uncertainty estimation methods.
    
    Args:
        results: Dictionary mapping method names to prediction results
                Each result should have 'mean' and 'std' keys
        y_true: True target values
        save_dir: Optional directory to save comparison plots
        
    Returns:
        Dictionary of metrics for each method
    """
    method_metrics = {}
    
    for method_name, result in results.items():
        y_pred = result["mean"]
        y_std = result["std"]
        
        # Calculate metrics
        metrics = uncertainty_metrics(y_true, y_pred, y_std)
        method_metrics[method_name] = metrics
        
        logger.info(f"{method_name} metrics: "
                   f"RMSE={metrics['rmse']:.3f}, "
                   f"NLL={metrics['nll']:.3f}, "
                   f"ECE={metrics['ece']:.3f}")
        
        # Save individual plots if requested
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Reliability diagram
            reliability_diagram(
                y_true, y_pred, y_std,
                save_path=save_dir / f"{method_name}_reliability.png"
            )
            
            # Prediction vs uncertainty plot
            plot_predictions_vs_uncertainty(
                y_true, y_pred, y_std,
                save_path=save_dir / f"{method_name}_predictions.png"
            )
    
    # Create comparison plot if multiple methods
    if len(results) > 1 and save_dir is not None:
        _plot_method_comparison(method_metrics, save_dir)
    
    return method_metrics


def _plot_method_comparison(
    method_metrics: Dict[str, Dict[str, float]], 
    save_dir: Path
) -> None:
    """Plot comparison of multiple uncertainty methods."""
    methods = list(method_metrics.keys())
    
    # Key metrics to compare
    key_metrics = ["rmse", "nll", "ece", "coverage_95", "error_uncertainty_correlation"]
    metric_labels = ["RMSE", "NLL", "ECE", "95% Coverage", "Error-Uncertainty Corr."]
    
    fig, axes = plt.subplots(1, len(key_metrics), figsize=(4 * len(key_metrics), 4))
    if len(key_metrics) == 1:
        axes = [axes]
    
    for i, (metric, label) in enumerate(zip(key_metrics, metric_labels)):
        values = [method_metrics[method][metric] for method in methods]
        
        bars = axes[i].bar(methods, values)
        axes[i].set_title(label)
        axes[i].set_ylabel(label)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # Rotate x-axis labels if needed
        if len(max(methods, key=len)) > 8:
            axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_dir / "method_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved method comparison to {save_dir / 'method_comparison.png'}")