#!/usr/bin/env python3
"""Evaluate trained models with comprehensive uncertainty analysis."""

import argparse
import logging
from pathlib import Path
import sys
import numpy as np
import torch

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mol_active.data import ESolDataModule
from mol_active.models.ensemble import DeepEnsemble
from mol_active.models.mc_dropout import MCDropoutPredictor
from mol_active.evaluation import (
    uncertainty_metrics, reliability_diagram, 
    plot_predictions_vs_uncertainty, compare_uncertainty_methods
)
from mol_active.utils import load_config, setup_logging, load_results


def main():
    """Main function for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate trained molecular property prediction models")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing trained model"
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        help="Path to data configuration file (if not in model dir)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="evaluation",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate evaluation plots"
    )
    parser.add_argument(
        "--compare-methods",
        action="store_true",
        help="Compare multiple uncertainty methods if available"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load model configuration
        model_dir = args.model_dir
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Load configurations from model directory
        if args.data_config:
            data_config = load_config(args.data_config)
        else:
            data_config = load_config(model_dir / "data_config.yaml")
        
        model_config = load_config(model_dir / "model_config.yaml")
        train_config = load_config(model_dir / "train_config.yaml")
        
        # Setup data module
        logger.info("Setting up data module...")
        datamodule = ESolDataModule(
            data_config=data_config,
            batch_size=128,
            num_workers=4,
        )
        datamodule.prepare_data()
        datamodule.setup()
        
        # Get the specified dataloader
        if args.split == "train":
            dataloader = datamodule.train_dataloader()
        elif args.split == "val":
            dataloader = datamodule.val_dataloader()
        else:
            dataloader = datamodule.test_dataloader()
        
        logger.info(f"Evaluating on {args.split} set with {len(dataloader.dataset)} samples")
        
        # Load model
        uncertainty_method = model_config.get("uncertainty_method", "ensemble")
        logger.info(f"Loading {uncertainty_method} model...")
        
        models_path = model_dir / "models"
        
        if uncertainty_method == "ensemble":
            model = DeepEnsemble.load(models_path)
        elif uncertainty_method == "mc_dropout":
            model = MCDropoutPredictor.load(models_path)
        else:
            raise ValueError(f"Unknown uncertainty method: {uncertainty_method}")
        
        # Format dataloader for model prediction
        formatted_batches = []
        all_targets = []
        
        for batch in dataloader:
            formatted_batch = {
                "features": batch["features"],
                "targets": batch["targets"],
                "index": batch.get("index", torch.arange(len(batch["features"])))
            }
            formatted_batches.append(formatted_batch)
            all_targets.append(batch["targets"].numpy())
        
        # Get predictions
        logger.info("Making predictions...")
        results = model.predict(formatted_batches, return_embeddings=True)
        
        # Get true targets
        y_true = np.concatenate(all_targets)
        y_pred = results["mean"]
        y_std = results["std"]
        
        logger.info(f"Prediction statistics:")
        logger.info(f"  Mean prediction: {np.mean(y_pred):.3f} ± {np.std(y_pred):.3f}")
        logger.info(f"  Mean uncertainty: {np.mean(y_std):.3f} ± {np.std(y_std):.3f}")
        
        # Calculate comprehensive metrics
        logger.info("Calculating evaluation metrics...")
        metrics = uncertainty_metrics(y_true, y_pred, y_std)
        
        logger.info("Evaluation metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
        
        # Create output directory
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate plots if requested
        if args.plot:
            logger.info("Generating evaluation plots...")
            
            # Reliability diagram
            reliability_data = reliability_diagram(
                y_true, y_pred, y_std,
                save_path=output_dir / f"reliability_diagram_{args.split}.png"
            )
            
            # Predictions vs uncertainty plot
            plot_predictions_vs_uncertainty(
                y_true, y_pred, y_std,
                save_path=output_dir / f"predictions_vs_uncertainty_{args.split}.png"
            )
            
            logger.info(f"Plots saved to {output_dir}")
        
        # Compare methods if requested and multiple methods available
        if args.compare_methods:
            logger.info("Checking for additional uncertainty methods to compare...")
            
            # Try to load both ensemble and MC-dropout if available
            comparison_results = {uncertainty_method: {"mean": y_pred, "std": y_std}}
            
            # Try to load the other method
            other_method = "mc_dropout" if uncertainty_method == "ensemble" else "ensemble"
            other_models_path = model_dir.parent / f"{other_method}_models"
            
            if other_models_path.exists():
                logger.info(f"Found {other_method} model, adding to comparison...")
                try:
                    if other_method == "ensemble":
                        other_model = DeepEnsemble.load(other_models_path)
                    else:
                        other_model = MCDropoutPredictor.load(other_models_path)
                    
                    other_results = other_model.predict(formatted_batches, return_embeddings=False)
                    comparison_results[other_method] = {
                        "mean": other_results["mean"],
                        "std": other_results["std"]
                    }
                except Exception as e:
                    logger.warning(f"Failed to load {other_method} model: {e}")
            
            if len(comparison_results) > 1:
                method_metrics = compare_uncertainty_methods(
                    comparison_results,
                    y_true,
                    save_dir=output_dir / "method_comparison" if args.plot else None
                )
                
                logger.info("Method comparison:")
                for method, method_metrics_dict in method_metrics.items():
                    logger.info(f"  {method}:")
                    for metric_name, value in method_metrics_dict.items():
                        logger.info(f"    {metric_name}: {value:.4f}")
        
        # Save evaluation results
        from mol_active.utils import save_results
        
        evaluation_results = {
            "model_config": {
                "model_dir": str(model_dir),
                "uncertainty_method": uncertainty_method,
                "split": args.split,
            },
            "dataset_stats": {
                "num_samples": len(y_true),
                "target_mean": float(np.mean(y_true)),
                "target_std": float(np.std(y_true)),
                "prediction_mean": float(np.mean(y_pred)),
                "prediction_std": float(np.std(y_pred)),
                "uncertainty_mean": float(np.mean(y_std)),
                "uncertainty_std": float(np.std(y_std)),
            },
            "metrics": metrics,
        }
        
        if args.plot:
            evaluation_results["reliability_data"] = {
                "bin_confidences": reliability_data["bin_confidences"].tolist(),
                "bin_accuracies": reliability_data["bin_accuracies"].tolist(),
                "bin_counts": reliability_data["bin_counts"].tolist(),
            }
        
        save_results(evaluation_results, output_dir / f"evaluation_results_{args.split}.yaml")
        
        logger.info(f"Evaluation completed successfully! Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()