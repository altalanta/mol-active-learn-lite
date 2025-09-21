#!/usr/bin/env python3
"""Train molecular property prediction models."""

import argparse
import logging
from pathlib import Path
import sys
import time

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mol_active.data import ESolDataModule
from mol_active.models.ensemble import DeepEnsemble
from mol_active.models.mc_dropout import MCDropoutPredictor
from mol_active.features import MolecularFeaturizer
from mol_active.utils import (
    load_config, merge_configs, setup_logging, 
    set_seed, create_experiment_dir, save_config
)


def main():
    """Main function for training models."""
    parser = argparse.ArgumentParser(description="Train molecular property prediction models")
    parser.add_argument(
        "--data-config",
        type=Path,
        default="configs/data/esol.yaml",
        help="Path to data configuration file"
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default="configs/model/ensemble_mlp.yaml",
        help="Path to model configuration file"
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default="configs/train/default.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="experiments",
        help="Output directory for experiments"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="train_model",
        help="Name of the experiment"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--uncertainty-method",
        choices=["ensemble", "mc_dropout"],
        help="Uncertainty estimation method (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Set random seed
        set_seed(args.seed)
        
        # Load configurations
        data_config = load_config(args.data_config)
        model_config = load_config(args.model_config)
        train_config = load_config(args.train_config)
        
        # Override uncertainty method if specified
        if args.uncertainty_method:
            model_config["uncertainty_method"] = args.uncertainty_method
        
        # Create experiment directory
        experiment_dir = create_experiment_dir(args.output_dir, args.experiment_name)
        
        # Save configurations
        save_config(data_config, experiment_dir / "data_config.yaml")
        save_config(model_config, experiment_dir / "model_config.yaml")
        save_config(train_config, experiment_dir / "train_config.yaml")
        
        # Setup data module
        logger.info("Setting up data module...")
        datamodule = ESolDataModule(
            data_config=data_config,
            batch_size=train_config.get("batch_size", 128),
            num_workers=train_config.get("num_workers", 4),
        )
        
        # Prepare data
        datamodule.prepare_data()
        datamodule.setup()
        
        logger.info(f"Dataset loaded: train={len(datamodule.train_dataset)}, "
                   f"val={len(datamodule.val_dataset)}, test={len(datamodule.test_dataset)}")
        
        # Get feature dimension
        sample_batch = next(iter(datamodule.train_dataloader()))
        input_dim = sample_batch["features"].shape[1]
        logger.info(f"Input feature dimension: {input_dim}")
        
        # Update model config
        model_config["input_dim"] = input_dim
        
        # Train model based on uncertainty method
        uncertainty_method = model_config.get("uncertainty_method", "ensemble")
        
        logger.info(f"Training {uncertainty_method} model...")
        start_time = time.time()
        
        if uncertainty_method == "ensemble":
            # Train ensemble
            model = DeepEnsemble(
                model_config=model_config,
                train_config=train_config,
                ensemble_size=model_config.get("ensemble_size", 5),
                input_dim=input_dim,
            )
            
            model.fit(
                datamodule=datamodule,
                save_dir=experiment_dir / "models"
            )
            
        elif uncertainty_method == "mc_dropout":
            # Train MC-Dropout model
            model = MCDropoutPredictor(
                model_config=model_config,
                train_config=train_config,
                n_samples=model_config.get("mc_samples", 100),
                input_dim=input_dim,
            )
            
            model.fit(
                datamodule=datamodule,
                save_dir=experiment_dir / "models"
            )
            
        else:
            raise ValueError(f"Unknown uncertainty method: {uncertainty_method}")
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save model
        model.save(experiment_dir / "models")
        
        # Quick evaluation on test set
        logger.info("Evaluating on test set...")
        test_dataloader = datamodule.test_dataloader()
        
        # Format dataloader for model prediction
        formatted_batches = []
        for batch in test_dataloader:
            formatted_batch = {
                "features": batch["features"],
                "targets": batch["targets"],
                "index": batch.get("index", torch.arange(len(batch["features"])))
            }
            formatted_batches.append(formatted_batch)
        
        # Get predictions
        results = model.predict(formatted_batches, return_embeddings=False)
        
        # Calculate metrics
        from mol_active.evaluation import uncertainty_metrics
        import numpy as np
        
        # Get true targets
        all_targets = []
        for batch in test_dataloader:
            all_targets.append(batch["targets"].numpy())
        y_true = np.concatenate(all_targets)
        
        metrics = uncertainty_metrics(y_true, results["mean"], results["std"])
        
        logger.info("Test set metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
        
        # Save results
        from mol_active.utils import save_results
        
        final_results = {
            "experiment_config": {
                "data_config": str(args.data_config),
                "model_config": str(args.model_config),
                "train_config": str(args.train_config),
                "uncertainty_method": uncertainty_method,
                "seed": args.seed,
            },
            "training_time": training_time,
            "test_metrics": metrics,
            "model_path": str(experiment_dir / "models"),
        }
        
        save_results(final_results, experiment_dir / "results.yaml")
        
        logger.info(f"Training completed successfully! Results saved to {experiment_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()