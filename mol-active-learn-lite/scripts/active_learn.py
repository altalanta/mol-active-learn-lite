#!/usr/bin/env python3
"""Run active learning experiments for molecular property prediction."""

import argparse
import logging
from pathlib import Path
import sys
import time
import numpy as np
import torch

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mol_active.data import ESolDataModule, create_active_learning_splits
from mol_active.models.ensemble import DeepEnsemble
from mol_active.models.mc_dropout import MCDropoutPredictor
from mol_active.active_learning import ActiveLearner, random_selection, uncertainty_sampling
from mol_active.evaluation import uncertainty_metrics
from mol_active.utils import (
    load_config, merge_configs, setup_logging, 
    set_seed, create_experiment_dir, save_config, save_results
)


def create_subset_datamodule(datamodule, indices, config):
    """Create a subset data module with specific indices."""
    # Get the original dataset
    full_dataset = datamodule.train_dataset
    
    # Create subset
    subset_smiles = [full_dataset.smiles[i] for i in indices]
    subset_targets = full_dataset.targets[indices].numpy()
    subset_features = full_dataset.features[indices].numpy()
    subset_indices = indices
    
    # Import here to avoid circular imports
    from mol_active.data import MolecularDataset
    
    subset_dataset = MolecularDataset(
        smiles=subset_smiles,
        targets=subset_targets,
        features=subset_features,
        indices=subset_indices
    )
    
    # Create new datamodule with subset
    from mol_active.data import ESolDataModule
    subset_datamodule = ESolDataModule(
        data_config=config,
        batch_size=128,
        num_workers=4,
    )
    
    # Override the datasets
    subset_datamodule.train_dataset = subset_dataset
    subset_datamodule.val_dataset = datamodule.val_dataset
    subset_datamodule.test_dataset = datamodule.test_dataset
    
    return subset_datamodule


def main():
    """Main function for active learning."""
    parser = argparse.ArgumentParser(description="Run active learning experiments")
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
        "--al-config",
        type=Path,
        default="configs/al/default.yaml",
        help="Path to active learning configuration file"
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
        default="active_learning",
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
        "--baseline",
        choices=["random", "uncertainty"],
        help="Run baseline method instead of full active learning"
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
        al_config = load_config(args.al_config)
        
        # Create experiment directory
        experiment_name = f"{args.experiment_name}_{args.baseline}" if args.baseline else args.experiment_name
        experiment_dir = create_experiment_dir(args.output_dir, experiment_name)
        
        # Save configurations
        save_config(data_config, experiment_dir / "data_config.yaml")
        save_config(model_config, experiment_dir / "model_config.yaml")
        save_config(train_config, experiment_dir / "train_config.yaml")
        save_config(al_config, experiment_dir / "al_config.yaml")
        
        # Setup data module
        logger.info("Setting up data module...")
        datamodule = ESolDataModule(
            data_config=data_config,
            batch_size=train_config.get("batch_size", 128),
            num_workers=train_config.get("num_workers", 4),
        )
        
        datamodule.prepare_data()
        datamodule.setup()
        
        # Create active learning splits
        logger.info("Creating active learning splits...")
        import pandas as pd
        
        # Load full training data
        df_path = Path(data_config["processed_file"])
        df = pd.read_csv(df_path)
        
        # Get training indices (we'll split the training set into seed + pool)
        train_size = len(datamodule.train_dataset)
        seed_size = int(al_config["seed_size"] * train_size)
        
        # Create seed and pool indices
        all_train_indices = np.arange(train_size)
        np.random.shuffle(all_train_indices)
        
        seed_indices = all_train_indices[:seed_size]
        pool_indices = all_train_indices[seed_size:]
        
        logger.info(f"Active learning setup: seed={len(seed_indices)}, pool={len(pool_indices)}")
        
        # AL parameters
        num_rounds = al_config["num_rounds"]
        batch_size = al_config["batch_size"]
        uncertainty_method = model_config.get("uncertainty_method", "ensemble")
        
        # Track results
        al_history = []
        current_labeled_indices = seed_indices.copy()
        current_pool_indices = pool_indices.copy()
        
        # Active learning loop
        for round_num in range(num_rounds):
            logger.info(f"\n--- Active Learning Round {round_num + 1}/{num_rounds} ---")
            logger.info(f"Labeled samples: {len(current_labeled_indices)}")
            logger.info(f"Pool samples: {len(current_pool_indices)}")
            
            # Create training datamodule with current labeled samples
            subset_datamodule = create_subset_datamodule(
                datamodule, current_labeled_indices, data_config
            )
            
            # Train model on current labeled set
            logger.info(f"Training {uncertainty_method} model...")
            start_time = time.time()
            
            # Get input dimension
            sample_batch = next(iter(subset_datamodule.train_dataloader()))
            input_dim = sample_batch["features"].shape[1]
            
            if uncertainty_method == "ensemble":
                model = DeepEnsemble(
                    model_config=model_config,
                    train_config=train_config,
                    ensemble_size=model_config.get("ensemble_size", 5),
                    input_dim=input_dim,
                )
            else:
                model = MCDropoutPredictor(
                    model_config=model_config,
                    train_config=train_config,
                    n_samples=model_config.get("mc_samples", 100),
                    input_dim=input_dim,
                )
            
            model.fit(
                datamodule=subset_datamodule,
                save_dir=experiment_dir / f"models_round_{round_num}"
            )
            
            training_time = time.time() - start_time
            
            # Evaluate on test set
            test_dataloader = datamodule.test_dataloader()
            formatted_test_batches = []
            all_test_targets = []
            
            for batch in test_dataloader:
                formatted_batch = {
                    "features": batch["features"],
                    "targets": batch["targets"],
                    "index": batch.get("index", torch.arange(len(batch["features"])))
                }
                formatted_test_batches.append(formatted_batch)
                all_test_targets.append(batch["targets"].numpy())
            
            test_results = model.predict(formatted_test_batches, return_embeddings=True)
            y_test_true = np.concatenate(all_test_targets)
            test_metrics = uncertainty_metrics(y_test_true, test_results["mean"], test_results["std"])
            
            logger.info(f"Test RMSE: {test_metrics['rmse']:.4f}, NLL: {test_metrics['nll']:.4f}")
            
            # Select next batch if not final round
            if round_num < num_rounds - 1:
                # Get pool features and embeddings
                pool_features = datamodule.train_dataset.features[current_pool_indices]
                
                # Get embeddings for pool
                pool_dataloader_batches = []
                pool_batch_size = 256
                
                for i in range(0, len(current_pool_indices), pool_batch_size):
                    batch_indices = current_pool_indices[i:i + pool_batch_size]
                    batch_features = datamodule.train_dataset.features[batch_indices]
                    
                    formatted_batch = {
                        "features": batch_features,
                        "index": torch.arange(len(batch_features))
                    }
                    pool_dataloader_batches.append(formatted_batch)
                
                pool_results = model.predict(pool_dataloader_batches, return_embeddings=True)
                pool_embeddings = pool_results["embeddings"]
                
                # Get labeled embeddings and targets
                labeled_dataloader_batches = []
                for i in range(0, len(current_labeled_indices), pool_batch_size):
                    batch_indices = current_labeled_indices[i:i + pool_batch_size]
                    batch_features = datamodule.train_dataset.features[batch_indices]
                    
                    formatted_batch = {
                        "features": batch_features,
                        "index": torch.arange(len(batch_features))
                    }
                    labeled_dataloader_batches.append(formatted_batch)
                
                labeled_results = model.predict(labeled_dataloader_batches, return_embeddings=True)
                labeled_embeddings = labeled_results["embeddings"]
                labeled_targets = datamodule.train_dataset.targets[current_labeled_indices].numpy()
                
                # Select candidates
                if args.baseline == "random":
                    selected_pool_indices = random_selection(
                        len(current_pool_indices), batch_size, seed=args.seed + round_num
                    )
                elif args.baseline == "uncertainty":
                    selected_pool_indices = uncertainty_sampling(
                        pool_results["std"], batch_size
                    )
                else:
                    # Full active learning with GP surrogate
                    active_learner = ActiveLearner(model, al_config)
                    
                    # Best observed value for Expected Improvement
                    y_best = np.max(labeled_targets) if al_config.get("acquisition_function") == "ei" else None
                    
                    selected_pool_indices = active_learner.select_candidates(
                        pool_features.numpy(),
                        pool_embeddings,
                        labeled_embeddings,
                        labeled_targets,
                        batch_size,
                        y_best=y_best
                    )
                
                # Update labeled and pool sets
                selected_global_indices = current_pool_indices[selected_pool_indices]
                current_labeled_indices = np.concatenate([current_labeled_indices, selected_global_indices])
                current_pool_indices = np.delete(current_pool_indices, selected_pool_indices)
                
                logger.info(f"Selected {len(selected_pool_indices)} new samples for labeling")
            
            # Record round results
            round_results = {
                "round": round_num,
                "num_labeled": len(current_labeled_indices),
                "num_pool": len(current_pool_indices),
                "training_time": training_time,
                "test_metrics": test_metrics,
            }
            
            if round_num < num_rounds - 1:
                round_results["selected_indices"] = selected_global_indices.tolist()
            
            al_history.append(round_results)
            
            logger.info(f"Round {round_num + 1} completed in {training_time:.2f}s")
        
        # Save final results
        final_results = {
            "experiment_config": {
                "data_config": str(args.data_config),
                "model_config": str(args.model_config),
                "train_config": str(args.train_config),
                "al_config": str(args.al_config),
                "uncertainty_method": uncertainty_method,
                "baseline": args.baseline,
                "seed": args.seed,
            },
            "al_setup": {
                "initial_seed_size": len(seed_indices),
                "initial_pool_size": len(pool_indices),
                "num_rounds": num_rounds,
                "batch_size": batch_size,
            },
            "al_history": al_history,
        }
        
        save_results(final_results, experiment_dir / "al_results.yaml")
        
        # Create learning curve plot
        try:
            import matplotlib.pyplot as plt
            
            rounds = [r["round"] + 1 for r in al_history]
            labeled_sizes = [r["num_labeled"] for r in al_history]
            test_rmse = [r["test_metrics"]["rmse"] for r in al_history]
            test_nll = [r["test_metrics"]["nll"] for r in al_history]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # RMSE learning curve
            ax1.plot(labeled_sizes, test_rmse, 'o-', linewidth=2, markersize=6)
            ax1.set_xlabel('Number of Labeled Samples')
            ax1.set_ylabel('Test RMSE')
            ax1.set_title('Active Learning Curve (RMSE)')
            ax1.grid(True, alpha=0.3)
            
            # NLL learning curve
            ax2.plot(labeled_sizes, test_nll, 'o-', linewidth=2, markersize=6, color='orange')
            ax2.set_xlabel('Number of Labeled Samples')
            ax2.set_ylabel('Test NLL')
            ax2.set_title('Active Learning Curve (NLL)')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(experiment_dir / "learning_curves.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Learning curves saved")
            
        except Exception as e:
            logger.warning(f"Failed to create learning curves: {e}")
        
        logger.info(f"Active learning experiment completed! Results saved to {experiment_dir}")
        
        # Print summary
        logger.info("\nFinal Results Summary:")
        logger.info(f"Initial labeled samples: {len(seed_indices)}")
        logger.info(f"Final labeled samples: {len(current_labeled_indices)}")
        logger.info(f"Final test RMSE: {al_history[-1]['test_metrics']['rmse']:.4f}")
        logger.info(f"Final test NLL: {al_history[-1]['test_metrics']['nll']:.4f}")
        
    except Exception as e:
        logger.error(f"Active learning experiment failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()