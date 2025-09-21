#!/usr/bin/env python3
"""Generate novel molecular candidates using genetic algorithm optimization."""

import argparse
import logging
from pathlib import Path
import sys
import numpy as np

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mol_active.models.ensemble import DeepEnsemble
from mol_active.models.mc_dropout import MCDropoutPredictor
from mol_active.features import MolecularFeaturizer
from mol_active.proposer import PropertyOptimizer, create_initial_population_from_chembl, predict_fitness_with_model
from mol_active.utils import load_config, setup_logging, save_results, set_seed


def create_fitness_function(model, featurizer, target_property="maximize"):
    """Create fitness function for genetic algorithm."""
    def fitness_function(smiles):
        return predict_fitness_with_model(model, featurizer, smiles)
    return fitness_function


def main():
    """Main function for molecular generation."""
    parser = argparse.ArgumentParser(description="Generate novel molecular candidates")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing trained model"
    )
    parser.add_argument(
        "--featurizer-path",
        type=Path,
        help="Path to fitted featurizer (if not in model dir)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="generation",
        help="Output directory for generated molecules"
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=100,
        help="GA population size"
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=50,
        help="Number of GA generations"
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=20,
        help="Number of final candidates to output"
    )
    parser.add_argument(
        "--target-property",
        choices=["maximize", "minimize"],
        default="maximize",
        help="Whether to maximize or minimize the target property"
    )
    parser.add_argument(
        "--initial-smiles",
        nargs="+",
        help="Initial SMILES strings for population (optional)"
    )
    parser.add_argument(
        "--property-constraints",
        type=str,
        help="Property constraints as JSON string, e.g., '{\"mw\": [100, 500]}'"
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
        "--mutation-rate",
        type=float,
        default=0.1,
        help="Mutation rate for GA"
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=0.7,
        help="Crossover rate for GA"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Set random seed
        set_seed(args.seed)
        
        # Load model configuration
        model_dir = args.model_dir
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        model_config = load_config(model_dir / "model_config.yaml")
        
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
        
        # Load featurizer
        if args.featurizer_path:
            featurizer_path = args.featurizer_path
        else:
            # Look for featurizer in model directory or data directory
            featurizer_path = model_dir / "featurizer.pkl"
            if not featurizer_path.exists():
                # Try data directory
                data_config = load_config(model_dir / "data_config.yaml")
                featurizer_path = Path(data_config["cache_dir"]) / "featurizer.pkl"
        
        if featurizer_path.exists():
            logger.info(f"Loading featurizer from {featurizer_path}")
            featurizer = MolecularFeaturizer.load(featurizer_path)
        else:
            logger.warning("Featurizer not found, creating new one from config")
            featurizer = MolecularFeaturizer(model_config)
            # This will need to be fitted, which requires training data
            # For now, we'll assume the featurizer is available
            raise FileNotFoundError(f"Featurizer not found at {featurizer_path}")
        
        # Parse property constraints
        property_constraints = {}
        if args.property_constraints:
            import json
            property_constraints = json.loads(args.property_constraints)
            logger.info(f"Property constraints: {property_constraints}")
        
        # Setup GA configuration
        ga_config = {
            "population_size": args.population_size,
            "num_generations": args.num_generations,
            "tournament_size": 3,
            "elite_fraction": 0.1,
            "property_constraints": property_constraints,
            "mutator_config": {
                "mutation_rate": args.mutation_rate,
                "max_mutations": 3,
            },
            "crossover_config": {
                "crossover_rate": args.crossover_rate,
            },
        }
        
        # Create optimizer
        optimizer = PropertyOptimizer(ga_config)
        
        # Create initial population
        if args.initial_smiles:
            logger.info(f"Using provided initial SMILES: {args.initial_smiles}")
            initial_population = args.initial_smiles
            
            # Pad with drug-like molecules if needed
            if len(initial_population) < args.population_size:
                additional_molecules = create_initial_population_from_chembl(
                    args.population_size - len(initial_population),
                    property_filters=property_constraints,
                    seed=args.seed
                )
                initial_population.extend(additional_molecules)
        else:
            logger.info("Creating initial population from drug-like molecules")
            initial_population = create_initial_population_from_chembl(
                args.population_size,
                property_filters=property_constraints,
                seed=args.seed
            )
        
        logger.info(f"Initial population size: {len(initial_population)}")
        
        # Create fitness function
        fitness_function = create_fitness_function(model, featurizer, args.target_property)
        
        # Run optimization
        logger.info("Starting molecular optimization...")
        final_population, final_fitness = optimizer.optimize(
            initial_population=initial_population,
            fitness_function=fitness_function,
            target_property=args.target_property,
            verbose=True
        )
        
        # Sort by fitness and select top candidates
        fitness_array = np.array(final_fitness)
        
        if args.target_property == "maximize":
            sorted_indices = np.argsort(fitness_array)[::-1]
        else:
            sorted_indices = np.argsort(fitness_array)
        
        top_candidates = []
        top_fitness = []
        
        for i in sorted_indices[:args.num_candidates]:
            if not np.isinf(fitness_array[i]):  # Exclude invalid molecules
                top_candidates.append(final_population[i])
                top_fitness.append(fitness_array[i])
        
        logger.info(f"Selected {len(top_candidates)} top candidates")
        
        # Get additional predictions for top candidates
        logger.info("Getting detailed predictions for top candidates...")
        candidate_details = []
        
        for smiles, fitness in zip(top_candidates, top_fitness):
            try:
                # Get uncertainty estimate
                features = featurizer.transform([smiles])
                
                import torch
                from torch.utils.data import DataLoader, TensorDataset
                
                feature_tensor = torch.FloatTensor(features)
                dataset = TensorDataset(feature_tensor)
                dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
                
                formatted_dataloader = []
                for batch in dataloader:
                    formatted_batch = {"features": batch[0]}
                    formatted_dataloader.append(formatted_batch)
                
                results = model.predict(formatted_dataloader, return_embeddings=False)
                prediction_mean = results["mean"][0]
                prediction_std = results["std"][0]
                
                # Calculate molecular properties
                from mol_active.features import compute_molecular_descriptors
                from rdkit import Chem
                
                mol = Chem.MolFromSmiles(smiles)
                descriptors = compute_molecular_descriptors(mol) if mol else {}
                
                candidate_details.append({
                    "smiles": smiles,
                    "fitness": float(fitness),
                    "prediction_mean": float(prediction_mean),
                    "prediction_std": float(prediction_std),
                    "molecular_properties": descriptors,
                })
                
            except Exception as e:
                logger.warning(f"Failed to get details for {smiles}: {e}")
                candidate_details.append({
                    "smiles": smiles,
                    "fitness": float(fitness),
                    "prediction_mean": None,
                    "prediction_std": None,
                    "molecular_properties": {},
                })
        
        # Create output directory
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        generation_results = {
            "experiment_config": {
                "model_dir": str(model_dir),
                "uncertainty_method": uncertainty_method,
                "target_property": args.target_property,
                "population_size": args.population_size,
                "num_generations": args.num_generations,
                "mutation_rate": args.mutation_rate,
                "crossover_rate": args.crossover_rate,
                "property_constraints": property_constraints,
                "seed": args.seed,
            },
            "optimization_summary": {
                "initial_population_size": len(initial_population),
                "final_population_size": len(final_population),
                "num_valid_candidates": len(top_candidates),
                "best_fitness": float(top_fitness[0]) if top_fitness else None,
            },
            "top_candidates": candidate_details,
        }
        
        save_results(generation_results, output_dir / "generation_results.yaml")
        
        # Save SMILES list
        with open(output_dir / "top_candidates.smi", "w") as f:
            for candidate in candidate_details:
                f.write(f"{candidate['smiles']}\t{candidate['fitness']:.4f}\n")
        
        # Create summary plot
        try:
            import matplotlib.pyplot as plt
            
            if len(top_fitness) > 0:
                plt.figure(figsize=(10, 6))
                
                # Plot fitness distribution
                plt.subplot(1, 2, 1)
                plt.bar(range(len(top_fitness)), top_fitness)
                plt.xlabel('Candidate Rank')
                plt.ylabel('Fitness Score')
                plt.title('Top Candidate Fitness Scores')
                plt.grid(True, alpha=0.3)
                
                # Plot prediction uncertainty
                plt.subplot(1, 2, 2)
                uncertainties = [c["prediction_std"] for c in candidate_details if c["prediction_std"] is not None]
                fitness_vals = [c["fitness"] for c in candidate_details if c["prediction_std"] is not None]
                
                if uncertainties:
                    plt.scatter(uncertainties, fitness_vals, alpha=0.7)
                    plt.xlabel('Prediction Uncertainty')
                    plt.ylabel('Fitness Score')
                    plt.title('Fitness vs. Uncertainty')
                    plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(output_dir / "generation_summary.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info("Summary plots saved")
            
        except Exception as e:
            logger.warning(f"Failed to create summary plots: {e}")
        
        logger.info(f"Molecular generation completed! Results saved to {output_dir}")
        
        # Print summary
        logger.info("\nGeneration Results Summary:")
        logger.info(f"Generated {len(top_candidates)} valid candidates")
        if top_fitness:
            logger.info(f"Best fitness: {top_fitness[0]:.4f}")
            logger.info(f"Best molecule: {top_candidates[0]}")
            
            logger.info("\nTop 5 candidates:")
            for i, (smiles, fitness) in enumerate(zip(top_candidates[:5], top_fitness[:5])):
                uncertainty = candidate_details[i].get("prediction_std", "N/A")
                logger.info(f"  {i+1:2d}. {smiles} (fitness: {fitness:.4f}, uncertainty: {uncertainty})")
        
    except Exception as e:
        logger.error(f"Molecular generation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()