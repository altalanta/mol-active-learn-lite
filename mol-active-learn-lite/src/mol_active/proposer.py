"""Molecular proposer using SMILES genetic algorithm for property optimization."""

import logging
import random
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from .features import validate_smiles, canonicalize_smiles

logger = logging.getLogger(__name__)


class SMILESMutator:
    """SMILES string mutator for genetic algorithm."""
    
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.mutation_rate = config.get("mutation_rate", 0.1)
        self.max_mutations = config.get("max_mutations", 3)
        
        # Common molecular fragments for insertion
        self.fragments = [
            "C", "CC", "CCC", "O", "N", "S", "F", "Cl", "Br",
            "c1ccccc1", "C=C", "C#C", "CO", "CN", "CF",
            "C(=O)", "C(=O)O", "C(=O)N", "S(=O)(=O)",
        ]
        
    def mutate(self, smiles: str) -> Optional[str]:
        """Apply random mutations to a SMILES string.
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            Mutated SMILES string or None if mutation failed
        """
        if random.random() > self.mutation_rate:
            return smiles
            
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        # Choose number of mutations
        num_mutations = random.randint(1, self.max_mutations)
        
        for _ in range(num_mutations):
            mutation_type = random.choice(["substitute", "insert", "delete"])
            
            if mutation_type == "substitute":
                mol = self._substitute_atom(mol)
            elif mutation_type == "insert":
                mol = self._insert_fragment(mol)
            elif mutation_type == "delete":
                mol = self._delete_fragment(mol)
                
            if mol is None:
                return None
                
        try:
            mutated_smiles = Chem.MolToSmiles(mol)
            return canonicalize_smiles(mutated_smiles)
        except:
            return None
            
    def _substitute_atom(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Substitute a random atom with another atom."""
        if mol.GetNumAtoms() == 0:
            return None
            
        # Choose random atom to substitute
        atom_idx = random.randint(0, mol.GetNumAtoms() - 1)
        atom = mol.GetAtomWithIdx(atom_idx)
        
        # Define possible substitutions based on current atom
        substitutions = {
            6: [7, 8, 16],  # C -> N, O, S
            7: [6, 8],      # N -> C, O
            8: [6, 7, 16],  # O -> C, N, S
            16: [6, 8],     # S -> C, O
        }
        
        current_atomic_num = atom.GetAtomicNum()
        if current_atomic_num not in substitutions:
            return mol
            
        new_atomic_num = random.choice(substitutions[current_atomic_num])
        
        # Create editable molecule
        edit_mol = Chem.EditableMol(mol)
        
        # Remove old atom and add new one
        neighbors = [mol.GetAtomWithIdx(n.GetIdx()) for n in atom.GetNeighbors()]
        neighbor_indices = [n.GetIdx() for n in neighbors]
        
        # This is a simplified substitution - in practice, you'd want more sophisticated logic
        edit_mol.ReplaceAtom(atom_idx, Chem.Atom(new_atomic_num))
        
        try:
            new_mol = edit_mol.GetMol()
            Chem.SanitizeMol(new_mol)
            return new_mol
        except:
            return mol
            
    def _insert_fragment(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Insert a random fragment into the molecule."""
        fragment_smiles = random.choice(self.fragments)
        fragment_mol = Chem.MolFromSmiles(fragment_smiles)
        
        if fragment_mol is None:
            return mol
            
        # Simple concatenation - in practice, you'd want smarter fragment insertion
        try:
            combined_mol = Chem.CombineMols(mol, fragment_mol)
            Chem.SanitizeMol(combined_mol)
            return combined_mol
        except:
            return mol
            
    def _delete_fragment(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Delete a random fragment from the molecule."""
        if mol.GetNumAtoms() <= 3:  # Don't make molecules too small
            return mol
            
        # Choose random atom to remove
        atom_idx = random.randint(0, mol.GetNumAtoms() - 1)
        
        edit_mol = Chem.EditableMol(mol)
        edit_mol.RemoveAtom(atom_idx)
        
        try:
            new_mol = edit_mol.GetMol()
            Chem.SanitizeMol(new_mol)
            
            # Check if molecule is still connected
            if len(Chem.GetMolFrags(new_mol)) == 1:
                return new_mol
            else:
                # Return largest fragment
                frags = Chem.GetMolFrags(new_mol, asMols=True)
                largest_frag = max(frags, key=lambda x: x.GetNumAtoms())
                return largest_frag
        except:
            return mol


class SMILESCrossover:
    """SMILES crossover operator for genetic algorithm."""
    
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.crossover_rate = config.get("crossover_rate", 0.7)
        
    def crossover(self, parent1: str, parent2: str) -> Tuple[Optional[str], Optional[str]]:
        """Perform crossover between two parent SMILES.
        
        Args:
            parent1: First parent SMILES
            parent2: Second parent SMILES
            
        Returns:
            Tuple of two offspring SMILES (or None if crossover failed)
        """
        if random.random() > self.crossover_rate:
            return parent1, parent2
            
        mol1 = Chem.MolFromSmiles(parent1)
        mol2 = Chem.MolFromSmiles(parent2)
        
        if mol1 is None or mol2 is None:
            return parent1, parent2
            
        # Simple fragment-based crossover
        try:
            # Get fragments from each molecule
            frags1 = self._get_fragments(mol1)
            frags2 = self._get_fragments(mol2)
            
            if len(frags1) < 2 or len(frags2) < 2:
                return parent1, parent2
                
            # Create offspring by combining fragments
            offspring1 = self._combine_fragments(frags1[:len(frags1)//2] + frags2[len(frags2)//2:])
            offspring2 = self._combine_fragments(frags2[:len(frags2)//2] + frags1[len(frags1)//2:])
            
            return offspring1, offspring2
            
        except:
            return parent1, parent2
            
    def _get_fragments(self, mol: Chem.Mol) -> List[str]:
        """Get molecular fragments for crossover."""
        # This is a simplified approach - break molecule at random bonds
        fragments = []
        smiles = Chem.MolToSmiles(mol)
        
        # Simple approach: split SMILES string at certain characters
        separators = [".", "(", ")", "[", "]"]
        current_frag = ""
        
        for char in smiles:
            if char in separators and current_frag:
                fragments.append(current_frag)
                current_frag = ""
            else:
                current_frag += char
                
        if current_frag:
            fragments.append(current_frag)
            
        return fragments if fragments else [smiles]
        
    def _combine_fragments(self, fragments: List[str]) -> Optional[str]:
        """Combine fragments into a valid SMILES."""
        combined = "".join(fragments)
        return canonicalize_smiles(combined)


class PropertyOptimizer:
    """Property-guided molecular optimization using genetic algorithm."""
    
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.mutator = SMILESMutator(config.get("mutator_config", {}))
        self.crossover = SMILESCrossover(config.get("crossover_config", {}))
        
        # GA parameters
        self.population_size = config.get("population_size", 100)
        self.num_generations = config.get("num_generations", 50)
        self.tournament_size = config.get("tournament_size", 3)
        self.elite_fraction = config.get("elite_fraction", 0.1)
        
        # Property constraints
        self.property_constraints = config.get("property_constraints", {})
        
    def optimize(
        self,
        initial_population: List[str],
        fitness_function: callable,
        target_property: str = "maximize",
        verbose: bool = True
    ) -> Tuple[List[str], List[float]]:
        """Run genetic algorithm to optimize molecular properties.
        
        Args:
            initial_population: Initial SMILES strings
            fitness_function: Function that takes SMILES and returns fitness score
            target_property: "maximize" or "minimize"
            verbose: Whether to print progress
            
        Returns:
            Tuple of (final_population, fitness_scores)
        """
        logger.info(f"Starting GA optimization with {len(initial_population)} molecules")
        
        # Initialize population
        population = self._validate_population(initial_population)
        
        best_fitness_history = []
        
        for generation in range(self.num_generations):
            # Evaluate fitness
            fitness_scores = []
            for smiles in population:
                try:
                    score = fitness_function(smiles)
                    if not self._satisfies_constraints(smiles):
                        score = -np.inf  # Penalize constraint violations
                    fitness_scores.append(score)
                except:
                    fitness_scores.append(-np.inf)
                    
            fitness_scores = np.array(fitness_scores)
            
            # Track best fitness
            if target_property == "maximize":
                best_fitness = np.max(fitness_scores)
                best_idx = np.argmax(fitness_scores)
            else:
                best_fitness = np.min(fitness_scores)
                best_idx = np.argmin(fitness_scores)
                
            best_fitness_history.append(best_fitness)
            
            if verbose and generation % 10 == 0:
                logger.info(f"Generation {generation}: best fitness = {best_fitness:.3f}, "
                           f"best molecule = {population[best_idx]}")
                
            # Selection and reproduction
            new_population = []
            
            # Elitism: keep best molecules
            num_elite = int(self.elite_fraction * self.population_size)
            if target_property == "maximize":
                elite_indices = np.argsort(fitness_scores)[-num_elite:]
            else:
                elite_indices = np.argsort(fitness_scores)[:num_elite]
                
            for idx in elite_indices:
                new_population.append(population[idx])
                
            # Generate offspring
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self._tournament_selection(population, fitness_scores, target_property)
                parent2 = self._tournament_selection(population, fitness_scores, target_property)
                
                # Crossover
                offspring1, offspring2 = self.crossover.crossover(parent1, parent2)
                
                # Mutation
                if offspring1:
                    offspring1 = self.mutator.mutate(offspring1)
                if offspring2:
                    offspring2 = self.mutator.mutate(offspring2)
                    
                # Add valid offspring
                for offspring in [offspring1, offspring2]:
                    if offspring and validate_smiles(offspring) and len(new_population) < self.population_size:
                        new_population.append(offspring)
                        
            population = new_population[:self.population_size]
            
        # Final evaluation
        final_fitness = []
        for smiles in population:
            try:
                score = fitness_function(smiles)
                final_fitness.append(score)
            except:
                final_fitness.append(-np.inf)
                
        logger.info("GA optimization completed")
        return population, final_fitness
        
    def _validate_population(self, population: List[str]) -> List[str]:
        """Validate and filter initial population."""
        valid_population = []
        for smiles in population:
            if validate_smiles(smiles):
                canonical = canonicalize_smiles(smiles)
                if canonical:
                    valid_population.append(canonical)
                    
        if len(valid_population) < self.population_size:
            logger.warning(f"Only {len(valid_population)} valid molecules in initial population")
            
        return valid_population
        
    def _tournament_selection(
        self,
        population: List[str],
        fitness_scores: np.ndarray,
        target_property: str
    ) -> str:
        """Tournament selection of parent."""
        tournament_indices = np.random.choice(
            len(population), 
            size=min(self.tournament_size, len(population)), 
            replace=False
        )
        tournament_fitness = fitness_scores[tournament_indices]
        
        if target_property == "maximize":
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        else:
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
            
        return population[winner_idx]
        
    def _satisfies_constraints(self, smiles: str) -> bool:
        """Check if molecule satisfies property constraints."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
            
        for prop_name, (min_val, max_val) in self.property_constraints.items():
            try:
                if prop_name == "mw":
                    value = Descriptors.MolWt(mol)
                elif prop_name == "logp":
                    value = Crippen.MolLogP(mol)
                elif prop_name == "tpsa":
                    value = rdMolDescriptors.CalcTPSA(mol)
                elif prop_name == "heavy_atoms":
                    value = mol.GetNumHeavyAtoms()
                else:
                    continue
                    
                if not (min_val <= value <= max_val):
                    return False
            except:
                return False
                
        return True


def create_initial_population_from_chembl(
    size: int,
    property_filters: Optional[Dict] = None,
    seed: Optional[int] = None
) -> List[str]:
    """Create initial population from ChEMBL-like drug molecules.
    
    Args:
        size: Population size
        property_filters: Optional property filters
        seed: Random seed
        
    Returns:
        List of SMILES strings
    """
    # This is a simplified version - in practice, you'd query ChEMBL database
    # Here we provide some drug-like molecules as starting points
    
    drug_like_smiles = [
        "CCO",  # Ethanol
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CC1=C(C=C(C=C1)C(=O)C2=CC=CC=C2)C",  # Tolmetin core
        "C1=CC=C(C=C1)C(=O)O",  # Benzoic acid
        "CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O",  # Salbutamol
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CN(C)CCOC1=CC=C(C=C1)CC2=CC=CC=C2",  # Diphenhydramine core
        "C1=CC=C2C(=C1)C(=CN2)CCN",  # Tryptamine
        "CC1=CC(=NO1)C2=CC=CC=C2C(=O)NC3=CC=CC=C3",  # Oxaprozin
    ]
    
    if seed is not None:
        random.seed(seed)
        
    # Start with drug-like molecules and add variations
    population = []
    
    # Add original drug-like molecules
    for smiles in drug_like_smiles:
        if len(population) < size:
            canonical = canonicalize_smiles(smiles)
            if canonical and validate_smiles(canonical):
                population.append(canonical)
                
    # Fill remaining slots with simple molecules
    simple_molecules = [
        "C", "CC", "CCC", "CCCC", "CO", "CCO", "CN", "CCN",
        "c1ccccc1", "Cc1ccccc1", "c1ccccc1O", "c1ccccc1N",
        "C=C", "C=CC", "C#C", "C#CC",
    ]
    
    while len(population) < size:
        smiles = random.choice(simple_molecules)
        canonical = canonicalize_smiles(smiles)
        if canonical and canonical not in population:
            population.append(canonical)
            
    return population[:size]


def predict_fitness_with_model(
    uncertainty_model,
    featurizer,
    smiles: str
) -> float:
    """Use trained model to predict fitness (property value) for a SMILES.
    
    Args:
        uncertainty_model: Trained uncertainty model (ensemble or MC-dropout)
        featurizer: Fitted molecular featurizer
        smiles: SMILES string
        
    Returns:
        Predicted property value (fitness score)
    """
    try:
        # Compute features
        features = featurizer.transform([smiles])
        
        # Create simple dataloader
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        
        feature_tensor = torch.FloatTensor(features)
        dataset = TensorDataset(feature_tensor)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # Format for model
        formatted_dataloader = []
        for batch in dataloader:
            formatted_batch = {"features": batch[0]}
            formatted_dataloader.append(formatted_batch)
        
        # Get prediction
        results = uncertainty_model.predict(formatted_dataloader, return_embeddings=False)
        return float(results["mean"][0])
        
    except Exception as e:
        logger.warning(f"Failed to predict fitness for {smiles}: {e}")
        return -np.inf