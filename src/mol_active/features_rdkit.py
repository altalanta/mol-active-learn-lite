"""RDKit-based molecular feature extraction."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pathlib import Path

from .logging import logger
from .schemas import validate_feature_matrix, DataContractError


def compute_morgan_fingerprints(
    smiles_list: List[str],
    radius: int = 2,
    n_bits: int = 1024,
    use_features: bool = False
) -> np.ndarray:
    """Compute Morgan (ECFP) fingerprints.
    
    Args:
        smiles_list: List of SMILES strings
        radius: Fingerprint radius (ECFP4 = radius 2)
        n_bits: Number of bits in fingerprint
        use_features: Whether to use feature-based fingerprints
        
    Returns:
        Fingerprint matrix (n_molecules x n_bits)
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        raise DataContractError(
            "RDKit required for fingerprint computation. Install with: pip install mol-active-learn-lite[rdkit]"
        )
    
    fingerprints = []
    
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                # Use zero vector for invalid molecules
                fp = np.zeros(n_bits, dtype=np.int8)
            else:
                if use_features:
                    fp = AllChem.GetMorganFingerprintAsBitVect(
                        mol, radius, nBits=n_bits, useFeatures=True
                    )
                else:
                    fp = AllChem.GetMorganFingerprintAsBitVect(
                        mol, radius, nBits=n_bits
                    )
                fp = np.array(fp, dtype=np.int8)
            
            fingerprints.append(fp)
            
        except Exception:
            # Use zero vector for problematic molecules
            fingerprints.append(np.zeros(n_bits, dtype=np.int8))
    
    fingerprint_matrix = np.vstack(fingerprints)
    logger.info(f"Computed Morgan fingerprints: {fingerprint_matrix.shape}")
    
    return fingerprint_matrix


def compute_rdkit_descriptors(smiles_list: List[str]) -> pd.DataFrame:
    """Compute RDKit molecular descriptors.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        DataFrame with molecular descriptors
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Crippen, Lipinski, QED
    except ImportError:
        raise DataContractError(
            "RDKit required for descriptor computation. Install with: pip install mol-active-learn-lite[rdkit]"
        )
    
    descriptor_data = []
    
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                # Use NaN for invalid molecules
                desc_dict = {name: np.nan for name in _get_descriptor_names()}
            else:
                desc_dict = {
                    # Basic molecular properties
                    "MolWt": Descriptors.MolWt(mol),
                    "NumHeavyAtoms": mol.GetNumHeavyAtoms(),
                    "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
                    "NumAromaticRings": Descriptors.NumAromaticRings(mol),
                    "NumAliphaticRings": Descriptors.NumAliphaticRings(mol),
                    
                    # Lipophilicity and solubility
                    "LogP": Crippen.MolLogP(mol),
                    "TPSA": Descriptors.TPSA(mol),
                    
                    # Drug-likeness
                    "NumHBD": Descriptors.NumHBD(mol),
                    "NumHBA": Descriptors.NumHBA(mol),
                    "QED": QED.qed(mol),
                    
                    # Lipinski descriptors
                    "MolWt_Lipinski": Descriptors.MolWt(mol),
                    "NumViolations": (
                        (Descriptors.MolWt(mol) > 500) +
                        (Crippen.MolLogP(mol) > 5) +
                        (Descriptors.NumHBD(mol) > 5) +
                        (Descriptors.NumHBA(mol) > 10)
                    ),
                    
                    # Additional descriptors
                    "FractionCsp3": Descriptors.FractionCsp3(mol),
                    "BertzCT": Descriptors.BertzCT(mol),
                    "NumSaturatedRings": Descriptors.NumSaturatedRings(mol),
                    "NumHeterocycles": Descriptors.NumHeterocycles(mol),
                    "RingCount": Descriptors.RingCount(mol),
                }
            
            descriptor_data.append(desc_dict)
            
        except Exception:
            # Use NaN for problematic molecules
            desc_dict = {name: np.nan for name in _get_descriptor_names()}
            descriptor_data.append(desc_dict)
    
    df = pd.DataFrame(descriptor_data)
    
    # Handle NaN values by filling with median values
    for col in df.columns:
        if df[col].isna().any():
            median_val = df[col].median()
            if pd.isna(median_val):
                # If all values are NaN, use 0
                median_val = 0.0
            df[col] = df[col].fillna(median_val)
    
    logger.info(f"Computed RDKit descriptors: {df.shape}")
    return df


def _get_descriptor_names() -> List[str]:
    """Get list of descriptor names."""
    return [
        "MolWt", "NumHeavyAtoms", "NumRotatableBonds", "NumAromaticRings",
        "NumAliphaticRings", "LogP", "TPSA", "NumHBD", "NumHBA", "QED",
        "MolWt_Lipinski", "NumViolations", "FractionCsp3", "BertzCT",
        "NumSaturatedRings", "NumHeterocycles", "RingCount"
    ]


def compute_combined_features(
    smiles_list: List[str],
    use_fingerprints: bool = True,
    use_descriptors: bool = True,
    fp_radius: int = 2,
    fp_n_bits: int = 1024,
    fp_use_features: bool = False
) -> pd.DataFrame:
    """Compute combined molecular features.
    
    Args:
        smiles_list: List of SMILES strings
        use_fingerprints: Whether to include Morgan fingerprints
        use_descriptors: Whether to include RDKit descriptors
        fp_radius: Fingerprint radius
        fp_n_bits: Number of fingerprint bits
        fp_use_features: Whether to use feature-based fingerprints
        
    Returns:
        Combined feature DataFrame
    """
    if not use_fingerprints and not use_descriptors:
        raise ValueError("Must use at least one feature type")
    
    features_list = []
    
    if use_descriptors:
        descriptors = compute_rdkit_descriptors(smiles_list)
        features_list.append(descriptors)
        logger.info(f"Added {descriptors.shape[1]} RDKit descriptors")
    
    if use_fingerprints:
        fingerprints = compute_morgan_fingerprints(
            smiles_list, fp_radius, fp_n_bits, fp_use_features
        )
        fp_df = pd.DataFrame(
            fingerprints,
            columns=[f"fp_{i}" for i in range(fingerprints.shape[1])]
        )
        features_list.append(fp_df)
        logger.info(f"Added {fp_df.shape[1]} Morgan fingerprint bits")
    
    # Combine features
    combined_features = pd.concat(features_list, axis=1)
    combined_features.index = range(len(smiles_list))
    
    logger.info(f"Combined features shape: {combined_features.shape}")
    return combined_features


def featurize_molecules(
    df: pd.DataFrame,
    smiles_col: str,
    target_col: str,
    feature_config: Optional[dict] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """Featurize molecules from DataFrame.
    
    Args:
        df: DataFrame with molecules
        smiles_col: SMILES column name
        target_col: Target column name
        feature_config: Feature configuration dictionary
        
    Returns:
        Tuple of (features DataFrame, targets Series)
    """
    if feature_config is None:
        feature_config = {
            "use_fingerprints": True,
            "use_descriptors": True,
            "fp_radius": 2,
            "fp_n_bits": 1024,
            "fp_use_features": False
        }
    
    smiles_list = df[smiles_col].tolist()
    targets = df[target_col].copy()
    
    # Compute features
    features = compute_combined_features(
        smiles_list,
        use_fingerprints=feature_config.get("use_fingerprints", True),
        use_descriptors=feature_config.get("use_descriptors", True),
        fp_radius=feature_config.get("fp_radius", 2),
        fp_n_bits=feature_config.get("fp_n_bits", 1024),
        fp_use_features=feature_config.get("fp_use_features", False)
    )
    
    # Ensure indices match
    features.index = df.index
    targets.index = df.index
    
    # Validate
    validate_feature_matrix(features, targets)
    
    return features, targets


def save_features(
    features: pd.DataFrame,
    targets: pd.Series,
    smiles: pd.Series,
    output_path: Path
) -> None:
    """Save features and targets to Parquet file.
    
    Args:
        features: Feature matrix
        targets: Target values
        smiles: SMILES strings
        output_path: Output file path
    """
    # Combine into single DataFrame for saving
    combined = features.copy()
    combined["target"] = targets
    combined["smiles"] = smiles
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=True)
    
    logger.info(f"Saved features to {output_path}")


def load_features(input_path: Path) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load features and targets from Parquet file.
    
    Args:
        input_path: Input file path
        
    Returns:
        Tuple of (features, targets, smiles)
    """
    combined = pd.read_parquet(input_path)
    
    targets = combined["target"]
    smiles = combined["smiles"]
    features = combined.drop(["target", "smiles"], axis=1)
    
    validate_feature_matrix(features, targets)
    
    logger.info(f"Loaded features from {input_path}: {features.shape}")
    return features, targets, smiles