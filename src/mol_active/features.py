"""Molecular feature computation using RDKit."""

import logging
import numpy as np
from typing import Dict, List, Optional, Union

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class MolecularFeaturizer:
    """Molecular featurizer using RDKit descriptors and fingerprints."""

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.feature_type = config.get("feature_type", "combined")
        self.descriptor_names = config.get("descriptor_names", [])
        self.ecfp_radius = config.get("ecfp_radius", 3)
        self.ecfp_bits = config.get("ecfp_bits", 1024)
        self.scaler: Optional[StandardScaler] = None

    def fit(self, smiles_list: List[str]) -> None:
        """Fit the featurizer (mainly for standardization)."""
        features = self._compute_features(smiles_list)
        
        # Fit scaler
        self.scaler = StandardScaler()
        self.scaler.fit(features)
        
        logger.info(f"Fitted featurizer on {len(smiles_list)} molecules")
        logger.info(f"Feature dimension: {features.shape[1]}")

    def transform(self, smiles_list: List[str]) -> np.ndarray:
        """Transform SMILES to features."""
        features = self._compute_features(smiles_list)
        
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        return features

    def fit_transform(self, smiles_list: List[str]) -> np.ndarray:
        """Fit and transform SMILES to features."""
        self.fit(smiles_list)
        return self.transform(smiles_list)

    def _compute_features(self, smiles_list: List[str]) -> np.ndarray:
        """Compute molecular features for a list of SMILES."""
        if self.feature_type == "descriptors":
            return self._compute_descriptors(smiles_list)
        elif self.feature_type == "ecfp":
            return self._compute_ecfp(smiles_list)
        elif self.feature_type == "combined":
            desc_features = self._compute_descriptors(smiles_list)
            ecfp_features = self._compute_ecfp(smiles_list)
            return np.hstack([desc_features, ecfp_features])
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")

    def _compute_descriptors(self, smiles_list: List[str]) -> np.ndarray:
        """Compute RDKit molecular descriptors."""
        descriptor_functions = {
            "MolWt": Descriptors.MolWt,
            "LogP": Crippen.MolLogP,
            "NumHDonors": Lipinski.NumHDonors,
            "NumHAcceptors": Lipinski.NumHAcceptors,
            "TPSA": rdMolDescriptors.CalcTPSA,
            "NumRotatableBonds": Descriptors.NumRotatableBonds,
            "NumAromaticRings": Descriptors.NumAromaticRings,
            "NumAliphaticRings": Descriptors.NumAliphaticRings,
            "RingCount": Descriptors.RingCount,
            "FractionCsp3": rdMolDescriptors.CalcFractionCsp3,
            "HeavyAtomCount": Descriptors.HeavyAtomCount,
            "NumHeteroatoms": Descriptors.NumHeteroatoms,
            "BalabanJ": Descriptors.BalabanJ,
            "BertzCT": Descriptors.BertzCT,
            "Chi0": rdMolDescriptors.CalcChi0v,
            "Chi1": rdMolDescriptors.CalcChi1v,
            "Kappa1": rdMolDescriptors.CalcKappa1,
            "Kappa2": rdMolDescriptors.CalcKappa2,
            "Kappa3": rdMolDescriptors.CalcKappa3,
            "LabuteASA": rdMolDescriptors.CalcLabuteASA,
        }

        features = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                # Handle invalid SMILES with NaN values
                mol_features = [np.nan] * len(self.descriptor_names)
            else:
                mol_features = []
                for desc_name in self.descriptor_names:
                    if desc_name in descriptor_functions:
                        try:
                            value = descriptor_functions[desc_name](mol)
                            mol_features.append(value)
                        except:
                            mol_features.append(np.nan)
                    else:
                        logger.warning(f"Unknown descriptor: {desc_name}")
                        mol_features.append(np.nan)
            
            features.append(mol_features)

        features = np.array(features, dtype=np.float32)
        
        # Handle NaN values by replacing with column means
        col_means = np.nanmean(features, axis=0)
        nan_mask = np.isnan(features)
        features[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        return features

    def _compute_ecfp(self, smiles_list: List[str]) -> np.ndarray:
        """Compute Extended Connectivity Fingerprints (ECFP)."""
        from rdkit.Chem import rdMolDescriptors

        features = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                # Handle invalid SMILES with zero vector
                fp = np.zeros(self.ecfp_bits, dtype=np.float32)
            else:
                try:
                    # Compute Morgan fingerprint (circular/ECFP)
                    fp_obj = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                        mol, self.ecfp_radius, nBits=self.ecfp_bits
                    )
                    fp = np.array(fp_obj, dtype=np.float32)
                except:
                    fp = np.zeros(self.ecfp_bits, dtype=np.float32)
            
            features.append(fp)

        return np.array(features, dtype=np.float32)

    def get_feature_dim(self) -> int:
        """Get the dimension of computed features."""
        if self.feature_type == "descriptors":
            return len(self.descriptor_names)
        elif self.feature_type == "ecfp":
            return self.ecfp_bits
        elif self.feature_type == "combined":
            return len(self.descriptor_names) + self.ecfp_bits
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")

    def save(self, path: Union[str, Path]) -> None:
        """Save the fitted featurizer."""
        if self.scaler is None:
            raise ValueError("Featurizer must be fitted before saving")
        
        save_dict = {
            "config": self.config,
            "scaler": self.scaler,
        }
        joblib.dump(save_dict, path)
        logger.info(f"Saved featurizer to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MolecularFeaturizer":
        """Load a fitted featurizer."""
        save_dict = joblib.load(path)
        
        featurizer = cls(save_dict["config"])
        featurizer.scaler = save_dict["scaler"]
        
        logger.info(f"Loaded featurizer from {path}")
        return featurizer


def compute_molecular_features(smiles_list: List[str], config: Dict) -> np.ndarray:
    """Compute molecular features for a list of SMILES strings."""
    featurizer = MolecularFeaturizer(config)
    return featurizer.fit_transform(smiles_list)


def compute_molecular_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    """Compute a comprehensive set of molecular descriptors for a single molecule."""
    if mol is None:
        return {}

    descriptors = {
        # Basic properties
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "NumHDonors": Lipinski.NumHDonors(mol),
        "NumHAcceptors": Lipinski.NumHAcceptors(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        
        # Structural features
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "NumAromaticRings": Descriptors.NumAromaticRings(mol),
        "NumAliphaticRings": Descriptors.NumAliphaticRings(mol),
        "RingCount": Descriptors.RingCount(mol),
        "FractionCsp3": rdMolDescriptors.CalcFractionCsp3(mol),
        
        # Complexity metrics
        "BertzCT": Descriptors.BertzCT(mol),
        "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
        "NumHeteroatoms": Descriptors.NumHeteroatoms(mol),
        
        # Topological indices
        "BalabanJ": Descriptors.BalabanJ(mol),
        "Chi0": rdMolDescriptors.CalcChi0v(mol),
        "Chi1": rdMolDescriptors.CalcChi1v(mol),
        "Kappa1": rdMolDescriptors.CalcKappa1(mol),
        "Kappa2": rdMolDescriptors.CalcKappa2(mol),
        "Kappa3": rdMolDescriptors.CalcKappa3(mol),
        
        # Surface area
        "LabuteASA": rdMolDescriptors.CalcLabuteASA(mol),
    }

    return descriptors


def validate_smiles(smiles: str) -> bool:
    """Validate a SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize a SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol)
    except:
        pass
    return None