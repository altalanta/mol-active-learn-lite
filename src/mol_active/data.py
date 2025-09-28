"""Data loading and processing for ESOL dataset."""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve

import lightning as L
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch

from .utils import set_seed

logger = logging.getLogger(__name__)


class MolecularDataset(Dataset):
    """PyTorch dataset for molecular property prediction."""

    def __init__(
        self,
        smiles: List[str],
        targets: np.ndarray,
        features: np.ndarray,
        indices: Optional[np.ndarray] = None,
    ) -> None:
        self.smiles = smiles
        self.targets = torch.FloatTensor(targets)
        self.features = torch.FloatTensor(features)
        self.indices = indices if indices is not None else np.arange(len(smiles))

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "features": self.features[idx],
            "targets": self.targets[idx],
            "index": torch.tensor(self.indices[idx], dtype=torch.long),
        }


class ESolDataModule(L.LightningDataModule):
    """Lightning data module for ESOL dataset."""

    def __init__(
        self,
        data_config: Dict,
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.data_config = data_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dataset: Optional[MolecularDataset] = None
        self.val_dataset: Optional[MolecularDataset] = None
        self.test_dataset: Optional[MolecularDataset] = None

    def prepare_data(self) -> None:
        """Download and prepare the ESOL dataset."""
        download_esol_data(self.data_config)

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup train/val/test datasets."""
        # Load processed data
        data_path = Path(self.data_config["processed_file"])
        if not data_path.exists():
            self.prepare_data()

        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} molecules from {data_path}")

        # Split data
        train_df, val_df, test_df = split_dataset(df, self.data_config)

        # Import features module here to avoid circular imports
        from .features import compute_molecular_features

        # Compute features for each split
        logger.info("Computing molecular features...")
        train_features = compute_molecular_features(
            train_df["smiles"].tolist(), self.data_config
        )
        val_features = compute_molecular_features(
            val_df["smiles"].tolist(), self.data_config
        )
        test_features = compute_molecular_features(
            test_df["smiles"].tolist(), self.data_config
        )

        # Create datasets
        target_col = self.data_config["target_column"]
        self.train_dataset = MolecularDataset(
            train_df["smiles"].tolist(),
            train_df[target_col].values,
            train_features,
            train_df.index.values,
        )
        self.val_dataset = MolecularDataset(
            val_df["smiles"].tolist(),
            val_df[target_col].values,
            val_features,
            val_df.index.values,
        )
        self.test_dataset = MolecularDataset(
            test_df["smiles"].tolist(),
            test_df[target_col].values,
            test_features,
            test_df.index.values,
        )

        logger.info(
            f"Dataset splits: train={len(self.train_dataset)}, "
            f"val={len(self.val_dataset)}, test={len(self.test_dataset)}"
        )

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


def download_esol_data(config: Dict) -> None:
    """Download and preprocess ESOL dataset."""
    url = config["url"]
    cache_dir = Path(config["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    raw_file = cache_dir / "esol_raw.csv"
    processed_file = Path(config["processed_file"])

    # Download if not exists
    if not raw_file.exists():
        logger.info(f"Downloading ESOL data from {url}")
        urlretrieve(url, raw_file)

    # Process data
    logger.info("Processing ESOL data...")
    df = pd.read_csv(raw_file)

    # Clean and filter data
    df = clean_molecular_data(df, config)

    # Save processed data
    processed_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_file, index=False)
    logger.info(f"Saved processed data to {processed_file}")


def clean_molecular_data(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Clean and filter molecular data."""
    smiles_col = config["smiles_column"]
    target_col = config["target_column"]

    logger.info(f"Initial dataset size: {len(df)}")

    # Remove rows with missing SMILES or targets
    df = df.dropna(subset=[smiles_col, target_col])
    logger.info(f"After removing NaN: {len(df)}")

    # Canonicalize SMILES and remove invalid molecules
    valid_smiles = []
    valid_indices = []

    for idx, smiles in enumerate(df[smiles_col]):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Apply filters
            num_heavy = mol.GetNumHeavyAtoms()
            if (
                config.get("min_heavy_atoms", 0)
                <= num_heavy
                <= config.get("max_heavy_atoms", 1000)
            ):
                # Canonicalize
                if config.get("canonicalize_smiles", True):
                    canonical_smiles = Chem.MolToSmiles(mol)
                    valid_smiles.append(canonical_smiles)
                else:
                    valid_smiles.append(smiles)
                valid_indices.append(idx)

    # Filter dataframe
    df = df.iloc[valid_indices].copy()
    df[smiles_col] = valid_smiles

    logger.info(f"After SMILES validation and filtering: {len(df)}")

    # Remove duplicates
    df = df.drop_duplicates(subset=[smiles_col])
    logger.info(f"After deduplication: {len(df)}")

    return df


def split_dataset(
    df: pd.DataFrame, config: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train/val/test using scaffold or random splitting."""
    set_seed(config.get("data_seed", 42))

    split_type = config.get("split_type", "scaffold")
    train_frac = config["train_frac"]
    val_frac = config["val_frac"]
    test_frac = config["test_frac"]

    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    if split_type == "scaffold":
        return scaffold_split(df, config, train_frac, val_frac, test_frac)
    else:
        return random_split(df, config, train_frac, val_frac, test_frac)


def random_split(
    df: pd.DataFrame, config: Dict, train_frac: float, val_frac: float, test_frac: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Random split of dataset."""
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_frac,
        random_state=config.get("data_seed", 42),
        shuffle=True,
    )

    # Second split: train vs val
    val_size_adjusted = val_frac / (train_frac + val_frac)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        random_state=config.get("data_seed", 42),
        shuffle=True,
    )

    return train_df, val_df, test_df


def scaffold_split(
    df: pd.DataFrame, config: Dict, train_frac: float, val_frac: float, test_frac: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scaffold-based split to ensure chemical diversity."""
    smiles_col = config["smiles_column"]

    # Generate scaffolds
    scaffolds = {}
    for idx, smiles in enumerate(df[smiles_col]):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            if scaffold not in scaffolds:
                scaffolds[scaffold] = []
            scaffolds[scaffold].append(idx)

    # Sort scaffolds by size (largest first)
    scaffold_sets = list(scaffolds.values())
    scaffold_sets.sort(key=len, reverse=True)

    # Assign to splits
    train_indices, val_indices, test_indices = [], [], []
    train_size, val_size, test_size = 0, 0, 0
    total_size = len(df)

    for scaffold_set in scaffold_sets:
        if train_size / total_size < train_frac:
            train_indices.extend(scaffold_set)
            train_size += len(scaffold_set)
        elif val_size / total_size < val_frac:
            val_indices.extend(scaffold_set)
            val_size += len(scaffold_set)
        else:
            test_indices.extend(scaffold_set)
            test_size += len(scaffold_set)

    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)

    logger.info(
        f"Scaffold split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    return train_df, val_df, test_df


def create_active_learning_splits(
    df: pd.DataFrame, config: Dict, seed_frac: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create initial seed set and pool for active learning."""
    set_seed(config.get("data_seed", 42))

    # Use stratified sampling based on target quantiles
    target_col = config["target_column"]
    targets = df[target_col].values

    # Create quantile bins for stratification
    n_bins = min(10, len(df) // 20)  # At least 20 samples per bin
    quantiles = np.linspace(0, 1, n_bins + 1)
    target_bins = pd.cut(targets, bins=pd.Series(targets).quantile(quantiles))

    # Stratified split
    seed_df, pool_df = train_test_split(
        df,
        test_size=1 - seed_frac,
        stratify=target_bins,
        random_state=config.get("data_seed", 42),
    )

    logger.info(f"Active learning splits: seed={len(seed_df)}, pool={len(pool_df)}")

    return seed_df, pool_df