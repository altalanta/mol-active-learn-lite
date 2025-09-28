"""Data processing with deterministic splits and caching."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from .logging import logger
from .schemas import validate_molecules_dataframe, DataContractError, DataSplit
from .utils import set_global_seed


def download_data(url: str, cache_dir: Path, expected_checksum: Optional[str] = None) -> Path:
    """Download data with caching and checksum validation.
    
    Args:
        url: URL to download from
        cache_dir: Directory to cache downloaded files
        expected_checksum: Expected SHA256 checksum for validation
        
    Returns:
        Path to downloaded file
        
    Raises:
        DataContractError: If download or checksum validation fails
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename from URL
    filename = url.split("/")[-1]
    if not filename.endswith((".csv", ".tsv", ".txt")):
        filename = "downloaded_data.csv"
    
    cache_path = cache_dir / filename
    
    # Check if file exists and validate checksum
    if cache_path.exists():
        if expected_checksum:
            actual_checksum = _compute_checksum(cache_path)
            if actual_checksum == expected_checksum:
                logger.info(f"Using cached file: {cache_path}")
                return cache_path
            else:
                logger.warning(f"Checksum mismatch for cached file, re-downloading")
        else:
            logger.info(f"Using cached file: {cache_path}")
            return cache_path
    
    # Download file
    logger.info(f"Downloading data from {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(cache_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        # Validate checksum if provided
        if expected_checksum:
            actual_checksum = _compute_checksum(cache_path)
            if actual_checksum != expected_checksum:
                cache_path.unlink()  # Remove invalid file
                raise DataContractError(
                    f"Checksum validation failed. Expected: {expected_checksum}, "
                    f"got: {actual_checksum}"
                )
        
        logger.info(f"Downloaded and cached: {cache_path}")
        return cache_path
        
    except requests.RequestException as e:
        raise DataContractError(f"Failed to download data from {url}: {e}")


def _compute_checksum(file_path: Path) -> str:
    """Compute SHA256 checksum of file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def load_esol_data(data_config: Dict[str, Any]) -> pd.DataFrame:
    """Load and validate ESOL dataset.
    
    Args:
        data_config: Data configuration dictionary
        
    Returns:
        Validated DataFrame
    """
    url = data_config["url"]
    smiles_col = data_config["smiles_column"]
    target_col = data_config["target_column"]
    cache_dir = Path(data_config.get("cache_dir", "data/cache"))
    
    # Download data
    data_file = download_data(url, cache_dir)
    
    # Load and validate
    df = pd.read_csv(data_file)
    validate_molecules_dataframe(df, smiles_col, target_col)
    
    logger.info(f"Loaded {len(df)} molecules from ESOL dataset")
    return df


def clean_molecules(
    df: pd.DataFrame, 
    smiles_col: str,
    target_col: str,
    remove_salts: bool = True,
    canonicalize_smiles: bool = True,
    max_heavy_atoms: Optional[int] = None,
    min_heavy_atoms: Optional[int] = None
) -> pd.DataFrame:
    """Clean and filter molecular data.
    
    Args:
        df: Input DataFrame
        smiles_col: SMILES column name
        target_col: Target column name  
        remove_salts: Whether to remove salts from SMILES
        canonicalize_smiles: Whether to canonicalize SMILES
        max_heavy_atoms: Maximum number of heavy atoms
        min_heavy_atoms: Minimum number of heavy atoms
        
    Returns:
        Cleaned DataFrame
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import SaltRemover
    except ImportError:
        raise DataContractError(
            "RDKit required for molecule cleaning. Install with: pip install mol-active-learn-lite[rdkit]"
        )
    
    df = df.copy()
    initial_count = len(df)
    
    # Process SMILES
    valid_smiles = []
    valid_indices = []
    
    salt_remover = SaltRemover.SaltRemover() if remove_salts else None
    
    for idx, smiles in enumerate(df[smiles_col]):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            # Remove salts
            if salt_remover:
                mol = salt_remover.StripMol(mol)
            
            # Filter by heavy atom count
            if max_heavy_atoms is not None and mol.GetNumHeavyAtoms() > max_heavy_atoms:
                continue
            if min_heavy_atoms is not None and mol.GetNumHeavyAtoms() < min_heavy_atoms:
                continue
            
            # Canonicalize SMILES
            if canonicalize_smiles:
                clean_smiles = Chem.MolToSmiles(mol, canonical=True)
            else:
                clean_smiles = smiles
            
            valid_smiles.append(clean_smiles)
            valid_indices.append(idx)
            
        except Exception:
            # Skip invalid molecules
            continue
    
    # Filter DataFrame
    df_clean = df.iloc[valid_indices].copy()
    df_clean[smiles_col] = valid_smiles
    
    # Remove duplicates based on SMILES
    df_clean = df_clean.drop_duplicates(subset=[smiles_col])
    
    logger.info(
        f"Cleaned molecules: {initial_count} -> {len(df_clean)} "
        f"({len(df_clean)/initial_count*100:.1f}% retained)"
    )
    
    return df_clean


def scaffold_split(
    df: pd.DataFrame,
    smiles_col: str,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42
) -> DataSplit:
    """Perform scaffold-based splitting for molecules.
    
    Args:
        df: DataFrame with molecules
        smiles_col: SMILES column name
        train_frac: Training fraction
        val_frac: Validation fraction
        test_frac: Test fraction
        seed: Random seed
        
    Returns:
        DataSplit with train/val/test indices
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError:
        raise DataContractError(
            "RDKit required for scaffold splitting. Install with: pip install mol-active-learn-lite[rdkit]"
        )
    
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("Split fractions must sum to 1.0")
    
    set_global_seed(seed)
    np.random.seed(seed)
    
    # Compute Murcko scaffolds
    scaffolds = {}
    for idx, smiles in enumerate(df[smiles_col]):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                scaffold = smiles  # Fallback to original SMILES
            else:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            scaffold = smiles  # Fallback to original SMILES
        
        if scaffold not in scaffolds:
            scaffolds[scaffold] = []
        scaffolds[scaffold].append(idx)
    
    # Sort scaffolds by size (largest first) for more balanced splits
    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)
    
    # Assign scaffolds to splits
    train_indices, val_indices, test_indices = [], [], []
    train_size, val_size, test_size = 0, 0, 0
    total_size = len(df)
    
    for scaffold_indices in scaffold_sets:
        # Decide which split to add this scaffold to
        train_target = int(train_frac * total_size)
        val_target = int(val_frac * total_size)
        
        if train_size < train_target:
            train_indices.extend(scaffold_indices)
            train_size += len(scaffold_indices)
        elif val_size < val_target:
            val_indices.extend(scaffold_indices)
            val_size += len(scaffold_indices)
        else:
            test_indices.extend(scaffold_indices)
            test_size += len(scaffold_indices)
    
    # Shuffle within each split while maintaining determinism
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)
    
    split = DataSplit(
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices
    )
    
    logger.info(
        f"Scaffold split: train={len(train_indices)} ({len(train_indices)/total_size*100:.1f}%), "
        f"val={len(val_indices)} ({len(val_indices)/total_size*100:.1f}%), "
        f"test={len(test_indices)} ({len(test_indices)/total_size*100:.1f}%)"
    )
    
    return split


def random_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42
) -> DataSplit:
    """Perform random splitting.
    
    Args:
        df: DataFrame with molecules
        train_frac: Training fraction
        val_frac: Validation fraction
        test_frac: Test fraction
        seed: Random seed
        
    Returns:
        DataSplit with train/val/test indices
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("Split fractions must sum to 1.0")
    
    set_global_seed(seed)
    np.random.seed(seed)
    
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    
    train_end = int(train_frac * len(df))
    val_end = train_end + int(val_frac * len(df))
    
    train_indices = indices[:train_end].tolist()
    val_indices = indices[train_end:val_end].tolist()
    test_indices = indices[val_end:].tolist()
    
    split = DataSplit(
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices
    )
    
    logger.info(
        f"Random split: train={len(train_indices)} ({len(train_indices)/len(df)*100:.1f}%), "
        f"val={len(val_indices)} ({len(val_indices)/len(df)*100:.1f}%), "
        f"test={len(test_indices)} ({len(test_indices)/len(df)*100:.1f}%)"
    )
    
    return split


def save_split_indices(split: DataSplit, output_path: Path) -> None:
    """Save split indices to JSON file.
    
    Args:
        split: DataSplit object
        output_path: Path to save JSON file
    """
    split_dict = split.dict()
    with open(output_path, 'w') as f:
        json.dump(split_dict, f, indent=2)
    logger.info(f"Saved split indices to {output_path}")


def load_split_indices(split_path: Path) -> DataSplit:
    """Load split indices from JSON file.
    
    Args:
        split_path: Path to JSON file
        
    Returns:
        DataSplit object
    """
    with open(split_path, 'r') as f:
        split_dict = json.load(f)
    
    split = DataSplit(**split_dict)
    logger.info(f"Loaded split indices from {split_path}")
    return split


def download_and_process_data(data_config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """Download and process ESOL data with splits.
    
    Args:
        data_config: Data configuration
        output_dir: Output directory
        
    Returns:
        Processing results dictionary
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load raw data
    df = load_esol_data(data_config)
    
    # Clean molecules
    df_clean = clean_molecules(
        df,
        data_config["smiles_column"],
        data_config["target_column"],
        remove_salts=data_config.get("remove_salts", True),
        canonicalize_smiles=data_config.get("canonicalize_smiles", True),
        max_heavy_atoms=data_config.get("max_heavy_atoms"),
        min_heavy_atoms=data_config.get("min_heavy_atoms")
    )
    
    # Create splits
    split_type = data_config.get("split_type", "scaffold")
    seed = data_config.get("data_seed", 42)
    
    if split_type == "scaffold":
        split = scaffold_split(
            df_clean,
            data_config["smiles_column"],
            train_frac=data_config.get("train_frac", 0.7),
            val_frac=data_config.get("val_frac", 0.15),
            test_frac=data_config.get("test_frac", 0.15),
            seed=seed
        )
    elif split_type == "random":
        split = random_split(
            df_clean,
            train_frac=data_config.get("train_frac", 0.7),
            val_frac=data_config.get("val_frac", 0.15),
            test_frac=data_config.get("test_frac", 0.15),
            seed=seed
        )
    else:
        raise ValueError(f"Unknown split type: {split_type}")
    
    # Save processed data
    processed_file = output_dir / "processed_molecules.csv"
    df_clean.to_csv(processed_file, index=False)
    
    # Save split indices
    split_file = output_dir / "split_indices.json"
    save_split_indices(split, split_file)
    
    return {
        "num_molecules": len(df_clean),
        "processed_file": processed_file,
        "split_file": split_file,
        "splits": {
            "train": len(split.train_indices),
            "val": len(split.val_indices),
            "test": len(split.test_indices)
        }
    }