"""Data contracts and validation schemas using Pydantic."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from pydantic import BaseModel, Field, validator, root_validator
import pandas as pd


class DataContractError(Exception):
    """Raised when data fails schema validation."""


class MoleculeRecord(BaseModel):
    """Schema for a single molecule record."""
    smiles: str = Field(..., min_length=1, description="SMILES string")
    target: float = Field(..., description="Target property value")
    
    @validator("smiles")
    def validate_smiles(cls, v: str) -> str:
        """Validate SMILES string is not empty."""
        if not v.strip():
            raise ValueError("SMILES cannot be empty")
        return v.strip()


class DatasetSchema(BaseModel):
    """Schema for molecular dataset."""
    molecules: List[MoleculeRecord]
    
    @validator("molecules")
    def validate_molecules(cls, v: List[MoleculeRecord]) -> List[MoleculeRecord]:
        """Validate molecule list is not empty."""
        if len(v) == 0:
            raise ValueError("Dataset cannot be empty")
        return v


class FeatureMatrix(BaseModel):
    """Schema for featurized molecular data."""
    features: List[List[float]] = Field(..., description="Feature matrix")
    targets: List[float] = Field(..., description="Target values")
    smiles: List[str] = Field(..., description="SMILES strings")
    
    @root_validator
    def validate_lengths(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate all arrays have same length."""
        features = values.get("features", [])
        targets = values.get("targets", [])
        smiles = values.get("smiles", [])
        
        if not (len(features) == len(targets) == len(smiles)):
            raise ValueError("Features, targets, and SMILES must have same length")
        
        return values


class DataSplit(BaseModel):
    """Schema for train/val/test split indices."""
    train_indices: List[int] = Field(..., description="Training set indices")
    val_indices: List[int] = Field(..., description="Validation set indices")
    test_indices: List[int] = Field(..., description="Test set indices")
    
    @root_validator
    def validate_split(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate split indices are disjoint and non-empty."""
        train = set(values.get("train_indices", []))
        val = set(values.get("val_indices", []))
        test = set(values.get("test_indices", []))
        
        if not train or not val or not test:
            raise ValueError("All splits must be non-empty")
        
        # Check for overlaps
        if train & val:
            raise ValueError("Train and validation sets overlap")
        if train & test:
            raise ValueError("Train and test sets overlap")
        if val & test:
            raise ValueError("Validation and test sets overlap")
        
        return values


class ModelPredictions(BaseModel):
    """Schema for model predictions with uncertainty."""
    predictions: List[float] = Field(..., description="Mean predictions")
    uncertainties: List[float] = Field(..., description="Prediction uncertainties")
    targets: Optional[List[float]] = Field(None, description="True target values")
    smiles: List[str] = Field(..., description="SMILES strings")
    split: List[str] = Field(..., description="Data split (train/val/test)")
    
    @root_validator
    def validate_predictions(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate prediction arrays have consistent lengths."""
        predictions = values.get("predictions", [])
        uncertainties = values.get("uncertainties", [])
        smiles = values.get("smiles", [])
        split = values.get("split", [])
        targets = values.get("targets")
        
        lengths = [len(predictions), len(uncertainties), len(smiles), len(split)]
        if targets is not None:
            lengths.append(len(targets))
        
        if not all(l == lengths[0] for l in lengths):
            raise ValueError("All prediction arrays must have same length")
        
        # Validate split values
        valid_splits = {"train", "val", "test"}
        if not all(s in valid_splits for s in split):
            raise ValueError(f"Split values must be in {valid_splits}")
        
        return values


class EvaluationMetrics(BaseModel):
    """Schema for model evaluation metrics."""
    rmse: float = Field(..., ge=0, description="Root mean squared error")
    mae: float = Field(..., ge=0, description="Mean absolute error") 
    r2: float = Field(..., description="R-squared score")
    nll: Optional[float] = Field(None, description="Negative log likelihood")
    ece: Optional[float] = Field(None, ge=0, le=1, description="Expected calibration error")
    coverage: Optional[float] = Field(None, ge=0, le=1, description="Prediction interval coverage")


class CalibrationData(BaseModel):
    """Schema for calibration analysis data."""
    bin_boundaries: List[float] = Field(..., description="Calibration bin boundaries")
    bin_accuracies: List[float] = Field(..., description="Empirical accuracies per bin")
    bin_confidences: List[float] = Field(..., description="Mean confidences per bin")
    bin_counts: List[int] = Field(..., description="Number of samples per bin")
    
    @root_validator
    def validate_calibration(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate calibration data consistency."""
        boundaries = values.get("bin_boundaries", [])
        accuracies = values.get("bin_accuracies", [])
        confidences = values.get("bin_confidences", [])
        counts = values.get("bin_counts", [])
        
        # Number of bins should be consistent
        n_bins = len(boundaries) - 1  # boundaries includes edges
        if not (len(accuracies) == len(confidences) == len(counts) == n_bins):
            raise ValueError("Calibration arrays must have consistent lengths")
        
        return values


def validate_dataframe_schema(df: pd.DataFrame, required_columns: List[str]) -> None:
    """Validate DataFrame has required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Raises:
        DataContractError: If validation fails
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise DataContractError(f"Missing required columns: {missing_columns}")
    
    if df.empty:
        raise DataContractError("DataFrame cannot be empty")


def validate_molecules_dataframe(df: pd.DataFrame, smiles_col: str, target_col: str) -> None:
    """Validate molecular dataset DataFrame.
    
    Args:
        df: DataFrame containing molecular data
        smiles_col: Name of SMILES column
        target_col: Name of target column
        
    Raises:
        DataContractError: If validation fails
    """
    validate_dataframe_schema(df, [smiles_col, target_col])
    
    # Check for missing values
    if df[smiles_col].isna().any():
        raise DataContractError(f"Missing SMILES values in column '{smiles_col}'")
    
    if df[target_col].isna().any():
        raise DataContractError(f"Missing target values in column '{target_col}'")
    
    # Check SMILES are not empty strings
    empty_smiles = df[smiles_col].str.strip().eq("").any()
    if empty_smiles:
        raise DataContractError("Found empty SMILES strings")
    
    # Check targets are numeric
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise DataContractError(f"Target column '{target_col}' must be numeric")


def validate_feature_matrix(features: pd.DataFrame, targets: pd.Series) -> None:
    """Validate feature matrix and targets alignment.
    
    Args:
        features: Feature matrix DataFrame
        targets: Target values Series
        
    Raises:
        DataContractError: If validation fails
    """
    if features.empty:
        raise DataContractError("Feature matrix cannot be empty")
    
    if len(features) != len(targets):
        raise DataContractError("Features and targets must have same length")
    
    if not features.index.equals(targets.index):
        raise DataContractError("Features and targets must have same index")
    
    # Check for NaN values
    if features.isna().any().any():
        raise DataContractError("Feature matrix contains NaN values")
    
    if targets.isna().any():
        raise DataContractError("Target values contain NaN values")