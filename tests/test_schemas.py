"""Test data schemas and validation."""

import pytest
import pandas as pd
import numpy as np

from mol_active.schemas import (
    MoleculeRecord,
    DatasetSchema,
    FeatureMatrix,
    DataSplit,
    ModelPredictions,
    EvaluationMetrics,
    validate_molecules_dataframe,
    validate_feature_matrix,
    DataContractError
)


def test_molecule_record_valid():
    """Test valid molecule record."""
    record = MoleculeRecord(smiles="CCO", target=1.5)
    assert record.smiles == "CCO"
    assert record.target == 1.5


def test_molecule_record_invalid_smiles():
    """Test invalid SMILES."""
    with pytest.raises(ValueError):
        MoleculeRecord(smiles="", target=1.5)


def test_dataset_schema_valid():
    """Test valid dataset schema."""
    molecules = [
        MoleculeRecord(smiles="CCO", target=1.5),
        MoleculeRecord(smiles="CCC", target=2.0)
    ]
    dataset = DatasetSchema(molecules=molecules)
    assert len(dataset.molecules) == 2


def test_dataset_schema_empty():
    """Test empty dataset."""
    with pytest.raises(ValueError):
        DatasetSchema(molecules=[])


def test_feature_matrix_valid():
    """Test valid feature matrix."""
    features = [[1.0, 2.0], [3.0, 4.0]]
    targets = [1.5, 2.5]
    smiles = ["CCO", "CCC"]
    
    matrix = FeatureMatrix(features=features, targets=targets, smiles=smiles)
    assert len(matrix.features) == 2
    assert len(matrix.targets) == 2
    assert len(matrix.smiles) == 2


def test_feature_matrix_length_mismatch():
    """Test feature matrix with mismatched lengths."""
    with pytest.raises(ValueError):
        FeatureMatrix(
            features=[[1.0, 2.0]],
            targets=[1.5, 2.5],  # Different length
            smiles=["CCO"]
        )


def test_data_split_valid():
    """Test valid data split."""
    split = DataSplit(
        train_indices=[0, 1, 2],
        val_indices=[3, 4],
        test_indices=[5, 6]
    )
    assert len(split.train_indices) == 3
    assert len(split.val_indices) == 2
    assert len(split.test_indices) == 2


def test_data_split_overlapping():
    """Test data split with overlapping indices."""
    with pytest.raises(ValueError):
        DataSplit(
            train_indices=[0, 1, 2],
            val_indices=[2, 3],  # Overlap with train
            test_indices=[4, 5]
        )


def test_data_split_empty():
    """Test data split with empty splits."""
    with pytest.raises(ValueError):
        DataSplit(
            train_indices=[],  # Empty
            val_indices=[1, 2],
            test_indices=[3, 4]
        )


def test_model_predictions_valid():
    """Test valid model predictions."""
    predictions = ModelPredictions(
        predictions=[1.0, 2.0],
        uncertainties=[0.1, 0.2],
        targets=[1.1, 1.9],
        smiles=["CCO", "CCC"],
        split=["train", "test"]
    )
    assert len(predictions.predictions) == 2


def test_model_predictions_invalid_split():
    """Test model predictions with invalid split values."""
    with pytest.raises(ValueError):
        ModelPredictions(
            predictions=[1.0, 2.0],
            uncertainties=[0.1, 0.2],
            smiles=["CCO", "CCC"],
            split=["train", "invalid"]  # Invalid split value
        )


def test_evaluation_metrics_valid():
    """Test valid evaluation metrics."""
    metrics = EvaluationMetrics(
        rmse=0.5,
        mae=0.3,
        r2=0.8,
        nll=1.2,
        ece=0.05,
        coverage=0.95
    )
    assert metrics.rmse == 0.5
    assert metrics.r2 == 0.8


def test_evaluation_metrics_invalid():
    """Test invalid evaluation metrics."""
    with pytest.raises(ValueError):
        EvaluationMetrics(
            rmse=-0.5,  # Should be >= 0
            mae=0.3,
            r2=0.8
        )


def test_validate_molecules_dataframe_valid():
    """Test valid molecules DataFrame validation."""
    df = pd.DataFrame({
        "smiles": ["CCO", "CCC", "CCCC"],
        "target": [1.0, 2.0, 3.0]
    })
    
    # Should not raise
    validate_molecules_dataframe(df, "smiles", "target")


def test_validate_molecules_dataframe_missing_column():
    """Test DataFrame validation with missing column."""
    df = pd.DataFrame({
        "smiles": ["CCO", "CCC"],
        # Missing target column
    })
    
    with pytest.raises(DataContractError):
        validate_molecules_dataframe(df, "smiles", "target")


def test_validate_molecules_dataframe_empty_smiles():
    """Test DataFrame validation with empty SMILES."""
    df = pd.DataFrame({
        "smiles": ["CCO", "", "CCC"],  # Empty SMILES
        "target": [1.0, 2.0, 3.0]
    })
    
    with pytest.raises(DataContractError):
        validate_molecules_dataframe(df, "smiles", "target")


def test_validate_molecules_dataframe_nan_values():
    """Test DataFrame validation with NaN values."""
    df = pd.DataFrame({
        "smiles": ["CCO", np.nan, "CCC"],  # NaN SMILES
        "target": [1.0, 2.0, 3.0]
    })
    
    with pytest.raises(DataContractError):
        validate_molecules_dataframe(df, "smiles", "target")


def test_validate_feature_matrix_valid():
    """Test valid feature matrix validation."""
    features = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0],
        "feature2": [0.5, 1.5, 2.5]
    })
    targets = pd.Series([1.0, 2.0, 3.0])
    
    # Should not raise
    validate_feature_matrix(features, targets)


def test_validate_feature_matrix_length_mismatch():
    """Test feature matrix validation with length mismatch."""
    features = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0],
        "feature2": [0.5, 1.5, 2.5]
    })
    targets = pd.Series([1.0, 2.0])  # Different length
    
    with pytest.raises(DataContractError):
        validate_feature_matrix(features, targets)


def test_validate_feature_matrix_nan_features():
    """Test feature matrix validation with NaN in features."""
    features = pd.DataFrame({
        "feature1": [1.0, np.nan, 3.0],  # NaN in features
        "feature2": [0.5, 1.5, 2.5]
    })
    targets = pd.Series([1.0, 2.0, 3.0])
    
    with pytest.raises(DataContractError):
        validate_feature_matrix(features, targets)