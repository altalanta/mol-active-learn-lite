# Molecular Active Learning Lite

A production-quality Python package for probabilistic molecular property prediction and Bayesian active learning for small-molecule design.

## Features

- **Probabilistic Regressors**: Train uncertainty-aware models using deep ensembles or MC-Dropout
- **Uncertainty Estimation**: Comprehensive uncertainty quantification with calibration metrics
- **Active Learning**: Bayesian optimization with BoTorch for efficient data selection
- **Molecular Features**: RDKit-based molecular descriptors and ECFP fingerprints
- **Property Optimization**: Genetic algorithm for molecular design and optimization
- **Production Ready**: Type hints, comprehensive tests, CI/CD, and containerization

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/mol-active-learn-lite.git
cd mol-active-learn-lite

# Install in development mode
pip install -e .

# Or install from PyPI (when published)
pip install mol-active-learn-lite
```

### Basic Usage

1. **Download and preprocess data**:
```bash
mol-download --config configs/data/esol.yaml --output-dir data
```

2. **Train a probabilistic model**:
```bash
mol-train \
    --data-config configs/data/esol.yaml \
    --model-config configs/model/ensemble_mlp.yaml \
    --train-config configs/train/default.yaml \
    --output-dir experiments/ensemble_run
```

3. **Evaluate model uncertainty**:
```bash
mol-evaluate \
    --model-dir experiments/ensemble_run \
    --output-dir evaluation \
    --plot
```

4. **Run active learning**:
```bash
mol-active-learn \
    --data-config configs/data/esol.yaml \
    --model-config configs/model/ensemble_mlp.yaml \
    --al-config configs/al/default.yaml \
    --output-dir experiments/active_learning
```

5. **Generate novel molecules**:
```bash
mol-propose \
    --model-dir experiments/ensemble_run \
    --output-dir generation \
    --num-candidates 20 \
    --target-property maximize
```

## Architecture

### Core Components

- **`mol_active.data`**: Data loading, preprocessing, and splitting
- **`mol_active.features`**: Molecular featurization with RDKit
- **`mol_active.models`**: Neural network models with uncertainty estimation
- **`mol_active.active_learning`**: Bayesian optimization and acquisition functions
- **`mol_active.proposer`**: Genetic algorithm for molecular generation
- **`mol_active.evaluation`**: Uncertainty evaluation and calibration metrics

### Uncertainty Methods

1. **Deep Ensembles**: Train multiple models with different random seeds
2. **MC-Dropout**: Use dropout at inference time for uncertainty estimation

### Active Learning

- **GP Surrogate**: Gaussian Process on penultimate layer embeddings
- **Acquisition Functions**: Expected Improvement (EI) and Upper Confidence Bound (UCB)
- **Baselines**: Random sampling and uncertainty sampling

## Configuration

All components are configured via YAML files:

### Data Configuration (`configs/data/esol.yaml`)
```yaml
url: "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
smiles_column: "smiles"
target_column: "measured log solubility in mols per litre"
split_type: "scaffold"  # or "random"
train_frac: 0.8
val_frac: 0.1
test_frac: 0.1
```

### Model Configuration (`configs/model/ensemble_mlp.yaml`)
```yaml
uncertainty_method: "ensemble"  # or "mc_dropout"
ensemble_size: 5
hidden_dims: [512, 256, 128]
dropout_rate: 0.2
activation: "relu"
feature_type: "combined"  # descriptors + ECFP
```

### Training Configuration (`configs/train/default.yaml`)
```yaml
num_epochs: 100
batch_size: 128
optimizer:
  name: "adam"
  lr: 0.001
  weight_decay: 1e-5
early_stopping_patience: 20
```

## Development

### Setup Development Environment

```bash
# Clone and install in development mode
git clone https://github.com/your-org/mol-active-learn-lite.git
cd mol-active-learn-lite
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mol_active --cov-report=html

# Run specific test categories
pytest tests/test_models.py
pytest tests/test_active_learning.py
```

### Code Quality

```bash
# Format code
black src/ tests/
ruff check src/ tests/

# Type checking
mypy src/
```

## API Reference

### Training Models

```python
from mol_active.data import ESolDataModule
from mol_active.models import DeepEnsemble
from mol_active.utils import load_config

# Load configuration
config = load_config("configs/model/ensemble_mlp.yaml")

# Setup data
datamodule = ESolDataModule(data_config, batch_size=128)
datamodule.setup()

# Train ensemble
model = DeepEnsemble(model_config, train_config, ensemble_size=5)
model.fit(datamodule, save_dir="models/")

# Make predictions with uncertainty
results = model.predict(test_dataloader, return_embeddings=True)
predictions = results["mean"]
uncertainties = results["std"]
```

### Active Learning

```python
from mol_active.active_learning import ActiveLearner

# Initialize active learner
learner = ActiveLearner(uncertainty_model, al_config)

# Select candidates
selected_indices = learner.select_candidates(
    pool_features=pool_features,
    pool_embeddings=pool_embeddings,
    labeled_embeddings=labeled_embeddings,
    labeled_targets=labeled_targets,
    batch_size=32
)
```

### Molecular Generation

```python
from mol_active.proposer import PropertyOptimizer

# Setup optimizer
optimizer = PropertyOptimizer(ga_config)

# Define fitness function
def fitness_function(smiles):
    return predict_property_with_model(model, smiles)

# Run optimization
final_population, fitness_scores = optimizer.optimize(
    initial_population=initial_smiles,
    fitness_function=fitness_function,
    target_property="maximize"
)
```

## Evaluation Metrics

### Uncertainty Metrics

- **Negative Log Likelihood (NLL)**: Proper scoring rule for probabilistic predictions
- **Expected Calibration Error (ECE)**: Measures calibration quality
- **Coverage**: Fraction of true values within prediction intervals
- **Reliability Diagrams**: Visual assessment of calibration

### Active Learning Metrics

- **Learning Curves**: Performance vs. number of labeled samples
- **Sample Efficiency**: Comparison with random sampling baselines
- **Diversity Metrics**: Chemical diversity of selected compounds

## Examples

See the `notebooks/` directory for detailed examples:

- `01_basic_training.ipynb`: Train and evaluate uncertainty models
- `02_active_learning.ipynb`: Active learning experiment
- `03_molecular_generation.ipynb`: Generate novel molecules
- `04_uncertainty_analysis.ipynb`: Comprehensive uncertainty evaluation

## Docker

```bash
# Build image
docker build -t mol-active-learn-lite .

# Run container
docker run -v $(pwd)/data:/app/data mol-active-learn-lite \
    mol-train --data-config configs/data/esol.yaml
```

## Performance

### Benchmarks

- **ESOL Dataset**: ~1,100 molecules, log solubility prediction
- **Training Time**: ~2-5 minutes for ensemble (5 models) on CPU
- **Inference**: ~1ms per molecule on CPU
- **Memory**: <2GB for full pipeline

### Scaling

- Supports distributed training with PyTorch Lightning
- GPU acceleration for larger datasets
- Batch prediction for efficient inference

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run the test suite (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write comprehensive tests for new features
- Update documentation for API changes
- Use conventional commit messages

## Citation

If you use this software in your research, please cite:

```bibtex
@software{mol_active_learn_lite,
  title={Molecular Active Learning Lite: Probabilistic Property Prediction and Bayesian Optimization},
  author={ML Engineering Team},
  year={2024},
  url={https://github.com/your-org/mol-active-learn-lite}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [RDKit](https://www.rdkit.org/) for molecular informatics
- [PyTorch Lightning](https://lightning.ai/) for training infrastructure
- [BoTorch](https://botorch.org/) for Bayesian optimization
- [ESOL dataset](https://pubs.acs.org/doi/10.1021/ci034243x) for solubility data

## Support

- **Documentation**: [docs.example.com](https://docs.example.com)
- **Issues**: [GitHub Issues](https://github.com/your-org/mol-active-learn-lite/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/mol-active-learn-lite/discussions)
- **Email**: support@example.com