"""Unified command-line interface for mol-active-learn-lite."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer

from . import __version__
from .logging import configure_logging, logger
from .utils import load_config, set_global_seed

app = typer.Typer(help="Molecular Active Learning Lite - Probabilistic property prediction and Bayesian optimization")

def _emit_json(payload: dict[str, Any]) -> None:
    """Emit JSON output to stdout."""
    typer.echo(json.dumps(payload, indent=2))


def _validate_extras(extras: list[str]) -> None:
    """Check if required extras are installed."""
    missing = []
    
    if "rdkit" in extras:
        try:
            import rdkit  # noqa: F401
        except ImportError:
            missing.append("rdkit")
    
    if "al" in extras:
        try:
            import botorch  # noqa: F401
            import gpytorch  # noqa: F401
        except ImportError:
            missing.append("al")
    
    if "viz" in extras:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            missing.append("viz")
    
    if missing:
        typer.secho(
            f"Missing required extras: {missing}. Install with: pip install mol-active-learn-lite[{','.join(missing)}]",
            fg=typer.colors.RED,
            err=True
        )
        raise typer.Exit(1)


@app.command()
def download(
    data_config: Path = typer.Option(..., "--data-config", help="Path to data configuration file"),
    out: Path = typer.Option("data", "--out", help="Output directory for downloaded data"),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
) -> None:
    """Download and preprocess ESOL dataset."""
    set_global_seed(seed)
    configure_logging()
    
    # Import here to allow graceful fallback if rdkit not installed
    try:
        from .data import download_and_process_data
    except ImportError as e:
        typer.secho(f"RDKit required for data processing. Install with: pip install mol-active-learn-lite[rdkit]", fg=typer.colors.RED, err=True)
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    
    config = load_config(data_config)
    result = download_and_process_data(config, out)
    
    payload = {
        "dataset": config.get("name", "esol"),
        "output_dir": str(out),
        "num_molecules": result["num_molecules"],
        "processed_file": str(result["processed_file"]),
        "splits": result["splits"],
    }
    _emit_json(payload)


@app.command()
def train(
    data_config: Path = typer.Option(..., "--data-config", help="Path to data configuration file"),
    model_config: Path = typer.Option(..., "--model-config", help="Path to model configuration file"),
    train_config: Path = typer.Option(..., "--train-config", help="Path to training configuration file"),
    out: Path = typer.Option("artifacts", "--out", help="Output directory for training artifacts"),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    device: str = typer.Option("auto", "--device", help="Device selection (auto|cpu|cuda)"),
) -> None:
    """Train ensemble or MC-dropout model."""
    set_global_seed(seed)
    configure_logging()
    
    _validate_extras(["rdkit"])
    
    from .training import train_model
    
    # Load configurations
    data_cfg = load_config(data_config)
    model_cfg = load_config(model_config)
    train_cfg = load_config(train_config)
    
    # Override config with CLI args
    if seed is not None:
        train_cfg["seed"] = seed
        data_cfg["seed"] = seed
    
    result = train_model(data_cfg, model_cfg, train_cfg, out, device)
    
    payload = {
        "run_id": result["run_id"],
        "model_dir": str(result["model_dir"]),
        "train_metrics": result["train_metrics"],
        "val_metrics": result["val_metrics"],
        "uncertainty_method": model_cfg.get("uncertainty_method", "ensemble"),
        "artifacts": {
            "model_file": str(result["model_file"]),
            "metrics_file": str(result["metrics_file"]),
            "config_file": str(result["config_file"]),
        }
    }
    _emit_json(payload)


@app.command()
def evaluate(
    model_config: Path = typer.Option(..., "--model-config", help="Path to model configuration file"),
    data_config: Path = typer.Option(..., "--data-config", help="Path to data configuration file"),
    model_dir: Path = typer.Option(..., "--model-dir", help="Directory containing trained model"),
    out: Path = typer.Option("evaluation", "--out", help="Output directory for evaluation results"),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    device: str = typer.Option("auto", "--device", help="Device selection (auto|cpu|cuda)"),
) -> None:
    """Evaluate model uncertainty and calibration."""
    set_global_seed(seed)
    configure_logging()
    
    _validate_extras(["rdkit", "viz"])
    
    from .evaluation import evaluate_model
    
    data_cfg = load_config(data_config)
    model_cfg = load_config(model_config)
    
    result = evaluate_model(model_dir, data_cfg, model_cfg, out, device)
    
    payload = {
        "model_dir": str(model_dir),
        "output_dir": str(out),
        "test_metrics": result["test_metrics"],
        "calibration": result["calibration"],
        "uncertainty_stats": result["uncertainty_stats"],
        "artifacts": {
            "metrics_file": str(result["metrics_file"]),
            "predictions_file": str(result["predictions_file"]),
            "plots_dir": str(result["plots_dir"]),
        }
    }
    _emit_json(payload)


@app.command("active-learn")
def active_learn(
    data_config: Path = typer.Option(..., "--data-config", help="Path to data configuration file"),
    model_config: Path = typer.Option(..., "--model-config", help="Path to model configuration file"),
    al_config: Path = typer.Option(..., "--al-config", help="Path to active learning configuration file"),
    out: Path = typer.Option("active_learning", "--out", help="Output directory for AL results"),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    device: str = typer.Option("auto", "--device", help="Device selection (auto|cpu|cuda)"),
) -> None:
    """Run active learning experiment."""
    set_global_seed(seed)
    configure_logging()
    
    _validate_extras(["rdkit", "al", "viz"])
    
    from .active_learning import run_active_learning
    
    data_cfg = load_config(data_config)
    model_cfg = load_config(model_config)
    al_cfg = load_config(al_config)
    
    # Override with CLI args
    if seed is not None:
        data_cfg["seed"] = seed
        model_cfg["seed"] = seed
        al_cfg["seed"] = seed
    
    result = run_active_learning(data_cfg, model_cfg, al_cfg, out, device)
    
    payload = {
        "experiment_id": result["experiment_id"],
        "output_dir": str(out),
        "total_iterations": result["total_iterations"],
        "final_performance": result["final_performance"],
        "sample_efficiency": result["sample_efficiency"],
        "artifacts": {
            "learning_curves_file": str(result["learning_curves_file"]),
            "selections_file": str(result["selections_file"]),
            "final_model_dir": str(result["final_model_dir"]),
        }
    }
    _emit_json(payload)


@app.command()
def propose(
    model_dir: Path = typer.Option(..., "--model-dir", help="Directory containing trained model"),
    model_config: Path = typer.Option(..., "--model-config", help="Path to model configuration file"),
    out: Path = typer.Option("proposals", "--out", help="Output directory for generated molecules"),
    num_candidates: int = typer.Option(100, "--num-candidates", help="Number of candidates to generate"),
    target_property: str = typer.Option("maximize", "--target-property", help="Target property optimization (maximize|minimize)"),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    device: str = typer.Option("auto", "--device", help="Device selection (auto|cpu|cuda)"),
) -> None:
    """Generate novel molecules using genetic algorithm."""
    set_global_seed(seed)
    configure_logging()
    
    _validate_extras(["rdkit"])
    
    from .proposer import generate_molecules
    
    model_cfg = load_config(model_config)
    
    result = generate_molecules(
        model_dir, model_cfg, out, num_candidates, target_property, seed, device
    )
    
    payload = {
        "model_dir": str(model_dir),
        "output_dir": str(out),
        "num_generated": result["num_generated"],
        "best_score": result["best_score"], 
        "diversity_score": result["diversity_score"],
        "target_property": target_property,
        "artifacts": {
            "molecules_file": str(result["molecules_file"]),
            "scores_file": str(result["scores_file"]),
            "analysis_file": str(result["analysis_file"]),
        }
    }
    _emit_json(payload)


@app.callback()
def callback(
    version: bool = typer.Option(False, "--version", help="Show version and exit")
) -> None:
    """Molecular Active Learning Lite - Probabilistic property prediction and Bayesian optimization."""
    if version:
        typer.echo(f"mol-active-learn-lite {__version__}")
        raise typer.Exit()


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()