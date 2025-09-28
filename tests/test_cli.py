"""Test CLI functionality."""

import json
from pathlib import Path
from typer.testing import CliRunner
import pytest

from mol_active.cli import app


def test_cli_help():
    """Test CLI help command."""
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Molecular Active Learning Lite" in result.stdout


def test_cli_version():
    """Test CLI version command."""
    runner = CliRunner()
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "mol-active-learn-lite" in result.stdout


def test_download_help():
    """Test download subcommand help."""
    runner = CliRunner()
    result = runner.invoke(app, ["download", "--help"])
    assert result.exit_code == 0
    assert "Download and preprocess ESOL dataset" in result.stdout


def test_train_help():
    """Test train subcommand help."""
    runner = CliRunner()
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "Train ensemble or MC-dropout model" in result.stdout


def test_evaluate_help():
    """Test evaluate subcommand help."""
    runner = CliRunner()
    result = runner.invoke(app, ["evaluate", "--help"])
    assert result.exit_code == 0
    assert "Evaluate model uncertainty and calibration" in result.stdout


def test_active_learn_help():
    """Test active-learn subcommand help."""
    runner = CliRunner()
    result = runner.invoke(app, ["active-learn", "--help"])
    assert result.exit_code == 0
    assert "Run active learning experiment" in result.stdout


def test_propose_help():
    """Test propose subcommand help."""
    runner = CliRunner()
    result = runner.invoke(app, ["propose", "--help"])
    assert result.exit_code == 0
    assert "Generate novel molecules using genetic algorithm" in result.stdout