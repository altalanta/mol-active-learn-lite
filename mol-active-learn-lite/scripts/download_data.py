#!/usr/bin/env python3
"""Download and preprocess ESOL dataset."""

import argparse
import logging
from pathlib import Path
import sys

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mol_active.data import download_esol_data
from mol_active.utils import load_config, setup_logging


def main():
    """Main function for downloading data."""
    parser = argparse.ArgumentParser(description="Download and preprocess ESOL dataset")
    parser.add_argument(
        "--config",
        type=Path,
        default="configs/data/esol.yaml",
        help="Path to data configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="data",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if data exists"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Update output paths
        args.output_dir.mkdir(parents=True, exist_ok=True)
        config["cache_dir"] = str(args.output_dir / "cache")
        config["processed_file"] = str(args.output_dir / "esol_processed.csv")
        
        # Check if data already exists
        processed_file = Path(config["processed_file"])
        if processed_file.exists() and not args.force:
            logger.info(f"Processed data already exists at {processed_file}")
            logger.info("Use --force to re-download and reprocess")
            return
        
        # Download and process data
        logger.info("Starting data download and preprocessing...")
        download_esol_data(config)
        
        logger.info(f"Data successfully processed and saved to {processed_file}")
        
    except Exception as e:
        logger.error(f"Failed to download data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()