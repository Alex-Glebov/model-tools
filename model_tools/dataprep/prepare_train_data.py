#!/usr/bin/env python3
"""Prepare training data from first arg/trg file pair.

Creates X/Y arrays where:
- X: (n_samples, window_size, n_features) - sequence of feature windows
- Y: (n_samples,) - next value after each window (prediction target)
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    return logging.getLogger(__name__)


def prepare_data(
    arg_file: Path,
    trg_file: Path,
    window_size: int = 42,
    target_column: str = "peaks",
    step: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare training data from arg/trg file pair.

    Args:
        arg_file: Path to features feather file
        trg_file: Path to targets feather file
        window_size: Length of each sequence window
        target_column: Column to use as prediction target
        step: Step size for sliding window (1 = overlapping windows)

    Returns:
        X: (n_samples, window_size, n_features)
        Y: (n_samples,) - next value after each window
    """
    logger = logging.getLogger(__name__)

    # Load data
    logger.info(f"Loading features from: {arg_file}")
    arg_df = pd.read_feather(arg_file)

    logger.info(f"Loading targets from: {trg_file}")
    trg_df = pd.read_feather(trg_file)

    logger.info(f"Features shape: {arg_df.shape}, Targets shape: {trg_df.shape}")

    # Select numeric features
    numeric_cols = arg_df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove 'id' if it's just an identifier
    if 'id' in numeric_cols:
        numeric_cols.remove('id')

    logger.info(f"Using numeric features: {numeric_cols}")

    # Extract feature matrix and target vector
    X_raw = arg_df[numeric_cols].values  # (n_timesteps, n_features)
    Y_raw = trg_df[target_column].values  # (n_timesteps,)

    n_timesteps = len(X_raw)
    n_features = len(numeric_cols)

    logger.info(f"Raw data: {n_timesteps} timesteps, {n_features} features")

    # Create sliding windows
    X_windows = []
    Y_targets = []

    for start in range(0, n_timesteps - window_size, step):
        end = start + window_size

        # X is the window of features
        window = X_raw[start:end]  # (window_size, n_features)

        # Y is the NEXT value after the window (prediction target)
        if end < len(Y_raw):
            target = Y_raw[end]  # Single next value
            X_windows.append(window)
            Y_targets.append(target)

    X = np.array(X_windows)  # (n_samples, window_size, n_features)
    Y = np.array(Y_targets)  # (n_samples,)

    logger.info(f"Created {len(X)} training samples")
    logger.info(f"X shape: {X.shape}, Y shape: {Y.shape}")

    return X, Y, numeric_cols


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data from first arg/trg files"
    )
    parser.add_argument(
        "--arg-path",
        type=str,
        default="~/rawdata/csv/arg_final",
        help="Path to arg (features) directory"
    )
    parser.add_argument(
        "--trg-path",
        type=str,
        default="~/rawdata/csv/trg_final",
        help="Path to trg (targets) directory"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=42,
        help="Window size (default: 42 from chain analysis)"
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="peaks",
        help="Target column name (default: peaks)"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Step size for sliding window (default: 1)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="~/rawdata/",
        help="Output directory for X.npy and Y.npy"
    )

    args = parser.parse_args()
    logger = setup_logging()

    # Get first file from each directory
    arg_path = Path(args.arg_path).expanduser()
    trg_path = Path(args.trg_path).expanduser()

    arg_files = sorted(arg_path.glob("*.feather"))
    trg_files = sorted(trg_path.glob("*.feather"))

    if not arg_files:
        logger.error(f"No feather files found in {arg_path}")
        return
    if not trg_files:
        logger.error(f"No feather files found in {trg_path}")
        return

    # Take first matching pair
    arg_file = arg_files[0]
    trg_file = trg_files[0]

    logger.info(f"Using arg file: {arg_file.name}")
    logger.info(f"Using trg file: {trg_file.name}")

    # Prepare data
    X, Y, feature_cols = prepare_data(
        arg_file=arg_file,
        trg_file=trg_file,
        window_size=args.window_size,
        target_column=args.target_column,
        step=args.step
    )

    # Save output
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as arg-final and trg-final folders (for compatibility with train.py)
    x_dir = output_dir / "arg-final"
    y_dir = output_dir / "trg-final"
    x_dir.mkdir(exist_ok=True)
    y_dir.mkdir(exist_ok=True)

    # Use same naming convention as source files
    pair_name = arg_file.stem.replace("-arg-final", "")
    x_file = x_dir / f"{pair_name}.npy"
    y_file = y_dir / f"{pair_name}.npy"

    np.save(x_file, X)
    np.save(y_file, Y)

    logger.info(f"Saved X to: {x_file}")
    logger.info(f"Saved Y to: {y_file}")

    # Show sample data
    logger.info("\n=== Sample Data ===")
    logger.info(f"First X window shape: {X[0].shape}")
    logger.info(f"First X window:\n{X[0]}")
    logger.info(f"First Y target (next value): {Y[0]}")
    logger.info(f"Last Y target: {Y[-1]}")
    logger.info(f"Y unique values: {np.unique(Y)}")


if __name__ == "__main__":
    main()
