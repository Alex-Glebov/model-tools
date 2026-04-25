#!/usr/bin/env python3
"""Test predictions on next date period for the same pair.

Usage:
    python test_prediction.py --model models/model_20260423_220531.keras --pair ADA_USDT --test-date 20201101
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from models import LSTMAttentionModel


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    return logging.getLogger(__name__)


def prepare_test_data(
    arg_file: Path,
    trg_file: Path,
    window_size: int = 42,
    target_column: str = "peaks",
    step: int = 1
):
    """Prepare test data from arg/trg file pair (same as training prep)."""
    arg_df = pd.read_feather(arg_file)
    trg_df = pd.read_feather(trg_file)

    numeric_cols = arg_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'id' in numeric_cols:
        numeric_cols.remove('id')

    X_raw = arg_df[numeric_cols].values
    Y_raw = trg_df[target_column].values

    n_timesteps = len(X_raw)
    n_features = len(numeric_cols)

    X_windows = []
    Y_targets = []
    timestamps = []

    for start in range(0, n_timesteps - window_size, step):
        end = start + window_size
        window = X_raw[start:end]
        if end < len(Y_raw):
            target = Y_raw[end]
            X_windows.append(window)
            Y_targets.append(target)
            timestamps.append(arg_df.index[end])

    X = np.array(X_windows)
    Y = np.array(Y_targets)

    return X, Y, timestamps, numeric_cols


def main():
    parser = argparse.ArgumentParser(
        description="Test predictions on next date period"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model .keras file"
    )
    parser.add_argument(
        "--pair",
        type=str,
        default="ADA_USDT",
        help="Currency pair to test on (default: ADA_USDT)"
    )
    parser.add_argument(
        "--test-date",
        type=str,
        default="20201101",
        help="Test date period (default: 20201101)"
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
        help="Window size (default: 42)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20,
        help="Max samples to display (default: 20)"
    )

    args = parser.parse_args()
    logger = setup_logging()

    # Build file paths
    arg_path = Path(args.arg_path).expanduser()
    trg_path = Path(args.trg_path).expanduser()

    arg_file = arg_path / f"{args.pair}-arg-final-{args.test_date}.feather"
    trg_file = trg_path / f"{args.pair}-trg-final-{args.test_date}.feather"

    logger.info(f"Loading test data from:")
    logger.info(f"  Features: {arg_file}")
    logger.info(f"  Targets:  {trg_file}")

    if not arg_file.exists():
        logger.error(f"Feature file not found: {arg_file}")
        return
    if not trg_file.exists():
        logger.error(f"Target file not found: {trg_file}")
        return

    # Prepare test data
    X_test, Y_true, timestamps, feature_cols = prepare_test_data(
        arg_file=arg_file,
        trg_file=trg_file,
        window_size=args.window_size,
        step=5  # Use step=5 to reduce samples for display
    )

    logger.info(f"Test data shape: X={X_test.shape}, Y={Y_true.shape}")

    # Load model
    logger.info(f"Loading model from: {args.model}")
    model = LSTMAttentionModel(
        sequence_length=args.window_size,
        n_features=X_test.shape[2],
        model_path=args.model
    )

    # Make predictions
    logger.info("Making predictions...")
    Y_pred = model.predict(X_test)

    # Since model outputs continuous values but targets are -1, 0, 1,
    # round predictions to nearest integer for classification
    Y_pred_rounded = np.round(Y_pred).astype(int)

    # Calculate accuracy
    accuracy = np.mean(Y_pred_rounded == Y_true)
    mae = np.mean(np.abs(Y_pred - Y_true))

    logger.info(f"\n{'='*60}")
    logger.info(f"Prediction Results")
    logger.info(f"{'='*60}")
    logger.info(f"Total samples: {len(Y_true)}")
    logger.info(f"Accuracy (rounded): {accuracy:.2%}")
    logger.info(f"MAE (continuous): {mae:.4f}")

    # Show value distribution
    true_counts = pd.Series(Y_true).value_counts().sort_index()
    pred_counts = pd.Series(Y_pred_rounded).value_counts().sort_index()

    logger.info(f"\nTrue values distribution:")
    for val, count in true_counts.items():
        logger.info(f"  {val}: {count} ({100*count/len(Y_true):.1f}%)")

    logger.info(f"\nPredicted values distribution:")
    for val, count in pred_counts.items():
        logger.info(f"  {val}: {count} ({100*count/len(Y_pred_rounded):.1f}%)")

    # Show sample predictions
    logger.info(f"\n{'='*60}")
    logger.info(f"Sample Predictions (first {args.max_samples}):")
    logger.info(f"{'='*60}")
    logger.info(f"{'Index':<8} {'Timestamp':<30} {'Actual':<8} {'Predicted':<12} {'Rounded':<8} {'Match':<6}")
    logger.info("-" * 60)

    n_show = min(args.max_samples, len(Y_true))
    for i in range(n_show):
        ts = str(timestamps[i])[:26] if i < len(timestamps) else "N/A"
        match = "✓" if Y_pred_rounded[i] == Y_true[i] else "✗"
        logger.info(f"{i:<8} {ts:<30} {Y_true[i]:<8} {Y_pred[i]:<12.4f} {Y_pred_rounded[i]:<8} {match:<6}")

    # Confusion matrix style analysis
    logger.info(f"\n{'='*60}")
    logger.info(f"Prediction Accuracy by Class:")
    logger.info(f"{'='*60}")

    for cls in [-1, 0, 1]:
        mask = Y_true == cls
        if mask.sum() > 0:
            cls_acc = np.mean(Y_pred_rounded[mask] == cls)
            cls_count = mask.sum()
            logger.info(f"Class {cls:>2}: {cls_acc:.2%} correct ({cls_count} samples)")

    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
