#!/usr/bin/env python3
"""Analyze target data chains to find minimum sequence length.

A chain is a contiguous sequence of rows with the same 'pair' value.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    return logging.getLogger(__name__)


def extract_chain_lengths(file_path: Path) -> Dict[str, int]:
    """
    Extract chain lengths from a feather file.

    A chain is a contiguous sequence of rows with the same 'pair' value.
    When the pair value changes, a new chain starts.

    Args:
        file_path: Path to feather file

    Returns:
        Dict mapping chain_id -> length
    """
    df = pd.read_feather(file_path)

    if 'pair' not in df.columns:
        return {}

    chains = {}
    chain_idx = 0
    current_pair = None
    current_length = 0

    for pair in df['pair']:
        if pair != current_pair:
            # Save previous chain
            if current_pair is not None:
                chains[f"{current_pair}_{chain_idx}"] = current_length
                chain_idx += 1
            # Start new chain
            current_pair = pair
            current_length = 1
        else:
            current_length += 1

    # Save last chain
    if current_pair is not None:
        chains[f"{current_pair}_{chain_idx}"] = current_length

    return chains


def analyze_all_chains(data_path: Path, pattern: str = "*.feather") -> Tuple[int, int, float, Dict]:
    """
    Analyze all chains in the target data directory.

    Args:
        data_path: Path to directory containing feather files
        pattern: Glob pattern for files

    Returns:
        (min_length, max_length, mean_length, all_lengths_dict)
    """
    logger = logging.getLogger(__name__)

    files = sorted(data_path.glob(pattern))
    logger.info(f"Found {len(files)} files in {data_path}")

    all_lengths = {}

    for i, file_path in enumerate(files):
        if i % 100 == 0:
            logger.info(f"Processed {i}/{len(files)} files...")

        try:
            file_chains = extract_chain_lengths(file_path)
            # Prefix with filename for uniqueness
            for chain_id, length in file_chains.items():
                all_lengths[f"{file_path.stem}__{chain_id}"] = length
        except Exception as e:
            logger.warning(f"Failed to process {file_path}: {e}")

    if not all_lengths:
        return 0, 0, 0.0, {}

    lengths = list(all_lengths.values())
    min_length = min(lengths)
    max_length = max(lengths)
    mean_length = np.mean(lengths)

    return min_length, max_length, mean_length, all_lengths


def main():
    parser = argparse.ArgumentParser(
        description="Analyze target data chains to find minimum sequence length"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="~/rawdata/csv/trg_final",
        help="Path to target data directory (default: ~/rawdata/csv/trg_final)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: save chain lengths to JSON file"
    )

    args = parser.parse_args()
    logger = setup_logging()

    data_path = Path(args.data_path).expanduser()
    logger.info(f"Analyzing chains in: {data_path}")

    min_len, max_len, mean_len, all_lengths = analyze_all_chains(data_path)

    logger.info("=" * 50)
    logger.info(f"Chain Length Statistics:")
    logger.info(f"  Total chains: {len(all_lengths)}")
    logger.info(f"  Min length: {min_len}")
    logger.info(f"  Max length: {max_len}")
    logger.info(f"  Mean length: {mean_len:.2f}")
    logger.info("=" * 50)

    # Show shortest chains
    logger.info("\nShortest chains (first 10):")
    sorted_chains = sorted(all_lengths.items(), key=lambda x: x[1])
    for chain_id, length in sorted_chains[:10]:
        logger.info(f"  {chain_id}: {length}")

    # Show longest chains
    logger.info("\nLongest chains (first 10):")
    for chain_id, length in sorted_chains[-10:]:
        logger.info(f"  {chain_id}: {length}")

    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump({
                "statistics": {
                    "total_chains": len(all_lengths),
                    "min_length": int(min_len),
                    "max_length": int(max_len),
                    "mean_length": float(mean_len)
                },
                "chains": all_lengths
            }, f, indent=2)
        logger.info(f"\nSaved detailed results to: {args.output}")

    logger.info(f"\nRecommended window_size: {min_len}")


if __name__ == "__main__":
    main()
