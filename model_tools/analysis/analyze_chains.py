#!/usr/bin/env python3
"""Analyze target data chains to find minimum sequence length.

A chain is a contiguous sequence of rows with the same 'pair' value.
"""

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
