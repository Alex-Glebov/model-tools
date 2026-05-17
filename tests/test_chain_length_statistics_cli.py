#!/usr/bin/env python3
"""CLI smoke test for chain-length statistics on real data directories.

Runs ``analyze_all_chains`` against a feather directory and prints
min / max / mean chain lengths plus the recommended ``window_size``.
Optionally persists the full result map to JSON.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from model_tools.analysis import analyze_all_chains


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    return logging.getLogger(__name__)


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
    logger = _setup_logging()

    data_path = Path(args.data_path).expanduser()
    logger.info(f"Analyzing chains in: {data_path}")

    min_len, max_len, mean_len, all_lengths = analyze_all_chains(data_path)

    logger.info("=" * 50)
    logger.info("Chain Length Statistics:")
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
