#!/usr/bin/env python3
"""CLI smoke test for plotting trade data with zigzag peaks overlay.

Loads a feather file of trade data and renders the price series together
with zigzag peak/valley markers and the peaks_line overlay.
Can display interactively or save to a PNG file.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from model_tools.viz.plot_price_peaks import plot_price_peaks


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot trades with zigzag peaks")
    parser.add_argument("file", help="Path to feather file")
    parser.add_argument("--save", "-o", help="Save plot to file instead of displaying")
    parser.add_argument("--title", "-t", default="", help="Plot title")
    args = parser.parse_args()

    if not Path(args.file).exists():
        print(f"File not found: {args.file}")
        return 1

    df = pd.read_feather(args.file).sort_values("timestamp").reset_index(drop=True)
    print(f"Loaded {len(df):,} rows")
    plot_price_peaks(df, title=args.title, save_path=args.save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
