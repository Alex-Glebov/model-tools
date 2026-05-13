"""Plot trade data with zigzag peaks overlay.

Usage:
    python plot_peaks.py /path/to/trades.feather           # interactive display
    python plot_peaks.py /path/to/trades.feather --save output.png

Expects columns: timestamp, price, peaks_line.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def plot_peaks(df: pd.DataFrame, title: str = "", save_path: str | None = None):
    """Plot price + peaks_line overlay."""
    import matplotlib
    if save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = df["timestamp"]
    price = pd.to_numeric(df["price"], errors="coerce")

    fig, ax = plt.subplots(figsize=(24, 10))

    # Price in high-contrast blue
    ax.plot(ts, price, color="#1f77b4", linewidth=0.8, alpha=0.9, label="price")

    # Zigzag line overlay
    if "peaks_line" in df.columns:
        peaks_line = pd.to_numeric(df["peaks_line"], errors="coerce")
        ax.plot(ts, peaks_line, color="black", linewidth=2.0, label="peaks_line")

        # Pivot markers
        peak_mask = df.get("peaks") == 1
        valley_mask = df.get("peaks") == -1
        if peak_mask.any():
            ax.scatter(ts[peak_mask], price[peak_mask], color="green", s=40, zorder=5, marker="^", label="peak")
        if valley_mask.any():
            ax.scatter(ts[valley_mask], price[valley_mask], color="red", s=40, zorder=5, marker="v", label="valley")
    else:
        print("WARNING: 'peaks_line' not found — run insert_peaks first.")

    ax.set_title(title or "Trades + Zigzag Peaks")
    ax.set_xlabel("timestamp")
    ax.set_ylabel("price")
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()


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
    plot_peaks(df, title=args.title, save_path=args.save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
