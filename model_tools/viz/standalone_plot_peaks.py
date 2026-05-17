"""Plot trade data with zigzag peaks overlay.

Expects columns: timestamp, price, peaks_line.
"""
from __future__ import annotations

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


