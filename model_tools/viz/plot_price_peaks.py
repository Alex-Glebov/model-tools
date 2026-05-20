"""Plot trade data with zigzag peaks overlay.

Expects columns: timestamp, price, peaks_line.
"""
from __future__ import annotations

import multiprocessing
from typing import Optional

import pandas as pd


def _plot_price_peaks_impl(ddf: pd.DataFrame, title: str, save_path: Optional[str],
                           **kwargs,
                          )->dict:
    """Shared plotting implementation (runs in main or child process)."""
    import matplotlib
    if save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = ddf["timestamp"]
    price = pd.to_numeric(ddf["price"], errors="coerce")

    fig, ax = plt.subplots(figsize=(24, 10))

    # Price in high-contrast blue
    ax.plot(ts, price, color="#1f77b4", linewidth=0.8, alpha=0.9, label="price")

    # Zigzag line overlay
    if "peaks_line" in ddf.columns:
        peaks_line = pd.to_numeric(ddf["peaks_line"], errors="coerce")
        ax.plot(ts, peaks_line, color="black", linewidth=2.0, label="peaks_line")

        # Pivot markers
        peak_mask = ddf.get("peaks") == 1
        valley_mask = ddf.get("peaks") == -1
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

    return {'ddf':ddf}


def plot_price_peaks(
    ddf: pd.DataFrame,
    title: str = "",
    save_path: Optional[str] = None,
    detach: bool = False,
    **kwargs,
    )-> dict:
#   )-> pd.DataFrame:
    """Plot price + peaks_line overlay.

    Args:
        ddf: DataFrame with timestamp, price, and optional peaks_line / peaks columns.
        title: Plot title.
        save_path: If given, save to file instead of displaying interactively.
        detach: When True, spawn the plot in a separate process so the caller
            does not block on ``plt.show()``. Ignored when ``save_path`` is set.

    Returns:
        The same *ddf* passed in, so the function can be chained in data-prep
        pipelines (e.g. ``ddf = plot_price_peaks(ddf, ...)``).
    """
    if detach:
        p = multiprocessing.Process(target=_plot_price_peaks_impl, args=(ddf, title, save_path))
        p.start()
    else:
        _plot_price_peaks_impl(ddf, title, save_path)
    return {'ddf':ddf}
