"""Test visualization utilities (smoke tests)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestPlotPeaks:
    """Smoke tests for :func:`model_tools.viz.plot_peaks`."""

    def test_plot_peaks_runs(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=50, freq="min"),
            "price": 100 + np.cumsum(np.random.randn(50) * 0.1),
            "peaks": np.random.choice([0, 1, -1], 50),
            "peaks_line": 100 + np.cumsum(np.random.randn(50) * 0.05),
        })
        from model_tools.viz import plot_peaks
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plot_peaks(ax, df)
        plt.close(fig)


class TestPlotTrainingHistory:
    """Smoke tests for :func:`model_tools.viz.plot_training_history`."""

    def test_plot_training_history_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "history": {
                    "loss": [0.5, 0.4, 0.3, 0.2],
                    "val_loss": [0.55, 0.45, 0.35, 0.25],
                },
                "test_metrics": {"loss": 0.18, "mae": 0.15},
            }
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(str(config).replace("'", '"'))
            from model_tools.viz import plot_training_history
            import matplotlib
            matplotlib.use("Agg")
            plot_training_history(config_path)

    def test_plot_training_history_no_history(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"history": {}}
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(str(config).replace("'", '"'))
            from model_tools.viz import plot_training_history
            plot_training_history(config_path)
            captured = capsys.readouterr()
            assert "No history" in captured.out or "No loss" in captured.out
