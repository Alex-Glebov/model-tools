#!/usr/bin/env python3
"""CLI smoke test for plotting training metrics from a model config JSON.

Loads a model config file (produced by model-core training) and renders
loss-curve / histogram plots via ``plot_training_history``.
Can display interactively or save to a PNG file.
"""
import argparse
import sys
from pathlib import Path

from model_tools.viz import plot_training_history


def main():
    parser = argparse.ArgumentParser(
        description="Plot training metrics from model config"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model config JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for plot image (e.g., plot.png). If not set, displays interactively"
    )

    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    output_path = Path(args.output).expanduser() if args.output else None

    plot_training_history(config_path, output_path)


if __name__ == "__main__":
    main()
