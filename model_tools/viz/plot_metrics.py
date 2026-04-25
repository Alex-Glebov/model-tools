#!/usr/bin/env python3
"""Plot training metrics from model config.

Usage:
    python plot_metrics.py --config models/model_20260423_220531_config.json
    python plot_metrics.py --config models/model_20260423_220531_config.json --output plot.png
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed. Install with: pip install matplotlib")
    sys.exit(1)


def plot_training_history(config_path: Path, output_path: Path = None):
    """Plot training history from config file.

    Args:
        config_path: Path to model config JSON
        output_path: Optional path to save plot image
    """
    # Load config
    with open(config_path) as f:
        config = json.load(f)

    # Extract history
    history = config.get('history', {})
    if not history:
        print(f"No history found in {config_path}")
        return

    loss = history.get('loss', [])
    val_loss = history.get('val_loss', [])

    if not loss:
        print("No loss data in history")
        return

    epochs = list(range(1, len(loss) + 1))

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Loss over epochs
    ax1 = axes[0]
    ax1.plot(epochs, loss, 'b-', label='Training Loss', linewidth=1.5)
    if val_loss:
        ax1.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training Loss Over Epochs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Loss distribution (histogram)
    ax2 = axes[1]
    ax2.hist(loss, bins=30, alpha=0.7, label='Training Loss', color='blue')
    if val_loss:
        ax2.hist(val_loss, bins=30, alpha=0.7, label='Validation Loss', color='red')
    ax2.set_xlabel('Loss Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Loss Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add summary text
    final_val = f"{val_loss[-1]:.4f}" if val_loss else 'N/A'
    min_val = f"{min(val_loss):.4f}" if val_loss else 'N/A'
    summary_text = f"""Model: LSTM+Attention
Epochs: {len(loss)}
Final Train Loss: {loss[-1]:.4f}
Final Val Loss: {final_val}
Min Train Loss: {min(loss):.4f}
Min Val Loss: {min_val}"""
    fig.text(0.02, 0.02, summary_text, fontsize=9, family='monospace',
             verticalalignment='bottom')

    # Layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)

    # Show test metrics if available
    test_metrics = config.get('test_metrics', {})
    if test_metrics:
        test_text = f"""Test Metrics:
  Loss: {test_metrics.get('loss', 0):.4f}
  MAE: {test_metrics.get('mae', 0):.4f}"""
        fig.text(0.98, 0.02, test_text, fontsize=9, family='monospace',
                 verticalalignment='bottom', horizontalalignment='right')

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()

    # Print summary
    print(f"\n{'='*50}")
    print(f"Training Summary")
    print(f"{'='*50}")
    print(f"Total epochs: {len(loss)}")
    print(f"Final train loss: {loss[-1]:.4f}")
    if val_loss:
        print(f"Final val loss:   {val_loss[-1]:.4f}")
        print(f"Min val loss:     {min(val_loss):.4f} (epoch {val_loss.index(min(val_loss)) + 1})")
    print(f"{'='*50}")


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
