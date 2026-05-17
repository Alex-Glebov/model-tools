# Model Tools

Diagnostics, analysis, and visualization for ML model development.

## Purpose

This package provides utilities used across model sub-projects:
- **analysis** — Chain-length analysis, metrics computation
- **viz** — Training curves, market plots (OHLCV, peaks, volume)
- **io** — Thin wrappers around homebrewlibra I/O

**What is NOT here:**
- Data preparation (sliding window, standardize, split) → `data-prep`
- Model-specific tensor loading → `model-core`

## Installation

```bash
# From model-tools directory
pip install -e .

# With tests
pip install -e ".[tests]"

# From GitHub
pip install git+https://github.com/Alex-Glebov/model-tools.git
```

## Package Structure

```
model_tools/
├── analysis/      # Chain analysis, metrics
│   └── analyze_chains.py
├── viz/           # Visualization
│   ├── plot_metrics.py
│   └── market_plots.py
└── io/            # I/O wrappers (homebrewlibra)
```

## Usage

```python
import model_tools
print(model_tools.__version__)  # 0.1.1

# Analysis
from model_tools.analysis import analyze_all_chains
min_len, max_len, mean_len, all_lengths = analyze_all_chains(data_path)

# Visualization
from model_tools.viz import plot_training_history
plot_training_history(config_path)
```

## Tests

```bash
pytest tests/ -v
```

## Version

`0.1.1` — defined in `pyproject.toml`.

## Author

Alex & Claude <python@iitsp.com.au>
