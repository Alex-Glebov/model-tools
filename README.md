# Model Tools

Shared utilities for ML model development in the model project.

## Purpose

This package provides common utilities used across model sub-projects:
- **data-prep** - Data preparation service
- **retrain-monitor** - Model drift detection
- **model-training** - Training orchestration

## Installation

```bash
# From model-tools directory
pip install -e .

# Or from model root
pip install -e model-tools/

# From GitHub (latest)
pip install git+https://github.com/Alex-Glebov/model-tools.git
```

## Package Structure

```
model_tools/
├── dataprep/      # Data preparation utilities
│   └── prepare_train_data.py
├── analysis/      # Metrics and chain analysis
│   └── analyze_chains.py
├── viz/           # Visualization
│   └── plot_metrics.py
└── io/            # File I/O helpers
```

## Usage

```python
# Check version
import model_tools
print(model_tools.__version__)  # 0.1.0

# Data preparation
from model_tools.dataprep import prepare_data
X, Y = prepare_data(arg_file, trg_file, window_size=42)

# Analysis
from model_tools.analysis import extract_chain_lengths
stats = extract_chain_lengths(data_file)

# Visualization
from model_tools.viz import plot_training_history
plot_training_history(config_path)
```

## Integration Tests

Tests that require model-core:
```bash
cd tests/integration
python test_prediction.py --model path/to/model.keras --pair BTC_USD
```

## Development

```bash
# Clone and install in dev mode
git clone https://github.com/Alex-Glebov/model-tools.git
cd model-tools
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## Version

Version is defined in `model_tools/__init__.py` as `__version__` and read dynamically by `pyproject.toml`.

## Author

Alex & Claude <python@iitsp.com.au>
