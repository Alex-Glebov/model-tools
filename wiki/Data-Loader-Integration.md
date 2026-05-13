# Data Loader Integration

This document describes how `model-tools`, `data-prep`, and `model-core` work together across package boundaries.

## Package Boundaries

| Package | Responsibility | Does NOT own |
|---|---|---|
| **data-prep** | Deterministic data transformations (standardize, split, sliding-window `.npy` generation) | Model-specific tensor shapes |
| **model-tools** | Diagnostics, analysis, visualization (chain lengths, loss curves, peak plots) | Data conversion, model training |
| **model-core** | Model-specific data loaders, training, inference | Generic data prep |
| **homebrewlibra** | Low-level I/O primitives (feather, CSV, npy, template resolution) | Domain-specific transforms |

## Data Flow

```
raw CSV/feather ──▶ data-prep ──▶ .npy files (arg-final/, trg-final/)
                                         │
                                         ▼
                              model-core data_loader.py
                                         │
                                         ▼
                              model-ready tensors (X, Y)
```

- **data-prep** outputs generic `.npy` files: `X.npy` `(n_samples, window_size, n_features)` and `Y.npy` `(n_samples,)`.
- **model-core** reads `.npy` and wraps into framework-specific tensors (TensorFlow ragged, PyTorch, etc.).
- **model-tools** inspects data and models for diagnostics (chain lengths, loss curves) but does not transform data.

## model-tools Analysis

### Chain Length Analysis

```python
from model_tools.analysis import analyze_all_chains

min_len, max_len, mean_len, all_lengths = analyze_all_chains(
    data_path=Path("~/rawdata/csv/trg_final"),
    pattern="*.feather"
)

print(f"Recommended window_size: {min_len}")
```

This is a **diagnostic** — it inspects target data and recommends hyperparameters. It does not produce training data.

## model-core Data Loading

### Two-Stage Architecture

Model-core uses a two-stage pipeline:

```
Stage 1 (Raw Reader)          Stage 2 (Converter)
     │                               │
     ▼                               ▼
┌──────────────┐              ┌──────────────┐
│  Read .npy   │ ─────────▶ │ Transform    │
│  files       │   X_raw,   │ (framework    │
│              │   Y_raw    │  specific)    │
└──────────────┘              └──────────────┘
                                     │
                                     ▼
                              ┌──────────────┐
                              │ Model-Ready  │
                              │ Tensors      │
                              └──────────────┘
```

Stage 1 reads `.npy` from disk. Stage 2 is model-specific (e.g., ragged tensors for TF, custom padding for LSTM+Attention).

### Custom Data Converter (Stage 2)

```python
from model_core.model import LSTMAttentionModel

def normalize_converter(X_raw, Y_raw, window_size=100, stride=1, **kwargs):
    """Custom converter with normalization."""
    from model_core.data_loader import default_convert_raw_to_model_format

    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0)
    X_normalized = (X_raw - X_mean) / (X_std + 1e-8)

    X_model, Y_model = default_convert_raw_to_model_format(
        X_normalized, Y_raw, window_size=window_size, stride=stride, **kwargs
    )
    return X_model, Y_model

model = LSTMAttentionModel(sequence_length=100, n_features=9)
model.set_data_converter(normalize_converter)
X, Y = model.load_data(X_path, Y_path)
```

## Testing Custom Converters

```python
import numpy as np

# Synthetic raw data (from data-prep output format)
X_raw = np.random.randn(1000, 9)  # 1000 timesteps, 9 features
Y_raw = np.random.randn(1000)

# Test converter
X_model, Y_model = normalize_converter(X_raw, Y_raw, window_size=100)

# Verify output
assert X_model.shape == (901, 100, 9)
assert Y_model.shape == (901,)
assert np.allclose(X_model.mean(), 0, atol=0.1)
```

## See Also

- `data-prep` — `feather_to_npy_file()` produces `.npy` from feather
- `homebrewlibra` — `helper_npy.load_npy()` / `save_npy()` for I/O
- `model-core/Data-Loading` — Two-stage architecture documentation
