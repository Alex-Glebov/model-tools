# Data Loader Integration

Model-tools provides custom data loaders that can be injected into model-core's `LSTMAttentionModel` using the two-stage architecture.

## Overview

Model-core uses a **two-stage data loading architecture**:

```
Stage 1 (Raw Reader)          Stage 2 (Converter)
     │                               │
     ▼                               ▼
┌──────────────┐              ┌──────────────┐
│  Read Files  │ ─────────▶ │ Transform    │
│  (.npy)      │   X_raw,   │ (windowing,  │
│              │   Y_raw    │ normalization│
└──────────────┘              └──────────────┘
                                     │
                                     ▼
                              ┌──────────────┐
                              │ Model-Ready  │
                              │ (X, Y)       │
                              └──────────────┘
```

Model-tools can provide:
- **Custom raw readers** - Read from S3, databases, etc.
- **Custom converters** - Normalization, augmentation, custom windowing

## Injecting Custom Components

### Custom Data Converter (Stage 2)

The most common customization - transform raw data before windowing:

```python
from model_core.model import LSTMAttentionModel

# Define custom converter
def normalize_converter(X_raw, Y_raw, window_size=100, stride=1, **kwargs):
    """Custom converter with normalization."""
    from model_core.data_loader import default_convert_raw_to_model_format
    
    # Normalize raw data
    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0)
    X_normalized = (X_raw - X_mean) / (X_std + 1e-8)
    
    # Apply standard windowing
    X_model, Y_model = default_convert_raw_to_model_format(
        X_normalized, Y_raw, window_size=window_size, stride=stride, **kwargs
    )
    
    return X_model, Y_model

# Use in model
model = LSTMAttentionModel(sequence_length=100, n_features=9)
model.set_data_converter(normalize_converter)

# Load data with custom converter
X, Y = model.load_data(X_path, Y_path)
```

### Custom Raw Reader (Stage 1)

Read data from custom sources:

```python
def s3_raw_reader(X_path, Y_path, pairs=None, date_range=None, **kwargs):
    """Read raw data from S3."""
    import boto3
    
    s3 = boto3.client('s3')
    
    for x_key, y_key in list_s3_files(X_path, Y_path, pairs, date_range):
        # Download and load
        x_data = download_from_s3(s3, x_key)
        y_data = download_from_s3(s3, y_key)
        
        X_raw = np.load(io.BytesIO(x_data))
        Y_raw = np.load(io.BytesIO(y_data))
        
        yield X_raw, Y_raw

# Use in model
model.set_raw_reader(s3_raw_reader)
X, Y = model.load_data("s3://bucket/arg-final", "s3://bucket/trg-final")
```

## Converter Examples

### Feature Scaling

```python
def min_max_scaler_converter(X_raw, Y_raw, window_size=100, **kwargs):
    """Scale features to [0, 1] range."""
    X_min = X_raw.min(axis=0)
    X_max = X_raw.max(axis=0)
    X_scaled = (X_raw - X_min) / (X_max - X_min + 1e-8)
    
    from model_core.data_loader import default_convert_raw_to_model_format
    return default_convert_raw_to_model_format(
        X_scaled, Y_raw, window_size=window_size, **kwargs
    )
```

### Data Augmentation

```python
def augmentation_converter(X_raw, Y_raw, window_size=100, noise_factor=0.01, **kwargs):
    """Add noise to raw data before windowing."""
    noise = np.random.normal(0, noise_factor, X_raw.shape)
    X_augmented = X_raw + noise
    
    from model_core.data_loader import default_convert_raw_to_model_format
    return default_convert_raw_to_model_format(
        X_augmented, Y_raw, window_size=window_size, **kwargs
    )
```

### Custom Windowing

```python
def custom_windowing_converter(X_raw, Y_raw, window_size=100, **kwargs):
    """Custom windowing with different target selection."""
    n_timesteps, n_features = X_raw.shape
    n_windows = n_timesteps - window_size + 1
    
    X_windows = np.zeros((n_windows, window_size, n_features))
    # Custom: predict average of window instead of last value
    Y_targets = np.zeros(n_windows)
    
    for i in range(n_windows):
        start = i
        end = i + window_size
        X_windows[i] = X_raw[start:end]
        Y_targets[i] = Y_raw[start:end].mean()  # Average target
    
    return X_windows, Y_targets
```

## Chained Converters

Chain multiple transformations:

```python
def chained_converter(X_raw, Y_raw, window_size=100, **kwargs):
    """Apply multiple transformations in sequence."""
    from model_tools.dataprep import (
        normalize_raw,
        augment_raw,
        window_raw
    )
    
    # Step 1: Normalize
    X = normalize_raw(X_raw)
    
    # Step 2: Augment
    X = augment_raw(X, noise_factor=0.01)
    
    # Step 3: Window
    X_model, Y_model = window_raw(X, Y_raw, window_size=window_size, **kwargs)
    
    return X_model, Y_model

model.set_data_converter(chained_converter)
```

## Streaming with Custom Components

Custom converters work with streaming:

```python
model.set_data_converter(normalize_converter)

# Stream with normalization applied to each file
for X_batch, Y_batch in model.load_data_stream(X_path, Y_path):
    # X_batch is already normalized and windowed
    train(X_batch, Y_batch)
```

## Type Signatures

```python
from typing import Callable, Iterator, Tuple
import numpy as np

# Raw reader: yields (X_raw, Y_raw) from files
# X_raw shape: (timesteps, features)
# Y_raw shape: (timesteps,)
RawReaderFn = Callable[..., Iterator[Tuple[np.ndarray, np.ndarray]]]

# Converter: transforms raw to model-ready
# Input: (timesteps, features), (timesteps,)
# Output: (n_windows, window_size, features), (n_windows,)
ConverterFn = Callable[..., Tuple[np.ndarray, np.ndarray]]
```

## Best Practices

1. **Keep converters pure** - Don't modify input arrays in place
2. **Handle edge cases** - Check for empty arrays, insufficient timesteps
3. **Preserve dtypes** - Use `astype(np.float32)` if needed
4. **Log transformations** - Use `logger.info()` for significant changes
5. **Test separately** - Test converters with synthetic data

## Testing Custom Converters

```python
import numpy as np

# Create synthetic raw data
X_raw = np.random.randn(1000, 9)  # 1000 timesteps, 9 features
Y_raw = np.random.randn(1000)

# Test converter
X_model, Y_model = normalize_converter(X_raw, Y_raw, window_size=100)

# Verify output
assert X_model.shape == (901, 100, 9)  # (n_windows, window_size, features)
assert Y_model.shape == (901,)         # (n_windows,)
assert np.allclose(X_model.mean(), 0, atol=0.1)  # Approximately normalized
```

## See Also

- [[model-core/Data-Loading]] - Two-stage architecture documentation
- [[model-core/Streaming]] - Streaming with two-stage pipeline
