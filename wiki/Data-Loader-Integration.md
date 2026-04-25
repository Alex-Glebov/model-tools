# Data Loader Integration

Model-tools provides custom data loaders that can be injected into model-core's `LSTMAttentionModel`.

## Overview

Model-core defines a standardized data loading interface. Model-tools can provide alternative implementations that are injected at runtime.

```python
# Default (model-core)
model = LSTMAttentionModel(...)
X, Y = model.load_data(X_path, Y_path)  # Uses default_load_chains

# With custom loader from model-tools
from model_tools.dataprep import custom_loader
model.set_data_loader(custom_loader)
X, Y = model.load_data(X_path, Y_path)  # Uses custom_loader
```

## Standardized Interface

All data loaders must follow this signature:

```python
def loader_fn(
    X_path: Union[str, Path],
    Y_path: Union[str, Path],
    pairs: Optional[list[str]] = None,
    date_range: Optional[str] = None,
    date_backward: bool = False,
    date_period: Optional[str] = None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Load data and return (X, Y) arrays."""
    ...
```

## Creating Custom Loaders

Example custom loader in model-tools:

```python
# model_tools/dataprep/custom_loader.py
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

def enhanced_loader(
    X_path: Path,
    Y_path: Path,
    pairs: Optional[list[str]] = None,
    date_range: Optional[str] = None,
    date_backward: bool = False,
    date_period: Optional[str] = None,
    normalize: bool = True,  # Custom param
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Custom loader with normalization."""
    # Load data using default logic
    from model_core.data_loader import default_load_chains
    X, Y = default_load_chains(
        X_path, Y_path,
        pairs=pairs,
        date_range=date_range,
        date_backward=date_backward,
        date_period=date_period
    )
    
    # Add custom processing
    if normalize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    return X, Y
```

## Streaming Loaders

For large datasets that don't fit in memory:

```python
from typing import Iterator, Tuple
import numpy as np

def streaming_loader(
    X_path: Path,
    Y_path: Path,
    batch_size: int = 1000,
    **kwargs
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield batches one at a time."""
    for batch_x, batch_y in yield_batches(X_path, Y_path, batch_size):
        yield batch_x, batch_y

# Usage in model
model.set_data_loader_stream(streaming_loader)
for X_batch, Y_batch in model.load_data_stream(X_path, Y_path, batch_size=1000):
    model.partial_fit(X_batch, Y_batch)
```

## Integration Example

```python
from model_core.model import LSTMAttentionModel
from model_tools.dataprep.custom_loader import enhanced_loader

# Create model
model = LSTMAttentionModel(sequence_length=100, n_features=9)

# Inject custom loader
model.set_data_loader(enhanced_loader)

# Load data with custom processing
X, Y = model.load_data(
    X_path="~/rawdata/arg-final",
    Y_path="~/rawdata/trg-final",
    pairs=["BTC_USD"],
    date_range="20200101-20201231",
    normalize=True  # Custom param passed through
)
```

## Future Loaders

Planned data loaders for model-tools:

- `cached_loader` - Caches loaded data to disk
- `augmented_loader` - Applies data augmentation
- `distributed_loader` - Loads from distributed storage
- `feather_loader` - Loads raw Feather files directly

## See Also

- [[model-core/Data-Loading]] - Core data loading documentation
- [[model-core/Streaming]] - Streaming large datasets
