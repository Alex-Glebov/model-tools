"""Model Tools - Shared utilities for ML model development.

This package provides shared utilities used across model sub-projects:
- analysis: Metrics computation and data analysis
- viz: Visualization and plotting
- io: File I/O utilities

Data preparation (sliding window, standardize, split) lives in data-prep.
Model-specific tensor loading lives in model-core.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("model-tools")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

__author__ = "Alex & Claude <python@iitsp.com.au>"

from . import analysis
from . import viz
from . import io

__all__ = ["__version__", "analysis", "viz", "io"]
