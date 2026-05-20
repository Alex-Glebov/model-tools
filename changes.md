# Changes

## 0.1.4 (2026-05-20)
- Rename `df` parameter to `ddf` across viz modules for naming consistency:
  - `market_plots.py`: `plot_volume`, `plot_prices`, `plot_peaks`, `plot_prices_OMWC`
  - `plot_price_peaks.py`: `_plot_price_peaks_impl`, `plot_price_peaks`
- No logic changes, no breaking changes.

## 0.1.3 (previous)
- Rename `plot_price_peaks`, add `detach` parameter, return `df` for pipeline chaining.
