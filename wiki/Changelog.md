# Changelog

## v0.1.3 — 2026-05-17

### Changed
- Renamed `model_tools.viz.standalone_plot_peaks` → `model_tools.viz.plot_price_peaks`.
- Function `plot_peaks()` → `plot_price_peaks()` now returns the input DataFrame so it
  can be chained in data-prep pipelines.
- Added `detach` parameter: when True, spawns the plot window in a separate process
  so the caller does not block on ``plt.show()``.

## v0.1.2 — 2026-05-17

### Changed
- `__version__` now resolved from installed package metadata (`importlib.metadata`) instead of a hard-coded string.
- Removed `main()` and `if __name__ == '__main__'` blocks from `model_tools.analysis.analyze_chains`, `model_tools.viz.plot_metrics`, and `model_tools.viz.standalone_plot_peaks`.
- Renamed `model_tools.viz.standalone_plot_peaks` → `model_tools.viz.plot_price_peaks`; function `plot_peaks` → `plot_price_peaks` with new `detach` parameter for non-blocking display.
- Added CLI runner scripts under `tests/` for the above utilities so they remain runnable as one-offs.

## v0.1.1 — 2026-05-13

### Added
- `model_tools.analysis` — chain-length analysis (`extract_chain_lengths`, `analyze_all_chains`) for determining optimal window sizes.
- `model_tools.viz` — training metrics plotting (`plot_training_history`) and market-data peak overlays.
- `model_tools.io` — thin wrappers around `homebrewlibra` I/O utilities.

### Notes
- Initial release.
