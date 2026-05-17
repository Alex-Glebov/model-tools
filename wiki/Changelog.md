# Changelog

## v0.1.2 — 2026-05-17

### Changed
- `__version__` now resolved from installed package metadata (`importlib.metadata`) instead of a hard-coded string.
- Removed `main()` and `if __name__ == '__main__'` blocks from `model_tools.analysis.analyze_chains`, `model_tools.viz.plot_metrics`, and `model_tools.viz.standalone_plot_peaks`.
- Added CLI runner scripts under `tests/` for the above utilities so they remain runnable as one-offs.

## v0.1.1 — 2026-05-13

### Added
- `model_tools.analysis` — chain-length analysis (`extract_chain_lengths`, `analyze_all_chains`) for determining optimal window sizes.
- `model_tools.viz` — training metrics plotting (`plot_training_history`) and market-data peak overlays.
- `model_tools.io` — thin wrappers around `homebrewlibra` I/O utilities.

### Notes
- Initial release.
