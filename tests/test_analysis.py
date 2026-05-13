"""Test chain analysis utilities."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestExtractChainLengths:
    """Tests for :func:`model_tools.analysis.extract_chain_lengths`."""

    def test_single_chain(self):
        df = pd.DataFrame({"pair": ["A"] * 10})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.feather"
            df.to_feather(path)
            from model_tools.analysis import extract_chain_lengths
            result = extract_chain_lengths(path)
        assert len(result) == 1
        assert list(result.values())[0] == 10

    def test_multiple_chains(self):
        df = pd.DataFrame({"pair": ["A"] * 5 + ["B"] * 3 + ["A"] * 2})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.feather"
            df.to_feather(path)
            from model_tools.analysis import extract_chain_lengths
            result = extract_chain_lengths(path)
        assert len(result) == 3
        lengths = sorted(result.values())
        assert lengths == [2, 3, 5]

    def test_missing_pair_column(self):
        df = pd.DataFrame({"other": [1, 2, 3]})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.feather"
            df.to_feather(path)
            from model_tools.analysis import extract_chain_lengths
            result = extract_chain_lengths(path)
        assert result == {}

    def test_empty_file(self):
        df = pd.DataFrame({"pair": pd.Series([], dtype=str)})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.feather"
            df.to_feather(path)
            from model_tools.analysis import extract_chain_lengths
            result = extract_chain_lengths(path)
        assert result == {}


class TestAnalyzeAllChains:
    """Tests for :func:`model_tools.analysis.analyze_all_chains`."""

    def test_directory_with_multiple_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            # File 1: two chains
            df1 = pd.DataFrame({"pair": ["A"] * 10 + ["B"] * 5})
            df1.to_feather(base / "file1.feather")
            # File 2: one chain
            df2 = pd.DataFrame({"pair": ["C"] * 20})
            df2.to_feather(base / "file2.feather")

            from model_tools.analysis import analyze_all_chains
            min_len, max_len, mean_len, all_lengths = analyze_all_chains(base)

        assert min_len == 5
        assert max_len == 20
        assert mean_len == (10 + 5 + 20) / 3
        assert len(all_lengths) == 3

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from model_tools.analysis import analyze_all_chains
            min_len, max_len, mean_len, all_lengths = analyze_all_chains(Path(tmpdir))
        assert min_len == 0
        assert max_len == 0
        assert mean_len == 0.0
        assert all_lengths == {}
