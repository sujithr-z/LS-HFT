# =============================================================================
# tests/test_data.py
#
# Tests for data pipeline: downloader validation, loader schema,
# cleaner invariants.
#
# Philosophy: test mathematical INVARIANTS, not just whether Python runs.
# =============================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cleaner import clean, CleaningReport
from src.data.loader  import REQUIRED_COLUMNS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 10, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 + rng.standard_normal(n).cumsum()
    close = np.abs(close) + 50   # ensure positive

    df = pd.DataFrame({
        "Open":   close * (1 + rng.uniform(-0.005, 0.005, n)),
        "High":   close * (1 + rng.uniform(0.000, 0.010, n)),
        "Low":    close * (1 - rng.uniform(0.000, 0.010, n)),
        "Close":  close,
        "Volume": rng.integers(100_000, 1_000_000, n).astype(float),
    }, index=dates)
    df.index.name = "Date"
    return df


# ---------------------------------------------------------------------------
# Cleaner tests
# ---------------------------------------------------------------------------

class TestCleaner:

    def test_clean_returns_two_values(self):
        """clean() must always return a tuple (DataFrame, CleaningReport)."""
        df = _make_ohlcv()
        result = clean(df)
        assert isinstance(result, tuple), "clean() must return a tuple"
        assert len(result) == 2

    def test_clean_report_type(self):
        df = _make_ohlcv()
        _, report = clean(df)
        assert isinstance(report, CleaningReport)

    def test_clean_rows_output_leq_rows_input(self):
        """
        Cleaning can only remove rows, never add them.
        Invariant: rows_output <= rows_input
        """
        df = _make_ohlcv()
        _, report = clean(df)
        assert report.rows_output <= report.rows_input

    def test_clean_sorts_timestamps(self):
        """
        After cleaning, timestamps must be strictly ascending:
        t_1 < t_2 < ... < t_N
        """
        df = _make_ohlcv(20)
        shuffled = df.sample(frac=1, random_state=99)   # randomly shuffle
        cleaned, report = clean(shuffled)
        assert cleaned.index.is_monotonic_increasing, "Timestamps must be ascending after clean()"
        assert report.sort_applied is True

    def test_clean_removes_duplicates(self):
        """
        Duplicate timestamps must be removed and reported.
        Invariant 4.1: ∀ i ≠ j, t_i ≠ t_j in D_clean.
        """
        df = _make_ohlcv(10)
        df_dup = pd.concat([df, df.iloc[[3]]])   # add one duplicate
        cleaned, report = clean(df_dup)
        assert report.duplicates_removed == 1
        assert cleaned.index.is_unique

    def test_clean_removes_nan_prices(self):
        """
        Rows with NaN in any price column must be removed.
        Invariant 4.2: C_t ≠ NaN in D_clean.
        """
        df = _make_ohlcv(10)
        df.loc[df.index[5], "Close"] = float("nan")
        cleaned, report = clean(df)
        assert report.missing_rows_removed >= 1
        assert cleaned["Close"].isna().sum() == 0

    def test_clean_removes_nonpositive_prices(self):
        """
        Non-positive prices must be removed.
        Invariant 4.4: P_t > 0 in D_clean.
        Required because ln(P_t / P_{t-1}) is undefined for P_t <= 0.
        """
        df = _make_ohlcv(10)
        df.loc[df.index[4], "Close"] = 0.0
        cleaned, report = clean(df)
        assert report.nonpositive_removed >= 1
        assert (cleaned["Close"] > 0).all()

    def test_clean_preserves_good_data(self):
        """If the input is already clean, output equals input."""
        df = _make_ohlcv(10)
        cleaned, report = clean(df)
        assert report.duplicates_removed == 0
        assert report.missing_rows_removed == 0
        assert report.nonpositive_removed == 0
        assert report.rows_output == report.rows_input


# ---------------------------------------------------------------------------
# Loader schema tests
# ---------------------------------------------------------------------------

class TestLoaderSchema:

    def test_required_columns_constant(self):
        """REQUIRED_COLUMNS must contain the five canonical OHLCV fields."""
        expected = {"Open", "High", "Low", "Close", "Volume"}
        assert expected == set(REQUIRED_COLUMNS)

    def test_missing_column_raises(self):
        """
        Loader must fail loudly if a required column is absent.
        """
        from src.data.loader import _validate_schema
        df_incomplete = pd.DataFrame({"Open": [1], "High": [1], "Low": [1], "Close": [1]})
        with pytest.raises(ValueError, match="Volume"):
            _validate_schema(df_incomplete)
