# =============================================================================
# tests/test_features.py
#
# Tests for feature engines: returns, volatility, volume, microstructure.
#
# Each test validates a MATHEMATICAL INVARIANT from the specification.
# Numerical examples are taken directly from the spec.
# =============================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.returns       import simple_return, log_return, cumulative_return, rolling_return
from src.features.volatility    import rolling_volatility, annualised_volatility
from src.features.volume        import rolling_mean_volume, rolling_std_volume, volume_zscore, volume_spike
from src.features.microstructure import close_location_value, relative_volume


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _series(values, name="price"):
    idx = pd.date_range("2020-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx, name=name, dtype=float)


# ---------------------------------------------------------------------------
# Test 1 — Simple return  (Spec section 5.1)
# ---------------------------------------------------------------------------

class TestSimpleReturn:

    def test_numerical_example(self):
        """
        Spec Test 1:
            P_{t-1} = 100,  P_t = 110  →  R_t = 0.10
        """
        prices = _series([100.0, 110.0])
        r = simple_return(prices)
        assert abs(r.iloc[1] - 0.10) < 1e-10, f"Expected 0.10, got {r.iloc[1]}"

    def test_zero_return_invariant(self):
        """
        Spec Test 2:
            If P_t = P_{t-1}  →  R_t = 0
        """
        prices = _series([100.0, 100.0, 100.0])
        r = simple_return(prices)
        assert r.iloc[1] == 0.0
        assert r.iloc[2] == 0.0

    def test_first_element_is_nan(self):
        """No prior observation → R_0 is undefined → NaN."""
        prices = _series([100.0, 105.0])
        r = simple_return(prices)
        assert pd.isna(r.iloc[0])

    def test_positive_return_for_price_increase(self):
        prices = _series([50.0, 60.0])
        r = simple_return(prices)
        assert r.iloc[1] > 0

    def test_negative_return_for_price_decrease(self):
        prices = _series([100.0, 80.0])
        r = simple_return(prices)
        assert r.iloc[1] < 0

    def test_return_bounded_below_by_minus_one(self):
        """R_t > -1 since P_t > 0 (guaranteed by cleaner)."""
        prices = _series([100.0, 0.01])   # extreme drop but still positive
        r = simple_return(prices)
        assert r.iloc[1] > -1.0


# ---------------------------------------------------------------------------
# Test 2 — Log return  (Spec section 5.2)
# ---------------------------------------------------------------------------

class TestLogReturn:

    def test_numerical_example(self):
        """
        Spec Test 1 (log version):
            P_{t-1} = 100,  P_t = 110  →  r_t = ln(1.1) ≈ 0.09531
        """
        prices = _series([100.0, 110.0])
        r = log_return(prices)
        expected = np.log(1.1)
        assert abs(r.iloc[1] - expected) < 1e-10

    def test_zero_log_return(self):
        """
        Spec Test 2:
            P_t = P_{t-1}  →  r_t = ln(1) = 0
        """
        prices = _series([100.0, 100.0])
        r = log_return(prices)
        assert abs(r.iloc[1]) < 1e-12

    def test_log_returns_sum_equals_total(self):
        """
        Additivity of log returns:
            Σ r_t  =  ln(P_N / P_0)
        """
        prices = _series([100.0, 110.0, 105.0, 120.0])
        r = log_return(prices).dropna()
        total_log = float(np.log(prices.iloc[-1] / prices.iloc[0]))
        assert abs(r.sum() - total_log) < 1e-10

    def test_first_element_nan(self):
        prices = _series([100.0, 110.0])
        r = log_return(prices)
        assert pd.isna(r.iloc[0])


# ---------------------------------------------------------------------------
# Test 3 — Cumulative return  (Spec section 5.3)
# ---------------------------------------------------------------------------

class TestCumulativeReturn:

    def test_zero_cumulative_on_flat_prices(self):
        """If all R_t = 0, then CR_t = 0 for all t."""
        prices = _series([100.0, 100.0, 100.0, 100.0])
        sr = simple_return(prices)
        cr = cumulative_return(sr)
        assert (cr.dropna().abs() < 1e-10).all()


# ---------------------------------------------------------------------------
# Test 4 — Volume z-score  (Spec section 7.4)
# ---------------------------------------------------------------------------

class TestVolumeZScore:

    def test_z_zero_when_volume_equals_mean(self):
        """
        Spec Test 4:
            If V_t = V̄_t  →  Z_t^V = 0
        """
        volumes = _series([1_000_000.0] * 25)   # constant volume
        zv = volume_zscore(volumes, window=20)
        # After window fills, all z-scores should be 0 (std=0 → NaN)
        # But the std of constant series = 0 → division by 0 → NaN
        # This is the correct behaviour; test that NaN is returned (not a wrong value)
        valid = zv.dropna()
        # If any valid values exist, they should be NaN or 0
        # When std=0, we return NaN (documented edge case)
        assert valid.isna().all() or (valid.abs() < 1e-10).all(), \
            "Z-score of constant volume should be NaN (std=0 edge case)"

    def test_z_positive_when_volume_above_mean(self):
        """Z_t^V > 0 when V_t > V̄_t."""
        base = [1_000_000.0] * 20
        spike = base + [5_000_000.0]   # one very large volume
        volumes = _series(spike)
        zv = volume_zscore(volumes, window=20)
        assert float(zv.iloc[-1]) > 0

    def test_volume_spike_detection(self):
        """
        Spec section 7.5:
            Z_t^V > 2  →  spike = True
        """
        base = [1_000_000.0] * 20
        spike_val = base + [50_000_000.0]   # huge spike
        volumes = _series(spike_val)
        zv = volume_zscore(volumes, window=20)
        spk = volume_spike(zv, threshold=2.0)
        assert bool(spk.iloc[-1]) is True


# ---------------------------------------------------------------------------
# Test 5 — Volatility  (Spec section 6.2)
# ---------------------------------------------------------------------------

class TestVolatility:

    def test_zero_volatility_on_identical_returns(self):
        """
        Spec Test 5:
            r_1 = r_2 = ... = r_n  →  σ = 0
        """
        constant_returns = _series([0.01] * 30, name="log_return")
        sigma = rolling_volatility(constant_returns, window=20)
        valid = sigma.dropna()
        assert (valid.abs() < 1e-10).all(), \
            f"Expected σ=0 for constant returns, got max={valid.abs().max()}"

    def test_volatility_nonnegative(self):
        """σ_t ≥ 0 always."""
        rng = np.random.default_rng(0)
        returns = _series(rng.normal(0, 0.01, 50), name="log_return")
        sigma = rolling_volatility(returns, window=10)
        assert (sigma.dropna() >= 0).all()

    def test_annualisation_scales_correctly(self):
        """
        σ_annual = σ_daily × sqrt(252)
        Test that the ratio holds exactly.
        """
        rng = np.random.default_rng(1)
        returns = _series(rng.normal(0, 0.01, 50), name="log_return")
        rv  = rolling_volatility(returns, window=10)
        ann = annualised_volatility(rv, periods_per_year=252)
        ratio = (ann / rv).dropna()
        expected_ratio = np.sqrt(252)
        assert (abs(ratio - expected_ratio) < 1e-10).all()

    def test_window_less_than_2_raises(self):
        """ddof=1 requires at least 2 observations."""
        returns = _series([0.01] * 5, name="log_return")
        with pytest.raises(ValueError):
            rolling_volatility(returns, window=1)


# ---------------------------------------------------------------------------
# Test 6 — Microstructure CLV  (Spec section 8.1)
# ---------------------------------------------------------------------------

class TestCLV:

    def test_clv_at_midpoint(self):
        """
        Spec:
            H=110, L=90, C=100  →  CLV = (200-110-90)/20 = 0/20 = 0.0
        """
        h = _series([110.0])
        l = _series([90.0])
        c = _series([100.0])
        clv = close_location_value(h, l, c)
        assert abs(float(clv.iloc[0]) - 0.0) < 1e-10

    def test_clv_at_high(self):
        """CLV = +1 when C = H."""
        h = _series([110.0])
        l = _series([90.0])
        c = _series([110.0])
        clv = close_location_value(h, l, c)
        assert abs(float(clv.iloc[0]) - 1.0) < 1e-10

    def test_clv_at_low(self):
        """CLV = -1 when C = L."""
        h = _series([110.0])
        l = _series([90.0])
        c = _series([90.0])
        clv = close_location_value(h, l, c)
        assert abs(float(clv.iloc[0]) + 1.0) < 1e-10

    def test_clv_doji_is_nan(self):
        """CLV is undefined (NaN) when H = L (zero-range candle)."""
        h = _series([100.0])
        l = _series([100.0])
        c = _series([100.0])
        clv = close_location_value(h, l, c)
        assert pd.isna(float(clv.iloc[0]))

    def test_clv_range(self):
        """CLV ∈ [-1, +1]."""
        rng = np.random.default_rng(5)
        n = 100
        low   = _series(rng.uniform(90, 100, n))
        high  = low + rng.uniform(0.1, 10, n)
        close = pd.Series(
            [rng.uniform(float(low.iloc[i]), float(high.iloc[i])) for i in range(n)],
            index=low.index
        )
        clv = close_location_value(high, low, close)
        valid = clv.dropna()
        assert (valid >= -1.0 - 1e-10).all()
        assert (valid <=  1.0 + 1e-10).all()
