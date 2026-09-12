# =============================================================================
# tests/test_risk.py
#
# Tests for src/analysis/risk.py
#
# Each test validates a MATHEMATICAL INVARIANT from the specification.
# Numerical examples are taken directly from the spec (section 11.1).
# =============================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.risk import (
    wealth_curve,
    drawdown,
    max_drawdown,
    historical_var,
    expected_shortfall,
    downside_deviation,
)
from src.features.returns import simple_return


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _returns_from_prices(prices: list[float]) -> pd.Series:
    """Convert a list of prices to a simple-return Series."""
    idx = pd.date_range("2020-01-01", periods=len(prices), freq="B")
    s = pd.Series(prices, index=idx, dtype=float)
    return s.pct_change()


# ---------------------------------------------------------------------------
# Test 3 — Drawdown  (Spec section 11.1)
# ---------------------------------------------------------------------------

class TestDrawdown:

    def test_spec_numerical_example(self):
        """
        Spec Test 3:
            Wealth = [100, 110, 105, 90, 95]
            Peak   = [100, 110, 110, 110, 110]
            D_t    = [0,   0,   4.545%, 18.182%, 13.636%]
            MDD    = 18.182%
        """
        wealth_vals = [100.0, 110.0, 105.0, 90.0, 95.0]
        idx = pd.date_range("2020-01-01", periods=len(wealth_vals), freq="B")
        w = pd.Series(wealth_vals, index=idx, name="wealth")

        dd = drawdown(w)

        expected_dd = [0.0, 0.0, 1 - 105/110, 1 - 90/110, 1 - 95/110]
        for i, (computed, expected) in enumerate(zip(dd.values, expected_dd)):
            assert abs(computed - expected) < 1e-6, \
                f"D_{i}: expected {expected:.6f}, got {computed:.6f}"

        mdd = max_drawdown(dd)
        expected_mdd = 1 - 90/110
        assert abs(mdd - expected_mdd) < 1e-6, \
            f"MDD: expected {expected_mdd:.6f}, got {mdd:.6f}"

    def test_drawdown_is_nonnegative(self):
        """D_t ≥ 0 always."""
        prices = [100, 110, 120, 115, 105, 125, 130, 120]
        ret = _returns_from_prices(prices)
        w = wealth_curve(ret)
        dd = drawdown(w)
        assert (dd >= -1e-10).all(), f"Negative drawdown found: {dd.min()}"

    def test_drawdown_at_alltime_high_is_zero(self):
        """D_t = 0 when W_t achieves a new all-time high."""
        prices = [100, 110, 120, 130]   # strictly increasing
        ret = _returns_from_prices(prices)
        w = wealth_curve(ret)
        dd = drawdown(w)
        # After the very first point, all subsequent are new highs → D_t ≈ 0
        assert (dd.dropna().abs() < 1e-10).all()

    def test_mdd_bounds(self):
        """MDD ∈ [0, 1]."""
        prices = [100, 80, 60, 40, 50, 70]
        ret = _returns_from_prices(prices)
        w = wealth_curve(ret)
        dd = drawdown(w)
        mdd = max_drawdown(dd)
        assert 0.0 <= mdd <= 1.0


# ---------------------------------------------------------------------------
# VaR tests  (Spec section 12)
# ---------------------------------------------------------------------------

class TestHistoricalVaR:

    def test_var_is_nonnegative(self):
        """VaR must be non-negative (it represents a loss magnitude)."""
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.01, 500))
        var_val = historical_var(returns, confidence=0.95)
        assert var_val >= 0.0

    def test_var_zero_for_all_positive_returns(self):
        """If every return is positive, VaR at 95% = 0."""
        returns = pd.Series([0.01] * 100)
        var_val = historical_var(returns, confidence=0.95)
        assert var_val == 0.0

    def test_var_quantile_relationship(self):
        """
        VaR_{95} = -Q_{0.05}
        Verify the mathematical relationship holds exactly.
        """
        rng = np.random.default_rng(7)
        returns = pd.Series(rng.normal(-0.001, 0.015, 1000))
        var_val = historical_var(returns, confidence=0.95)
        q005 = float(returns.quantile(0.05))
        expected = max(-q005, 0.0)
        assert abs(var_val - expected) < 1e-10


# ---------------------------------------------------------------------------
# ES tests  (Spec section 13)
# ---------------------------------------------------------------------------

class TestExpectedShortfall:

    def test_es_geq_var(self):
        """
        Mathematical invariant:
            ES_α ≥ VaR_α  always.
        ES measures the tail average; VaR measures the tail boundary.
        """
        rng = np.random.default_rng(13)
        returns = pd.Series(rng.normal(0, 0.02, 500))
        var_val = historical_var(returns, confidence=0.95)
        es_val  = expected_shortfall(returns, confidence=0.95)
        assert es_val >= var_val - 1e-10, \
            f"ES={es_val:.6f} < VaR={var_val:.6f} — invariant violated"

    def test_es_is_nonnegative(self):
        rng = np.random.default_rng(99)
        returns = pd.Series(rng.normal(0, 0.01, 200))
        es_val = expected_shortfall(returns, confidence=0.95)
        assert es_val >= 0.0


# ---------------------------------------------------------------------------
# Downside deviation tests  (Spec section 14)
# ---------------------------------------------------------------------------

class TestDownsideDeviation:

    def test_zero_for_all_nonneg_returns(self):
        """
        If R_t ≥ T = 0 for all t  →  σ_down = 0.
        """
        returns = pd.Series([0.01, 0.02, 0.005, 0.03])
        sigma_down = downside_deviation(returns, target=0.0)
        assert abs(sigma_down) < 1e-10

    def test_nonnegative(self):
        """σ_down ≥ 0 always."""
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0, 0.02, 100))
        sigma_down = downside_deviation(returns, target=0.0)
        assert sigma_down >= 0.0

    def test_manual_calculation(self):
        """
        Manual check:
            returns = [-0.02, 0.01, -0.03, 0.02],  T = 0
            D = [min(-0.02, 0), min(0.01, 0), min(-0.03, 0), min(0.02, 0)]
              = [-0.02, 0, -0.03, 0]
            σ_down = sqrt( (0.0004 + 0 + 0.0009 + 0) / 4 )
                   = sqrt(0.000325)
                   ≈ 0.018028
        """
        returns = pd.Series([-0.02, 0.01, -0.03, 0.02])
        expected = np.sqrt((0.02**2 + 0.03**2) / 4)
        result = downside_deviation(returns, target=0.0)
        assert abs(result - expected) < 1e-10, \
            f"Expected {expected:.8f}, got {result:.8f}"


# ---------------------------------------------------------------------------
# Wealth curve tests  (Spec section 10.1)
# ---------------------------------------------------------------------------

class TestWealthCurve:

    def test_wealth_starts_at_w0(self):
        """W_0 is the starting value."""
        prices = [100.0, 110.0, 105.0]
        ret = _returns_from_prices(prices)
        w = wealth_curve(ret, w0=1000.0)
        assert abs(float(w.iloc[0]) - 1000.0) < 1e-10

    def test_flat_returns_constant_wealth(self):
        """If all R_t = 0, W_t = W_0 for all t."""
        returns = pd.Series([0.0] * 10)
        returns.index = pd.date_range("2020-01-01", periods=10, freq="B")
        w = wealth_curve(returns, w0=1.0)
        assert (abs(w - 1.0) < 1e-10).all()
