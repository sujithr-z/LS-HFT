# =============================================================================
# src/features/volatility.py
#
# Mathematical specification
# ──────────────────────────
#
# 6.1  Sample mean of log returns over a window of size n:
#     r̄_t = (1/n) × Σ_{i=t-n+1}^{t} r_i
#
# 6.2  Sample standard deviation (Bessel-corrected, ddof=1):
#     s = sqrt( (1/(n-1)) × Σ_{i=1}^{n} (r_i - r̄)² )
#
# 6.3  Rolling volatility:
#     σ_t^(n) = std( r_{t-n+1}, ..., r_t )   using ddof=1
#
# 6.4  Annualised volatility:
#     σ_annual = σ_period × sqrt(N)
#     where N = number of observations per year (from config).
#     "1d" → 252,  "1wk" → 52,  "1mo" → 12.
#     The programmer must not blindly hard-code 252.
#
# 6.5  Volatility regime (analysis only — not a trading rule):
#     low    : σ_t  < Q_{0.33}
#     medium : Q_{0.33} ≤ σ_t < Q_{0.67}
#     high   : σ_t ≥ Q_{0.67}
#
# All computations use log returns (r_t), not simple returns.
# =============================================================================

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config.config import VOLATILITY_WINDOW, TRADING_PERIODS_PER_YEAR

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rolling_volatility(
    log_returns: pd.Series,
    window: int = VOLATILITY_WINDOW,
) -> pd.Series:
    """
    σ_t^(n) = std( r_{t-n+1}, ..., r_t )  with ddof=1

    Inputs:
        log_returns : pd.Series of r_t
        window      : n (integer ≥ 2; need at least 2 obs for ddof=1)

    Output:
        pd.Series of σ_t.  First (window-1) values are NaN.

    Invariant:
        If all returns in the window are identical (r_1 = r_2 = ... = r_n),
        then σ_t = 0.

    Invariant:
        σ_t ≥ 0  always.

    Numerical test:
        log_returns = [0.01, 0.01, 0.01, ...]  →  σ = 0
    """
    if window < 2:
        raise ValueError(f"Window must be ≥ 2 for ddof=1 standard deviation.  Got {window}.")

    sigma = log_returns.rolling(window=window, min_periods=window).std(ddof=1)
    sigma.name = f"rolling_vol_{window}"
    return sigma


def annualised_volatility(
    rolling_vol: pd.Series,
    periods_per_year: int = TRADING_PERIODS_PER_YEAR,
) -> pd.Series:
    """
    σ_annual = σ_period × sqrt(N)

    where N = periods_per_year (252 for daily, 52 weekly, 12 monthly).

    Inputs:
        rolling_vol     : pd.Series of σ_t^(n)  (from rolling_volatility)
        periods_per_year: N (derived from config.INTERVAL, not hard-coded here)

    Output:
        pd.Series of σ_t_annual.

    Units:
        If σ_period is the std of daily log returns (dimensionless fraction),
        then σ_annual is the annualised vol in the same units.

    Mathematical justification:
        Under i.i.d. returns, variance scales linearly with time:
            Var(r_{1..N}) = N × Var(r)
        Therefore std scales as sqrt(N).
    """
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be > 0.  Got {periods_per_year}.")

    ann = rolling_vol * np.sqrt(periods_per_year)
    ann.name = f"{rolling_vol.name}_annualised"
    return ann


def full_period_volatility(log_returns: pd.Series) -> float:
    """
    Scalar volatility over the entire sample:
        s = sqrt( (1/(N-1)) × Σ (r_i - r̄)² )

    Input:
        log_returns : pd.Series of r_t (NaN values are dropped)

    Output:
        float  (scalar)

    Invariant:
        If every return is identical → returns 0.0.
        Requires at least 2 non-NaN observations.
    """
    valid = log_returns.dropna()
    if len(valid) < 2:
        raise ValueError("Need at least 2 non-NaN observations for volatility.")
    return float(valid.std(ddof=1))


def volatility_regime(
    rolling_vol: pd.Series,
    low_quantile: float = 0.33,
    high_quantile: float = 0.67,
) -> pd.Series:
    """
    Classify each σ_t into a volatility regime.

    Thresholds:
        Q_{low_quantile}  and  Q_{high_quantile}
        computed from the non-NaN portion of rolling_vol.

    Classification:
        σ_t  < Q_{0.33}              → "low"
        Q_{0.33} ≤ σ_t < Q_{0.67}   → "medium"
        σ_t ≥ Q_{0.67}               → "high"

    Inputs:
        rolling_vol   : pd.Series of σ_t
        low_quantile  : lower quantile boundary (default 0.33)
        high_quantile : upper quantile boundary (default 0.67)

    Output:
        pd.Series of str   {"low", "medium", "high", NaN}

    Note: This is a descriptive analysis feature — NOT a trading signal.
    """
    valid = rolling_vol.dropna()
    q_low  = valid.quantile(low_quantile)
    q_high = valid.quantile(high_quantile)

    def _classify(v):
        if pd.isna(v):
            return float("nan")
        if v < q_low:
            return "low"
        if v < q_high:
            return "medium"
        return "high"

    regime = rolling_vol.map(_classify)
    regime.name = "vol_regime"
    return regime


def compute_all(
    log_returns: pd.Series,
    window: int = VOLATILITY_WINDOW,
    periods_per_year: int = TRADING_PERIODS_PER_YEAR,
) -> pd.DataFrame:
    """
    Compute all volatility features and return as a single DataFrame.

    Columns produced:
        rolling_vol_{window}
        rolling_vol_{window}_annualised
        vol_regime
    """
    rv  = rolling_volatility(log_returns, window)
    ann = annualised_volatility(rv, periods_per_year)
    reg = volatility_regime(rv)

    df = pd.DataFrame({
        rv.name:  rv,
        ann.name: ann,
        "vol_regime": reg,
    })

    logger.info(
        f"Volatility computed: {len(df)} rows  window={window}  "
        f"annualisation_factor=√{periods_per_year}"
    )
    return df
