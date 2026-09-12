# =============================================================================
# src/analysis/risk.py
#
# Mathematical specification
# ──────────────────────────
#
# 10.1  Wealth curve
#     W_0 = initial capital (from config: INITIAL_WEALTH = 1.0 normalised)
#     W_t = W_{t-1} × (1 + R_t)
#         = W_0 × Π_{i=1}^{t} (1 + R_i)
#
# 11.  Drawdown (positive magnitude representation)
#     Running peak:   M_t = max_{0 ≤ i ≤ t} W_i
#     Drawdown:       D_t = 1 - W_t / M_t   ∈ [0, 1]
#     D_t = 0  →  W_t is at or above all previous peaks
#     D_t = 1  →  total loss of capital (W_t = 0)
#
# 11.1  Maximum drawdown
#     MDD = max_t D_t
#
# 12.  Historical VaR at confidence α
#     VaR_α = -Q_{1-α}  (positive number representing loss magnitude)
#     For α = 0.95:
#         VaR_{95} = -Q_{0.05}
#     Interpretation: ~5% of days had losses exceeding VaR_{95}.
#     !! VaR is NOT the maximum possible loss. !!
#
# 13.  Expected Shortfall (ES / CVaR) at confidence α
#     ES_α = -E[R | R ≤ Q_{1-α}]
#     For α = 0.95:
#         ES_{95} = -E[R | R ≤ Q_{0.05}]
#     ES answers: given that we're in the worst (1-α)% tail,
#                 what is the average loss?
#
# 14.  Downside deviation
#     Target return T = DOWNSIDE_TARGET (default 0.0)
#     D_t = min(R_t - T, 0)
#     σ_down = sqrt( (1/N) × Σ D_t² )    (uses N, not N-1)
# =============================================================================

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config.config import (
    VAR_CONFIDENCE,
    ES_CONFIDENCE,
    DOWNSIDE_TARGET,
    INITIAL_WEALTH,
    TRADING_PERIODS_PER_YEAR,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def wealth_curve(
    simple_returns: pd.Series,
    w0: float = INITIAL_WEALTH,
) -> pd.Series:
    """
    W_t = W_0 × Π_{i=1}^{t} (1 + R_i)

    Inputs:
        simple_returns : pd.Series of R_t
        w0             : starting capital (default 1.0 normalised)

    Output:
        pd.Series of W_t with same index as simple_returns.
        W_0 prepended at the first available index (set to w0).

    Mathematical invariant:
        W_t / W_0 - 1  = cumulative_return at time t
    """
    valid = simple_returns.dropna()
    wt = w0 * (1 + valid).cumprod()
    # Prepend W_0 at the first observed date so drawdown calculation starts at 0
    w0_entry = pd.Series([w0], index=[valid.index[0]], name="wealth")
    wt = pd.concat([w0_entry, wt.iloc[1:]])
    wt.name = "wealth"
    return wt


def drawdown(wealth: pd.Series) -> pd.Series:
    """
    D_t = 1 - W_t / M_t   where  M_t = max_{0 ≤ i ≤ t} W_i

    Inputs:
        wealth : pd.Series of W_t (from wealth_curve)

    Output:
        pd.Series of D_t ∈ [0, 1].
        D_t = 0  →  new all-time high (or W_t at M_t).
        D_t = 1  →  total loss.

    Mathematical invariants:
        D_t ≥ 0  always
        D_t ≤ 1  always
        D_t = 0  iff  W_t = M_t

    Numerical test (from spec section 11.1):
        wealth = [1000, 1100, 1050, 900, 950]
        peak   = [1000, 1100, 1100, 1100, 1100]
        D_t    = [0,    0,    ~4.5%, ~18.2%, ~13.6%]
    """
    peak = wealth.cummax()
    dd = 1 - wealth / peak
    dd.name = "drawdown"
    return dd


def max_drawdown(dd: pd.Series) -> float:
    """
    MDD = max_t D_t

    Input: pd.Series of D_t ∈ [0, 1]
    Output: float ∈ [0, 1]

    Numerical test:
        wealth = [1000, 1100, 1050, 900, 950]
        MDD = 1 - 900/1100 ≈ 0.1818  (18.18%)
    """
    return float(dd.max())


def historical_var(
    returns: pd.Series,
    confidence: float = VAR_CONFIDENCE,
) -> float:
    """
    Historical VaR at confidence level α.

    VaR_α = -Q_{1-α}

    Interpretation:
        VaR_{0.95} = -Q_{0.05}
        In approximately (1-α)×100% = 5% of observations,
        the loss exceeded VaR.

    Inputs:
        returns    : pd.Series of R_t
        confidence : α ∈ (0, 1)  (default 0.95)

    Output:
        float  (positive number representing loss)

    IMPORTANT: VaR is NOT the maximum possible loss.
        It is the (1-α) quantile of the loss distribution.

    Edge case:
        If all returns are non-negative, VaR = 0 (no tail loss at that level).
    """
    valid = returns.dropna()
    tail_quantile = 1 - confidence     # 0.05 for α=0.95
    q = float(valid.quantile(tail_quantile))
    var_value = -q                     # convention: positive = loss
    return max(var_value, 0.0)         # VaR is non-negative by definition


def expected_shortfall(
    returns: pd.Series,
    confidence: float = ES_CONFIDENCE,
) -> float:
    """
    ES_α = -E[R | R ≤ Q_{1-α}]

    For α = 0.95:
        ES_{95} = -E[R | R ≤ Q_{0.05}]

    Interpretation:
        VaR asks: where does the bad (1-α)% tail begin?
        ES asks:  given we are inside that tail, what is the average loss?
        ES is always ≥ VaR  (it measures the tail average, not the boundary).

    Inputs:
        returns    : pd.Series of R_t
        confidence : α ∈ (0, 1)

    Output:
        float  (positive number)

    Mathematical invariant:
        ES_α ≥ VaR_α  always
    """
    valid = returns.dropna()
    tail_quantile = 1 - confidence
    cutoff = float(valid.quantile(tail_quantile))
    tail_returns = valid[valid <= cutoff]

    if len(tail_returns) == 0:
        logger.warning("No tail observations found for ES calculation — returning 0.0.")
        return 0.0

    es_value = -float(tail_returns.mean())
    return max(es_value, 0.0)


def downside_deviation(
    returns: pd.Series,
    target: float = DOWNSIDE_TARGET,
) -> float:
    """
    σ_down = sqrt( (1/N) × Σ min(R_t - T, 0)² )

    Inputs:
        returns : pd.Series of R_t
        target  : T, the minimum acceptable return (default 0.0)

    Output:
        float ≥ 0

    Interpretation:
        Measures variability of returns ONLY on the downside.
        Returns above T are treated as contributing zero.

    If T = 0:
        We penalise only negative returns.

    Mathematical invariant:
        σ_down = 0  iff  R_t ≥ T  for all t.
    """
    valid = returns.dropna()
    downside = np.minimum(valid - target, 0.0)
    sigma_down = float(np.sqrt(np.mean(downside ** 2)))
    return sigma_down


# ---------------------------------------------------------------------------
# Full risk report
# ---------------------------------------------------------------------------

def risk_report(
    simple_returns: pd.Series,
    log_returns: pd.Series,
    var_confidence: float = VAR_CONFIDENCE,
    es_confidence: float  = ES_CONFIDENCE,
    downside_target: float = DOWNSIDE_TARGET,
    w0: float = INITIAL_WEALTH,
    periods_per_year: int = TRADING_PERIODS_PER_YEAR,
) -> dict[str, Any]:
    """
    Produce the complete V0 risk summary.

    Returns a dict:
        total_observations
        mean_return              : E[R_t]
        return_volatility        : std(R_t)  daily
        annualised_volatility    : std(R_t) × sqrt(N)
        cumulative_return        : W_T / W_0 - 1
        max_drawdown             : MDD
        var_95                   : Historical VaR at var_confidence
        expected_shortfall_95    : ES at es_confidence
        downside_deviation       : σ_down  (target = downside_target)
        worst_return             : min(R_t)
        best_return              : max(R_t)
        wealth_series            : pd.Series of W_t
        drawdown_series          : pd.Series of D_t
    """
    valid_simple = simple_returns.dropna()
    valid_log    = log_returns.dropna()

    n = len(valid_simple)
    if n < 2:
        raise ValueError("Need at least 2 return observations for risk analysis.")

    # Wealth and drawdown series
    w = wealth_curve(valid_simple, w0)
    dd = drawdown(w)

    # Scalar risk measures
    cum_ret   = float(w.iloc[-1] / w0 - 1)
    mdd       = max_drawdown(dd)
    vol_daily = float(valid_log.std(ddof=1))
    vol_ann   = vol_daily * np.sqrt(periods_per_year)
    var_val   = historical_var(valid_simple, var_confidence)
    es_val    = expected_shortfall(valid_simple, es_confidence)
    dd_val    = downside_deviation(valid_simple, downside_target)

    report = {
        "total_observations":       n,
        "mean_return":              float(valid_simple.mean()),
        "return_volatility_daily":  vol_daily,
        "annualised_volatility":    vol_ann,
        "cumulative_return":        cum_ret,
        "max_drawdown":             mdd,
        f"var_{int(var_confidence*100)}":  var_val,
        f"es_{int(es_confidence*100)}":    es_val,
        "downside_deviation":       dd_val,
        "worst_return":             float(valid_simple.min()),
        "best_return":              float(valid_simple.max()),
        # Series for visualisation
        "wealth_series":            w,
        "drawdown_series":          dd,
    }

    logger.info(
        f"Risk report: MDD={mdd:.2%}  VaR={var_val:.4f}  ES={es_val:.4f}  "
        f"AnnVol={vol_ann:.4f}"
    )
    return report


def print_report(report: dict[str, Any]) -> None:
    """Pretty-print the risk report."""
    conf_v = next((k for k in report if k.startswith("var_")), "var_95")
    conf_e = next((k for k in report if k.startswith("es_")),  "es_95")

    print("\n" + "=" * 55)
    print("  RISK REPORT  (V0 Historical)")
    print("=" * 55)
    print(f"  Total observations       : {report['total_observations']:>8,}")
    print(f"  Mean return (daily)      : {report['mean_return']:>12.6f}")
    print(f"  Return volatility (daily): {report['return_volatility_daily']:>12.6f}")
    print(f"  Annualised volatility    : {report['annualised_volatility']:>12.4%}")
    print(f"  Cumulative return        : {report['cumulative_return']:>12.4%}")
    print(f"  Maximum drawdown         : {report['max_drawdown']:>12.4%}")
    print(f"  {conf_v.upper()} (historical)    : {report[conf_v]:>12.6f}")
    print(f"  {conf_e.upper()} (historical)    : {report[conf_e]:>12.6f}")
    print(f"  Downside deviation       : {report['downside_deviation']:>12.6f}")
    print(f"  Worst single return      : {report['worst_return']:>12.6f}")
    print(f"  Best single return       : {report['best_return']:>12.6f}")
    print("=" * 55 + "\n")
