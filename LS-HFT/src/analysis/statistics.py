# =============================================================================
# src/analysis/statistics.py
#
# Mathematical specification
# ──────────────────────────
#
# This module describes the empirical distribution of log returns.
# Input: r = {r_1, r_2, ..., r_N}  (from features/returns.py)
#
# 9.1  Mean
#     μ = (1/N) × Σ r_i
#
# 9.2  Variance  (sample, ddof=1)
#     s² = (1/(N-1)) × Σ (r_i - μ)²
#
# 9.3  Standard deviation
#     s = sqrt(s²)
#
# 9.4  Median
#     Middle value of sorted sample.
#
# 9.5  Quantiles
#     Q_q = F^{-1}(q)  where F is the empirical CDF.
#     Computed at: 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99
#
# 9.6  Skewness
#     γ_1 = E[(R - μ)³] / σ³
#     Implementation: scipy.stats.skew (Fisher's definition, bias=False)
#     Positive → longer right tail.
#     Negative → longer left tail.
#
# 9.7  Kurtosis
#     κ = E[(R - μ)⁴] / σ⁴
#     Implementation: scipy.stats.kurtosis with fisher=True  (excess kurtosis)
#     Excess kurtosis = kurtosis - 3
#     Normal distribution has excess kurtosis = 0.
#     Heavy-tailed → excess kurtosis > 0.
#     !! We explicitly report EXCESS kurtosis and say so in the output. !!
#
# 9.8  Correlation
#     ρ_{XY} = Cov(X, Y) / (σ_X × σ_Y)   ∈ [-1, 1]
# =============================================================================

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

QUANTILE_LEVELS: list[float] = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]


# ---------------------------------------------------------------------------
# Public API — individual statistics
# ---------------------------------------------------------------------------

def mean_return(returns: pd.Series) -> float:
    """μ = (1/N) × Σ r_i"""
    return float(returns.dropna().mean())


def variance_return(returns: pd.Series) -> float:
    """s² = (1/(N-1)) × Σ (r_i - μ)²   (ddof=1)"""
    return float(returns.dropna().var(ddof=1))


def std_return(returns: pd.Series) -> float:
    """s = sqrt(s²)"""
    return float(returns.dropna().std(ddof=1))


def median_return(returns: pd.Series) -> float:
    """Middle value of the sorted empirical distribution."""
    return float(returns.dropna().median())


def quantiles(
    returns: pd.Series,
    levels: list[float] = QUANTILE_LEVELS,
) -> dict[float, float]:
    """
    Q_q = F^{-1}(q) for each q in levels.

    Returns a dict: { q: value }
    """
    valid = returns.dropna()
    return {q: float(valid.quantile(q)) for q in levels}


def skewness(returns: pd.Series) -> float:
    """
    γ_1 = E[(R - μ)³] / σ³

    Uses scipy.stats.skew with bias=False (sample skewness).

    Interpretation:
        > 0  →  positive skew (right tail longer)
        < 0  →  negative skew (left tail longer, common in equities)
        = 0  →  symmetric distribution
    """
    return float(stats.skew(returns.dropna(), bias=False))


def excess_kurtosis(returns: pd.Series) -> float:
    """
    EXCESS kurtosis  =  κ - 3   =  E[(R-μ)⁴]/σ⁴  - 3

    Uses scipy.stats.kurtosis with fisher=True (fisher=True subtracts 3).

    Explicit documentation:
        This function returns EXCESS kurtosis, NOT raw kurtosis.
        Normal distribution → excess kurtosis = 0.
        Heavy tails         → excess kurtosis > 0.

    Interpretation:
        > 0  →  leptokurtic (fat tails), typical for daily equity returns
        < 0  →  platykurtic (thin tails)
        = 0  →  mesokurtic (normal-like tails)
    """
    return float(stats.kurtosis(returns.dropna(), fisher=True, bias=False))


def correlation(x: pd.Series, y: pd.Series) -> float:
    """
    ρ_{XY} = Cov(X, Y) / (σ_X × σ_Y) ∈ [-1, 1]

    Inputs:
        x, y : pd.Series (must share at least 2 common non-NaN indices)

    Output:
        float ∈ [-1, 1]
    """
    combined = pd.concat([x, y], axis=1).dropna()
    if len(combined) < 2:
        raise ValueError("Need at least 2 overlapping non-NaN observations.")
    return float(combined.iloc[:, 0].corr(combined.iloc[:, 1]))


# ---------------------------------------------------------------------------
# Full distribution report
# ---------------------------------------------------------------------------

def describe(returns: pd.Series) -> dict[str, Any]:
    """
    Compute the full empirical distribution summary.

    Returns a dict with:
        n                    : number of observations
        mean                 : μ
        std                  : s
        variance             : s²
        median               : Q_{0.50}
        skewness             : γ_1  (sample, unbiased)
        excess_kurtosis      : κ - 3  (EXCESS, explicitly stated)
        quantiles            : dict of {q: value}
        min                  : minimum observed return
        max                  : maximum observed return
        annualised_return    : μ × 252  (informational, uses full-sample mean)
    """
    valid = returns.dropna()
    n = len(valid)

    if n < 2:
        raise ValueError("Need at least 2 non-NaN return observations.")

    q_vals = quantiles(valid)

    result = {
        "n":                 n,
        "mean":              mean_return(valid),
        "std":               std_return(valid),
        "variance":          variance_return(valid),
        "median":            median_return(valid),
        "skewness":          skewness(valid),
        "excess_kurtosis":   excess_kurtosis(valid),
        "kurtosis_note":     "excess kurtosis reported (normal = 0)",
        "quantiles":         q_vals,
        "min":               float(valid.min()),
        "max":               float(valid.max()),
    }

    logger.info(
        f"Statistics computed on {n} observations: "
        f"μ={result['mean']:.6f}  σ={result['std']:.6f}  "
        f"skew={result['skewness']:.4f}  exc_kurt={result['excess_kurtosis']:.4f}"
    )
    return result


def print_summary(stats_dict: dict[str, Any], label: str = "Returns") -> None:
    """Pretty-print the describe() output."""
    print("\n" + "=" * 55)
    print(f"  STATISTICAL SUMMARY — {label}")
    print("=" * 55)
    print(f"  Observations         : {stats_dict['n']:>10,}")
    print(f"  Mean                 : {stats_dict['mean']:>14.6f}")
    print(f"  Std deviation        : {stats_dict['std']:>14.6f}")
    print(f"  Variance             : {stats_dict['variance']:>14.8f}")
    print(f"  Median               : {stats_dict['median']:>14.6f}")
    print(f"  Skewness             : {stats_dict['skewness']:>14.4f}")
    print(f"  Excess kurtosis      : {stats_dict['excess_kurtosis']:>14.4f}  ← excess (normal=0)")
    print(f"  Min return           : {stats_dict['min']:>14.6f}")
    print(f"  Max return           : {stats_dict['max']:>14.6f}")
    print()
    print("  Quantiles:")
    for q, v in stats_dict["quantiles"].items():
        print(f"    Q_{q:.2f}             : {v:>14.6f}")
    print("=" * 55 + "\n")
