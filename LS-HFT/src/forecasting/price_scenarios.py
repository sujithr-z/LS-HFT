# =============================================================================
# src/forecasting/price_scenarios.py
#
# Mathematical specification
# ──────────────────────────
#
# Scenarios are NOT arbitrary numbers like "10% up, 10% down".
# They must be derived from the fitted return distribution.
#
# At horizon h, the price distribution is log-normal:
#     P_{t+h} ~ LogNormal(ln(P_t) + μ_h,  σ_h²)
#
# Scenario definitions from distribution quantiles:
#     Bear : Q_{0.10}  — price below which 10% of outcomes fall
#     Base : Q_{0.50}  — median outcome
#     Bull : Q_{0.90}  — price above which 10% of outcomes fall
#
# Inverse log-normal quantile:
#     P_q = P_t × exp( μ_h + σ_h × Φ^{-1}(q) )
#
# where Φ^{-1}(q) is the standard normal quantile (ppf).
#
# This is principled because:
#     P(P_{t+h} ≤ P_q) = q  (by construction)
# =============================================================================

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.forecasting.return_distribution import FittedDistribution


# ---------------------------------------------------------------------------
# Scenario dataclass
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    """
    One price scenario at a given horizon.

    Attributes:
        label          : "bear" | "base" | "bull"
        quantile       : q  (e.g., 0.10, 0.50, 0.90)
        price_usd      : P_q = P_t × exp(μ_h + σ_h × Φ^{-1}(q))
        return_pct     : (P_q / P_t - 1)
        pnl_usd        : Q × (P_q - P_0)
        pnl_inr        : pnl_usd × inr_usd_rate
    """
    label      : str
    quantile   : float
    price_usd  : float
    return_pct : float
    pnl_usd    : float
    pnl_inr    : float


@dataclass
class ScenarioSet:
    """
    Bear / Base / Bull scenarios for a specific horizon.

    Attributes:
        horizon_h    : number of periods (hours)
        horizon_label: human-readable string
        bear         : Scenario at Q_{0.10}
        base         : Scenario at Q_{0.50}
        bull         : Scenario at Q_{0.90}
    """
    horizon_h    : int
    horizon_label: str
    bear         : Scenario
    base         : Scenario
    bull         : Scenario


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_scenarios(
    dist: FittedDistribution,
    h: int,
    horizon_label: str,
    p_current: float,
    p_entry: float,
    quantity: float,
    inr_usd_rate: float,
    bear_q: float = 0.10,
    base_q: float = 0.50,
    bull_q: float = 0.90,
) -> ScenarioSet:
    """
    Build the (bear, base, bull) scenario set for horizon h.

    Mathematical foundation:
        P_q = P_t × exp( μ_h + σ_h × Φ^{-1}(q) )

    Inputs:
        dist          : FittedDistribution from historical data
        h             : horizon in periods (hours for BERA)
        horizon_label : e.g., "7 days"
        p_current     : current market price (USD)
        p_entry       : average entry price P_0 (USD)
        quantity      : Q (number of tokens)
        inr_usd_rate  : ₹ per USD
        bear_q        : bear scenario quantile (default 0.10)
        base_q        : base scenario quantile (default 0.50)
        bull_q        : bull scenario quantile (default 0.90)

    Returns:
        ScenarioSet
    """
    mu_h, sigma_h = dist.scale(h)

    def _make(label: str, q: float) -> Scenario:
        # Inverse log-normal quantile
        # P_q = P_t × exp( μ_h + σ_h × Φ^{-1}(q) )
        z      = stats.norm.ppf(q)           # Φ^{-1}(q)
        log_r  = mu_h + sigma_h * z
        p_q    = p_current * np.exp(log_r)
        ret    = p_q / p_current - 1.0
        pnl    = quantity * (p_q - p_entry)
        return Scenario(
            label=label,
            quantile=q,
            price_usd=p_q,
            return_pct=ret,
            pnl_usd=pnl,
            pnl_inr=pnl * inr_usd_rate,
        )

    return ScenarioSet(
        horizon_h=h,
        horizon_label=horizon_label,
        bear=_make("bear", bear_q),
        base=_make("base", base_q),
        bull=_make("bull", bull_q),
    )
