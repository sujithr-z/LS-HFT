# =============================================================================
# src/position/earn.py
#
# Mathematical specification — Earn opportunity cost
# ─────────────────────────────────────────────────────
#
# The Earn product pays APR continuously on the BERA principal.
# If you EXIT now you FORFEIT future yield. That forfeited yield is
# the opportunity cost of exiting.
#
# Earn yield formula (compound interest, hourly periods):
#
#   Y_h = Q x P_t_inr x ((1 + APR)^(h/N) - 1)
#
#   where:
#     Q   = quantity of BERA (principal)
#     P_t_inr = current price of BERA in INR
#     APR = annual percentage rate  (e.g., 0.1382)
#     h   = holding horizon in hours
#     N   = EARN_PERIODS_PER_YEAR = 8760  (hours/year)
#
# Interpretation:
#   Y_h is the INR value of BERA you would earn over h hours
#   if you continue holding. This must be added to the price-based
#   EV to get the true earn-adjusted EV.
#
# Earn-adjusted expected P&L:
#   EV_earn(h) = EV_price(h) + Y_h - tx_cost_inr
#
# Break-even price (earn-adjusted):
#   The minimum future price at which holding for h hours is better
#   than exiting now, accounting for the earn yield:
#
#   P_break_inr = P_0_inr - Y_h/Q + tx_cost_exit_per_token
#
#   Because earn reduces the effective breakeven, the system should
#   show this as a lower bar for HOLD.
#
# Cumulative accrued interest already earned:
#   I_accrued = cumulative_interest_bera x P_t_inr   (INR value)
#
# Annualised effective yield (verify against stated APR):
#   APY = (1 + APR/N)^N - 1   (for compounding N times/year)
# =============================================================================

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config.config import EARN_APR, EARN_PERIODS_PER_YEAR, CUMULATIVE_INTEREST_BERA


# ---------------------------------------------------------------------------
# Earn yield dataclass — one per horizon
# ---------------------------------------------------------------------------

@dataclass
class EarnResult:
    """
    Earn yield metrics for one horizon.

    Attributes:
        horizon_h            : h in hours
        horizon_label        : e.g., "7 days"
        earn_tokens          : BERA earned over h hours   = Q x ((1+APR)^(h/N) - 1)
        earn_value_inr       : earn_tokens x P_t_inr
        earn_value_usd       : earn_value_inr / INR_USD_RATE
        effective_rate_h     : earn yield as % of position value at horizon h
        annualised_yield     : APY = (1+APR)^1 - 1  (same for all horizons, for reference)
        accrued_value_inr    : value of interest already accrued (not future)
    """
    horizon_h         : int
    horizon_label     : str
    earn_tokens       : float
    earn_value_inr    : float
    earn_value_usd    : float
    effective_rate_h  : float
    annualised_yield  : float
    accrued_value_inr : float


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

def compute_earn(
    quantity: float,
    current_price_inr: float,
    earn_apr: float             = EARN_APR,
    h: int                      = 24,
    horizon_label: str          = "24 hours",
    periods_per_year: int       = EARN_PERIODS_PER_YEAR,
    cumulative_interest: float  = CUMULATIVE_INTEREST_BERA,
    inr_usd_rate: float         = 84.0,
) -> EarnResult:
    """
    Compute the earn yield for holding for h more hours.

    Formula:
        earn_tokens    = Q x ((1 + APR)^(h/N) - 1)
        earn_value_inr = earn_tokens x P_t_inr
        Y_h            = earn_value_inr   (the INR gain from continuing to hold)

    Mathematical invariants:
        earn_tokens >= 0  always  (APR >= 0)
        earn_tokens = 0   iff  h = 0 or APR = 0
        earn_value_inr scales linearly with P_t_inr (price risk is shared)

    Edge cases:
        h = 0     -> earn_tokens = 0
        APR = 0   -> earn_tokens = 0
        N <= 0    -> raise ValueError
    """
    if periods_per_year <= 0:
        raise ValueError(f"EARN_PERIODS_PER_YEAR must be > 0.  Got {periods_per_year}.")
    if h < 0:
        raise ValueError(f"Horizon h must be >= 0.  Got {h}.")

    # Fractional period: h hours out of N hours/year
    growth_factor = (1.0 + earn_apr) ** (h / periods_per_year)
    earn_tokens   = quantity * (growth_factor - 1.0)
    earn_inr      = earn_tokens * current_price_inr
    earn_usd      = earn_inr / inr_usd_rate

    # Effective rate as % of current position value
    position_value_inr = quantity * current_price_inr
    eff_rate = earn_inr / position_value_inr if position_value_inr > 0 else 0.0

    # APY (annualised, for reference)
    apy = (1.0 + earn_apr) ** 1 - 1.0   # continuous APR is stated as APY here

    # Value of already-accrued interest
    accrued_inr = cumulative_interest * current_price_inr

    return EarnResult(
        horizon_h=h,
        horizon_label=horizon_label,
        earn_tokens=earn_tokens,
        earn_value_inr=earn_inr,
        earn_value_usd=earn_usd,
        effective_rate_h=eff_rate,
        annualised_yield=apy,
        accrued_value_inr=accrued_inr,
    )


def compute_earn_all_horizons(
    quantity: float,
    current_price_inr: float,
    horizons_h: list,
    horizon_labels: list,
    earn_apr: float            = EARN_APR,
    periods_per_year: int      = EARN_PERIODS_PER_YEAR,
    cumulative_interest: float = CUMULATIVE_INTEREST_BERA,
    inr_usd_rate: float        = 84.0,
) -> list:
    """Compute EarnResult for every horizon in horizons_h."""
    return [
        compute_earn(
            quantity=quantity,
            current_price_inr=current_price_inr,
            earn_apr=earn_apr,
            h=h,
            horizon_label=label,
            periods_per_year=periods_per_year,
            cumulative_interest=cumulative_interest,
            inr_usd_rate=inr_usd_rate,
        )
        for h, label in zip(horizons_h, horizon_labels)
    ]


def breakeven_price_inr(
    entry_price_inr: float,
    earn_value_inr: float,
    quantity: float,
    tx_cost_pct: float,
    current_price_inr: float,
) -> float:
    """
    The minimum future price at which HOLD for h hours is at least as good
    as EXIT now, after accounting for earn yield and transaction costs.

    Derivation:
        HOLD is better than EXIT if:
            Q x (P_future - P_0) + Y_h - tx_cost_future >= Q x (P_t - P_0) - tx_cost_now

        Simplifying (P_0 cancels):
            Q x P_future + Y_h - tx_cost_future >= Q x P_t - tx_cost_now

        Approximate tx costs as proportional to current price:
            tx_cost_now    = Q x P_t x tx_cost_pct
            tx_cost_future = Q x P_future x tx_cost_pct

        So:
            Q x P_future x (1 - tx_cost_pct) >= Q x P_t x (1 - tx_cost_pct) - Y_h

            P_break = P_t - Y_h / (Q x (1 - tx_cost_pct))

        If P_break < P_0, then HOLD is already better than break-even from entry.

    Returns:
        P_break_inr — breakeven price in INR
    """
    effective_Q = quantity * (1.0 - tx_cost_pct)
    if effective_Q <= 0:
        return float("nan")
    p_break = current_price_inr - earn_value_inr / effective_Q
    return p_break
