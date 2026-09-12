# =============================================================================
# src/decision/expected_value.py
#
# Mathematical specification
# ──────────────────────────
#
# For a long BERA position (Q tokens bought at P_0):
#
#   Future price P&L at horizon h (Monte Carlo, INR):
#       Pi_price_h^(i) = Q x (P_h^(i) - P_0_inr) - C_tx^(i)
#   where:
#       P_h^(i)   = simulated future price in INR  (= P_usd^(i) x INR_USD_RATE)
#       C_tx^(i)  = Q x P_h^(i) x tx_cost_pct     (exit cost at future price)
#
#   Earn yield over h hours (deterministic, from earn.py):
#       Y_h = Q x P_t_inr x ((1 + APR)^(h/N) - 1)
#
#   Earn-adjusted P&L for each path:
#       Pi_earn_h^(i) = Pi_price_h^(i) + Y_h
#
#   Expected values:
#       E[Pi_price] = mean(Pi_price_h)           (price-only EV)
#       E[Pi_earn]  = E[Pi_price] + Y_h          (earn-adjusted EV)
#
#   Probability of profit (earn-adjusted):
#       P(Pi_earn > 0) = #{Pi_earn_h^(i) > 0} / M
#
#   VaR (on earn-adjusted P&L):
#       VaR_a = -Q_{1-a}(Pi_earn_h)   [positive = loss]
#
#   Expected Shortfall (on earn-adjusted P&L):
#       ES_a = -E[Pi_earn | Pi_earn <= -VaR_a]
#
#   Hold vs Exit comparison (incremental EV of holding over exiting):
#       DeltaEV(h) = E[Pi_earn(h)] - Pi_exit_now
#   where:
#       Pi_exit_now = Q x (P_t - P_0) - Q x P_t x tx_cost_pct  (certain)
#
#   Interpretation:
#       DeltaEV > 0  -> HOLD has higher expected value than EXIT now
#       DeltaEV < 0  -> EXIT now has higher expected value than HOLD for h hours
# =============================================================================

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config.config import N_SIMULATIONS, RANDOM_SEED, VAR_CONFIDENCE, INR_USD_RATE
from src.forecasting.return_distribution import FittedDistribution

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass — one horizon
# ---------------------------------------------------------------------------

@dataclass
class EVResult:
    """
    All expected-value metrics for one (position, horizon) pair.

    Monetary values in INR unless suffixed _usd.
    Price expected values in USD (distribution is fitted in USD).

    Attributes:
        horizon_h               : h (hours)
        horizon_label           : e.g., "7 days"
        n_sim                   : Monte Carlo paths

        --- Price-only EV ---
        expected_return_pct     : E[R_h] = E[Pi_price] / cost_inr  (fraction)
        expected_pnl_inr        : E[Pi_price_h]  (INR)
        prob_profit_price       : P(Pi_price > 0)
        prob_loss_price         : P(Pi_price < 0)

        --- Earn yield ---
        earn_value_inr          : Y_h  (deterministic earn yield, INR)
        earn_pct                : Y_h / cost_inr

        --- Earn-adjusted EV ---
        expected_earn_pnl_inr   : E[Pi_earn] = E[Pi_price] + Y_h
        expected_earn_return_pct: E[Pi_earn] / cost_inr
        prob_profit_earn        : P(Pi_earn > 0)
        prob_loss_earn          : P(Pi_earn < 0)
        cond_gain_inr           : E[Pi_earn | Pi_earn > 0]
        cond_loss_inr           : -E[Pi_earn | Pi_earn < 0]

        --- Risk measures (on earn-adjusted P&L) ---
        var_inr                 : VaR_a  (positive = loss)
        es_inr                  : ES_a   (positive = loss)
        es_fraction             : ES / current_position_value

        --- Hold vs Exit ---
        exit_pnl_now_inr        : certain P&L from exiting immediately
        delta_ev_inr            : E[Pi_earn(h)] - exit_pnl_now  (> 0 = HOLD better)

        alpha                   : confidence level
    """
    horizon_h               : int
    horizon_label           : str
    n_sim                   : int

    expected_return_pct     : float
    expected_pnl_inr        : float

    prob_profit_price       : float
    prob_loss_price         : float

    earn_value_inr          : float
    earn_pct                : float

    expected_earn_pnl_inr   : float
    expected_earn_return_pct: float
    prob_profit_earn        : float
    prob_loss_earn          : float
    cond_gain_inr           : float
    cond_loss_inr           : float

    var_inr                 : float
    es_inr                  : float
    es_fraction             : float

    exit_pnl_now_inr        : float
    delta_ev_inr            : float

    alpha                   : float


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

def compute_ev(
    dist              : FittedDistribution,
    h                 : int,
    horizon_label     : str,
    p_current_usd     : float,
    p_entry_usd       : float,
    quantity          : float,
    inr_usd_rate      : float,
    tx_cost_pct       : float,
    earn_value_inr    : float,     # Y_h from earn.py — deterministic
    alpha             : float      = VAR_CONFIDENCE,
    n_sim             : int        = N_SIMULATIONS,
    seed              : int        = RANDOM_SEED,
) -> EVResult:
    """
    Compute all EV metrics for the position at horizon h.

    Steps:
        1. Monte Carlo: simulate M future prices in USD
        2. Convert to INR using inr_usd_rate
        3. Compute price P&L for each path (with exit tx cost)
        4. Add earn yield Y_h to get earn-adjusted P&L per path
        5. Derive distribution statistics
        6. Compute DeltaEV vs exiting now
    """
    # ------------------------------------------------------------------
    # Step 1 — simulate future prices (USD)
    # ------------------------------------------------------------------
    future_prices_usd = dist.simulate(p_current_usd, h, n_sim, seed)
    future_prices_inr = future_prices_usd * inr_usd_rate

    # ------------------------------------------------------------------
    # Step 2 — price P&L (INR) for each path
    #   Pi_price^(i) = Q x (P_h^(i) - P_0_inr) - Q x P_h^(i) x tx_cost_pct
    # ------------------------------------------------------------------
    p_entry_inr       = p_entry_usd * inr_usd_rate
    exit_costs_inr    = quantity * future_prices_inr * tx_cost_pct
    pnl_price         = quantity * (future_prices_inr - p_entry_inr) - exit_costs_inr

    # ------------------------------------------------------------------
    # Step 3 — earn-adjusted P&L (Y_h is deterministic, same for all paths)
    #   Pi_earn^(i) = Pi_price^(i) + Y_h
    # ------------------------------------------------------------------
    pnl_earn = pnl_price + earn_value_inr

    # ------------------------------------------------------------------
    # Step 4 — position value for normalisation
    # ------------------------------------------------------------------
    cost_inr           = quantity * p_entry_inr
    position_value_inr = quantity * p_current_usd * inr_usd_rate

    # ------------------------------------------------------------------
    # Step 5 — price-only statistics
    # ------------------------------------------------------------------
    expected_pnl_price = float(pnl_price.mean())
    exp_ret_price      = expected_pnl_price / cost_inr if cost_inr > 0 else float("nan")
    prob_profit_price  = float((pnl_price > 0).sum()) / n_sim
    prob_loss_price    = float((pnl_price < 0).sum()) / n_sim

    # ------------------------------------------------------------------
    # Step 6 — earn-adjusted statistics
    # ------------------------------------------------------------------
    expected_earn_pnl = float(pnl_earn.mean())
    exp_ret_earn      = expected_earn_pnl / cost_inr if cost_inr > 0 else float("nan")

    n_profit = int((pnl_earn > 0).sum())
    n_loss   = int((pnl_earn < 0).sum())
    prob_profit_earn = n_profit / n_sim
    prob_loss_earn   = n_loss   / n_sim

    gain_paths = pnl_earn[pnl_earn > 0]
    loss_paths = pnl_earn[pnl_earn < 0]
    cond_gain  = float(gain_paths.mean()) if len(gain_paths) > 0 else 0.0
    cond_loss  = float(-loss_paths.mean()) if len(loss_paths) > 0 else 0.0

    # ------------------------------------------------------------------
    # Step 7 — VaR and ES on earn-adjusted P&L
    # ------------------------------------------------------------------
    tail_q   = 1.0 - alpha
    var_inr  = float(max(-np.quantile(pnl_earn, tail_q), 0.0))

    tail_mask = pnl_earn <= -var_inr
    es_inr    = float(-pnl_earn[tail_mask].mean()) if tail_mask.sum() > 0 else 0.0
    es_inr    = max(es_inr, 0.0)
    es_frac   = es_inr / position_value_inr if position_value_inr > 0 else float("nan")

    # ------------------------------------------------------------------
    # Step 8 — Hold vs Exit (DeltaEV)
    #   exit_pnl_now = Q x (P_t - P_0) - Q x P_t x tx_cost_pct   (certain)
    # ------------------------------------------------------------------
    exit_pnl_now = (quantity * (p_current_usd - p_entry_usd) * inr_usd_rate
                    - quantity * p_current_usd * inr_usd_rate * tx_cost_pct)
    delta_ev = expected_earn_pnl - exit_pnl_now

    earn_pct = earn_value_inr / cost_inr if cost_inr > 0 else 0.0

    return EVResult(
        horizon_h               = h,
        horizon_label           = horizon_label,
        n_sim                   = n_sim,
        expected_return_pct     = exp_ret_price,
        expected_pnl_inr        = expected_pnl_price,
        prob_profit_price       = prob_profit_price,
        prob_loss_price         = prob_loss_price,
        earn_value_inr          = earn_value_inr,
        earn_pct                = earn_pct,
        expected_earn_pnl_inr   = expected_earn_pnl,
        expected_earn_return_pct= exp_ret_earn,
        prob_profit_earn        = prob_profit_earn,
        prob_loss_earn          = prob_loss_earn,
        cond_gain_inr           = cond_gain,
        cond_loss_inr           = cond_loss,
        var_inr                 = var_inr,
        es_inr                  = es_inr,
        es_fraction             = es_frac,
        exit_pnl_now_inr        = exit_pnl_now,
        delta_ev_inr            = delta_ev,
        alpha                   = alpha,
    )


def compute_all_horizons(
    dist          : FittedDistribution,
    horizons_h    : list,
    horizon_labels: list,
    p_current_usd : float,
    p_entry_usd   : float,
    quantity      : float,
    inr_usd_rate  : float,
    tx_cost_pct   : float,
    earn_results  : list,          # [EarnResult] from earn.py, one per horizon
    alpha         : float = VAR_CONFIDENCE,
    n_sim         : int   = N_SIMULATIONS,
    seed          : int   = RANDOM_SEED,
) -> list:
    """
    Run compute_ev() across all configured horizons.

    earn_results must be pre-computed from earn.py so that the
    earn mathematics stays in one module (earn.py) and EV stays
    in this module (expected_value.py).
    """
    results = []
    for ev_seed, (h, label, er) in enumerate(
        zip(horizons_h, horizon_labels, earn_results)
    ):
        ev = compute_ev(
            dist=dist,
            h=h,
            horizon_label=label,
            p_current_usd=p_current_usd,
            p_entry_usd=p_entry_usd,
            quantity=quantity,
            inr_usd_rate=inr_usd_rate,
            tx_cost_pct=tx_cost_pct,
            earn_value_inr=er.earn_value_inr,
            alpha=alpha,
            n_sim=n_sim,
            seed=seed + ev_seed * 17,   # distinct seed per horizon
        )
        results.append(ev)
    return results
