# =============================================================================
# src/forecasting/return_distribution.py
#
# Mathematical specification
# ──────────────────────────
#
# Objective:
#   Estimate F(R_{t+h} | X_t) — the conditional distribution of future
#   h-period log returns given the current information set.
#
# V0.1 approach: parametric Gaussian scaling from empirical moments.
#
# Assumptions (i.i.d. log returns):
#   r_1, r_2, ..., r_N ~ iid  with empirical mean μ̂ and std σ̂
#
# Horizon scaling:
#   For h-period horizon (in same units as raw data, e.g., hours):
#       μ_h = h × μ̂          (mean scales linearly)
#       σ_h = σ̂ × √h         (std scales as square root of time)
#
#   Justification: if returns are i.i.d., the h-period return is the
#   sum of h single-period returns, and variance is additive:
#       Var(r_{1..h}) = h × Var(r)   →   σ_h = √h × σ
#
# Price model:
#   Assuming log-normal prices:
#       P_{t+h} = P_t × exp(r_h)   where  r_h ~ N(μ_h, σ_h²)
#
#   Expected future price (log-normal expectation formula):
#       E[P_{t+h}] = P_t × exp(μ_h + σ_h²/2)
#
#   Note: the log-normal mean includes the Jensen's inequality correction
#         (+ σ_h²/2) because E[exp(X)] = exp(μ + σ²/2) for X ~ N(μ, σ²).
#
# Probability of profit:
#   P(P_{t+h} > P_0) = P(r_h > ln(P_0 / P_t))
#                    = 1 - Φ( (ln(P_0/P_t) - μ_h) / σ_h )
#
# Monte Carlo:
#   Sample M paths: r_h^(i) ~ N(μ_h, σ_h²),  i = 1..M
#   P_{t+h}^(i) = P_t × exp(r_h^(i))
#   Use sample statistics for VaR, ES, and distribution percentiles.
# =============================================================================

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config.config import N_SIMULATIONS, RANDOM_SEED

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fitted distribution dataclass
# ---------------------------------------------------------------------------

@dataclass
class FittedDistribution:
    """
    Empirical moments fitted from historical log returns.

    Attributes:
        mu_period    : sample mean of single-period log returns (μ̂)
        sigma_period : sample std  of single-period log returns (σ̂)
        n_obs        : number of observations used in fitting
        period_label : description of the time unit (e.g., "1h")

    Methods:
        scale(h)     : returns (μ_h, σ_h) for horizon h periods
        expected_price(P_current, h): E[P_{t+h}] under log-normal
        prob_profit(P_current, P0, h): P(P_{t+h} > P_0)
        simulate(P_current, h, n_sim): Monte Carlo price paths
    """
    mu_period    : float
    sigma_period : float
    n_obs        : int
    period_label : str = "1-period"

    def scale(self, h: int) -> tuple[float, float]:
        """
        Scale moments to h-period horizon.

        Returns:
            (μ_h, σ_h) where:
                μ_h = h × μ̂
                σ_h = σ̂ × √h

        Mathematical invariant:
            scale(1) == (mu_period, sigma_period)
        """
        if h < 1:
            raise ValueError(f"Horizon h must be ≥ 1.  Got {h}.")
        mu_h    = self.mu_period * h
        sigma_h = self.sigma_period * np.sqrt(h)
        return mu_h, sigma_h

    def expected_price(self, p_current: float, h: int) -> float:
        """
        E[P_{t+h}] = P_t × exp(μ_h + σ_h²/2)

        The +σ²/2 correction arises from the log-normal expectation formula:
            E[exp(X)] = exp(μ + σ²/2)  for X ~ N(μ, σ²)

        Without this correction, exp(μ_h) would give the median, not the mean.
        """
        mu_h, sigma_h = self.scale(h)
        return p_current * np.exp(mu_h + 0.5 * sigma_h ** 2)

    def median_price(self, p_current: float, h: int) -> float:
        """
        Median[P_{t+h}] = P_t × exp(μ_h)

        The median (not the mean) is exp(μ_h) for a log-normal distribution.
        """
        mu_h, _ = self.scale(h)
        return p_current * np.exp(mu_h)

    def prob_profit(self, p_current: float, p0: float, h: int) -> float:
        """
        P(P_{t+h} > P_0) = 1 - Φ( (ln(P_0/P_t) - μ_h) / σ_h )

        Interpretation: probability that the future price exceeds our entry price.

        Edge cases:
            If p0 ≤ 0 or p_current ≤ 0  →  raise ValueError
            If σ_h = 0                   →  deterministic: return 1.0 if p_current > p0 else 0.0
        """
        if p0 <= 0 or p_current <= 0:
            raise ValueError("Prices must be strictly positive.")
        mu_h, sigma_h = self.scale(h)
        if sigma_h == 0:
            return 1.0 if p_current > p0 else 0.0
        z = (np.log(p0 / p_current) - mu_h) / sigma_h
        return float(1 - stats.norm.cdf(z))

    def simulate(
        self,
        p_current: float,
        h: int,
        n_sim: int  = N_SIMULATIONS,
        seed: int   = RANDOM_SEED,
    ) -> np.ndarray:
        """
        Monte Carlo: sample M future prices at horizon h.

        r_h^(i) ~ N(μ_h, σ_h²),   i = 1..n_sim
        P_{t+h}^(i) = P_t × exp(r_h^(i))

        Returns:
            np.ndarray of shape (n_sim,)  — simulated future prices

        Invariant:
            All simulated prices are strictly positive  (exp is always > 0).
        """
        rng = np.random.default_rng(seed)
        mu_h, sigma_h = self.scale(h)
        log_returns = rng.normal(loc=mu_h, scale=sigma_h, size=n_sim)
        return p_current * np.exp(log_returns)


# ---------------------------------------------------------------------------
# Fitting function
# ---------------------------------------------------------------------------

def fit(log_returns: pd.Series, period_label: str = "1-period") -> FittedDistribution:
    """
    Fit the parametric distribution from historical log returns.

    Input:
        log_returns  : pd.Series of r_t  (NaN dropped automatically)
        period_label : human-readable period description

    Output:
        FittedDistribution

    Mathematical procedure:
        μ̂ = (1/N) × Σ r_i
        σ̂ = sqrt( (1/(N-1)) × Σ (r_i - μ̂)² )   [ddof=1]

    Invariant:
        Requires at least 2 non-NaN observations.
    """
    valid = log_returns.dropna()
    n = len(valid)

    if n < 2:
        raise ValueError(
            f"Need ≥ 2 log-return observations to fit distribution.  Got {n}."
        )

    mu    = float(valid.mean())
    sigma = float(valid.std(ddof=1))

    if sigma <= 0:
        raise ValueError(
            "Fitted σ̂ ≤ 0 — all returns are identical.  Cannot build distribution."
        )

    logger.info(
        f"Distribution fitted: n={n}  μ={mu:.6f}  σ={sigma:.6f}  "
        f"[{period_label}]  "
        f"annualised σ ≈ {sigma * np.sqrt(8760):.1%}  (if hourly)"
    )

    return FittedDistribution(
        mu_period=mu,
        sigma_period=sigma,
        n_obs=n,
        period_label=period_label,
    )


# ---------------------------------------------------------------------------
# Bootstrap alternative (empirical, not parametric)
# ---------------------------------------------------------------------------

def bootstrap_simulate(
    log_returns: pd.Series,
    p_current: float,
    h: int,
    n_sim: int = N_SIMULATIONS,
    seed: int  = RANDOM_SEED,
) -> np.ndarray:
    """
    Bootstrap Monte Carlo: build h-period returns by resampling historical
    single-period returns WITHOUT assuming a parametric distribution.

    Method:
        For each simulation i:
            r_h^(i) = Σ_{j=1}^{h} r_{k_j}   where k_j are drawn uniformly
                      from {1..N} with replacement.
        P_{t+h}^(i) = P_t × exp(r_h^(i))

    This preserves empirical fat tails and avoids the Gaussian assumption.
    For V0.1 we use parametric as primary; bootstrap is the validation tool.

    Returns:
        np.ndarray of shape (n_sim,)
    """
    valid = log_returns.dropna().values
    n_hist = len(valid)

    if n_hist < h:
        logger.warning(
            f"Bootstrap: only {n_hist} historical returns but horizon h={h}.  "
            "Results may be unreliable."
        )

    rng = np.random.default_rng(seed + 1)
    # Draw (n_sim × h) indices
    indices = rng.integers(0, n_hist, size=(n_sim, h))
    r_h = valid[indices].sum(axis=1)     # sum h single-period returns
    return p_current * np.exp(r_h)
