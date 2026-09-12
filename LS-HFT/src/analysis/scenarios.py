"""Future price scenarios and expected P&L calculations."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .position import PositionInput, account_position


@dataclass(frozen=True)
class ScenarioSummary:
    horizon_periods: int
    expected_price_inr: float
    expected_pnl_inr: float
    probability_profit: float
    probability_loss: float
    price_quantiles_inr: dict[float, float]
    pnl_quantiles_inr: dict[float, float]


def historical_return_distribution(returns: pd.Series) -> pd.Series:
    """Return a validated simple-return sample with invalid observations removed."""
    clean = pd.Series(returns, dtype=float).dropna()
    if clean.empty or (clean <= -1).any():
        raise ValueError("returns must contain at least one value greater than -1")
    return clean


def simulate_future_prices(current_price_inr: float, returns: pd.Series,
                           horizon_periods: int, simulations: int = 50_000,
                           seed: int | None = 42) -> np.ndarray:
    """Bootstrap compounded future prices from the historical simple-return sample."""
    if current_price_inr <= 0 or horizon_periods < 1 or simulations < 1:
        raise ValueError("price, horizon, and simulations must be positive")
    sample = historical_return_distribution(returns).to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.choice(sample, size=(simulations, horizon_periods), replace=True)
    return current_price_inr * np.prod(1.0 + draws, axis=1)


def expected_pnl(position: PositionInput, future_prices_inr: np.ndarray,
                 transaction_cost_pct: float = 0.0) -> np.ndarray:
    """Net P&L versus exit now, including a one-way exit cost."""
    if transaction_cost_pct < 0 or transaction_cost_pct >= 1:
        raise ValueError("transaction_cost_pct must be in [0, 1)")
    proceeds = position.quantity * future_prices_inr * (1.0 - transaction_cost_pct)
    return proceeds - position.cost_basis_inr


def summarize_scenarios(position: PositionInput, returns: pd.Series,
                        horizon_periods: int, transaction_cost_pct: float = 0.0,
                        simulations: int = 50_000, seed: int | None = 42) -> ScenarioSummary:
    current_price = account_position(position).current_implied_price_inr
    prices = simulate_future_prices(current_price, returns, horizon_periods, simulations, seed)
    pnl = expected_pnl(position, prices, transaction_cost_pct)
    q = {level: float(np.quantile(prices, level)) for level in (0.05, 0.50, 0.95)}
    pq = {level: float(np.quantile(pnl, level)) for level in (0.05, 0.50, 0.95)}
    return ScenarioSummary(horizon_periods, float(prices.mean()), float(pnl.mean()),
                           float((pnl > 0).mean()), float((pnl < 0).mean()), q, pq)
