# =============================================================================
# src/position/portfolio.py
#
# Mathematical responsibility: position accounting
# ─────────────────────────────────────────────────
#
# Raw inputs (user-provided — never derived from each other):
#   Q        : quantity of BERA tokens held (principal, excluding accrued)
#   C_inr    : total cost basis in INR  (amount paid)
#   V_inr    : current portfolio value in INR  (from platform/exchange)
#   earn_apr : annual earn rate (decimal, e.g., 0.1382 = 13.82%)
#   cum_int  : cumulative BERA interest already accrued (not yet claimed)
#
# ALL derived quantities are computed from the raw inputs:
#
#   P_0_inr   = C_inr / Q                           (entry price per token, INR)
#   P_t_inr   = V_inr / Q                           (current price per token, INR)
#   P_0_usd   = P_0_inr / INR_USD_RATE              (entry price, USD)
#   P_t_usd   = P_t_inr / INR_USD_RATE              (current price, USD)
#
#   unrealised_pnl_inr  = V_inr - C_inr             (total price gain/loss, INR)
#   return_pct          = unrealised_pnl_inr / C_inr (fraction)
#
#   tx_cost_inr  = V_inr x tx_cost_pct              (exit cost, INR)
#   exit_pnl_inr = unrealised_pnl_inr - tx_cost_inr (net after exit)
#
#   accrued_value_inr = cum_int x P_t_inr            (value of accrued BERA)
#   total_value_inr   = V_inr + accrued_value_inr    (position + accrued)
# =============================================================================

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config.config import (
    COST_BASIS,
    QUANTITY,
    CURRENT_VALUE_INR,
    INR_USD_RATE,
    TRANSACTION_COST_PCT,
    EARN_APR,
    CUMULATIVE_INTEREST_BERA,
)


# ---------------------------------------------------------------------------
# Position state dataclass — all fields are computed properties
# ---------------------------------------------------------------------------

@dataclass
class PositionState:
    """
    Complete accounting snapshot of the BERA position.

    Constructor takes only RAW inputs; all financial metrics are derived.

    Attributes (raw — set by caller):
        ticker                   : e.g. "BERA-USD"
        quantity                 : Q  — principal tokens held
        cost_basis_inr           : C_inr — total INR paid
        current_value_inr        : V_inr — current INR market value
        earn_apr                 : annual earn rate (decimal)
        cumulative_interest_bera : accrued BERA not yet claimed
        inr_usd_rate             : INR per USD
        tx_cost_pct              : one-way exit cost fraction

    Derived (computed properties — never stored directly):
        entry_price_inr / usd
        current_price_inr / usd
        unrealised_pnl_inr / usd
        return_pct
        tx_cost_inr / usd
        exit_pnl_inr / usd
        accrued_value_inr / usd
        total_value_inr
    """

    ticker                   : str
    quantity                 : float
    cost_basis_inr           : float
    current_value_inr        : float
    earn_apr                 : float   = EARN_APR
    cumulative_interest_bera : float   = CUMULATIVE_INTEREST_BERA
    inr_usd_rate             : float   = INR_USD_RATE
    tx_cost_pct              : float   = TRANSACTION_COST_PCT

    # ------------------------------------------------------------------
    # Price derivations
    # ------------------------------------------------------------------

    @property
    def entry_price_inr(self) -> float:
        """P_0_inr = C_inr / Q"""
        return self.cost_basis_inr / self.quantity

    @property
    def current_price_inr(self) -> float:
        """P_t_inr = V_inr / Q"""
        return self.current_value_inr / self.quantity

    @property
    def entry_price_usd(self) -> float:
        """P_0_usd = P_0_inr / INR_USD_RATE"""
        return self.entry_price_inr / self.inr_usd_rate

    @property
    def current_price_usd(self) -> float:
        """P_t_usd = P_t_inr / INR_USD_RATE"""
        return self.current_price_inr / self.inr_usd_rate

    # ------------------------------------------------------------------
    # P&L derivations
    # ------------------------------------------------------------------

    @property
    def unrealised_pnl_inr(self) -> float:
        """Pi_t = V_inr - C_inr"""
        return self.current_value_inr - self.cost_basis_inr

    @property
    def unrealised_pnl_usd(self) -> float:
        return self.unrealised_pnl_inr / self.inr_usd_rate

    @property
    def return_pct(self) -> float:
        """r_t = (V_inr - C_inr) / C_inr"""
        return self.unrealised_pnl_inr / self.cost_basis_inr

    # ------------------------------------------------------------------
    # Transaction cost derivations
    # ------------------------------------------------------------------

    @property
    def tx_cost_inr(self) -> float:
        """C_tx = V_inr x tx_cost_pct   (one-way exit cost)"""
        return self.current_value_inr * self.tx_cost_pct

    @property
    def tx_cost_usd(self) -> float:
        return self.tx_cost_inr / self.inr_usd_rate

    @property
    def exit_pnl_inr(self) -> float:
        """Pi_exit = Pi_t - C_tx   (net P&L after selling)"""
        return self.unrealised_pnl_inr - self.tx_cost_inr

    @property
    def exit_pnl_usd(self) -> float:
        return self.exit_pnl_inr / self.inr_usd_rate

    # ------------------------------------------------------------------
    # Earn derivations
    # ------------------------------------------------------------------

    @property
    def accrued_value_inr(self) -> float:
        """Value of already-accrued BERA interest = cum_int x P_t_inr"""
        return self.cumulative_interest_bera * self.current_price_inr

    @property
    def accrued_value_usd(self) -> float:
        return self.accrued_value_inr / self.inr_usd_rate

    @property
    def total_value_inr(self) -> float:
        """Total position value = V_inr + accrued_value_inr"""
        return self.current_value_inr + self.accrued_value_inr

    @property
    def total_quantity(self) -> float:
        """Q_total = Q + cumulative_interest_bera"""
        return self.quantity + self.cumulative_interest_bera


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_position(
    ticker: str                     = "BERA-USD",
    quantity: float                 = QUANTITY,
    cost_basis_inr: float           = COST_BASIS,
    current_value_inr: float        = CURRENT_VALUE_INR,
    earn_apr: float                 = EARN_APR,
    cumulative_interest_bera: float = CUMULATIVE_INTEREST_BERA,
    inr_usd_rate: float             = INR_USD_RATE,
    tx_cost_pct: float              = TRANSACTION_COST_PCT,
) -> PositionState:
    """
    Build a PositionState from the raw user inputs.

    All derived values (entry price, current price, P&L, return)
    are computed as properties — never stored or hard-coded.

    Example:
        pos = build_position(
            ticker="BERA-USD",
            quantity=19.36757634,
            cost_basis_inr=350.00,
            current_value_inr=358.42,
            earn_apr=0.1382,
            cumulative_interest_bera=0.00009685,
        )
        pos.entry_price_inr  # -> 18.073...  (computed from 350/19.367)
        pos.current_price_inr# -> 18.505...  (computed from 358.42/19.367)
        pos.return_pct       # -> 0.02406    (2.406%)
    """
    if quantity <= 0:
        raise ValueError(f"QUANTITY must be > 0.  Got {quantity}.")
    if cost_basis_inr <= 0:
        raise ValueError(f"COST_BASIS must be > 0.  Got {cost_basis_inr}.")
    if current_value_inr <= 0:
        raise ValueError(f"CURRENT_VALUE_INR must be > 0.  Got {current_value_inr}.")

    return PositionState(
        ticker=ticker,
        quantity=quantity,
        cost_basis_inr=cost_basis_inr,
        current_value_inr=current_value_inr,
        earn_apr=earn_apr,
        cumulative_interest_bera=cumulative_interest_bera,
        inr_usd_rate=inr_usd_rate,
        tx_cost_pct=tx_cost_pct,
    )
