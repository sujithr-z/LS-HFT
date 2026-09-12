"""Pure position accounting for the configured live research position."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PositionInput:
    symbol: str
    quantity: float
    cost_basis_inr: float
    current_value_inr: float
    earn_apr: float = 0.0
    cumulative_interest_bera: float = 0.0

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.cost_basis_inr < 0 or self.current_value_inr < 0:
            raise ValueError("values must be non-negative")
        if self.earn_apr < -1:
            raise ValueError("earn_apr must be greater than -100%")
        if self.cumulative_interest_bera < 0:
            raise ValueError("interest must be non-negative")


@dataclass(frozen=True)
class PositionAccounting:
    symbol: str
    quantity: float
    cost_basis_inr: float
    current_value_inr: float
    average_entry_price_inr: float
    current_implied_price_inr: float
    unrealized_pnl_inr: float
    unrealized_return: float
    interest_value_inr: float
    current_value_including_interest_inr: float
    return_including_interest: float


def account_position(position: PositionInput) -> PositionAccounting:
    """Derive entry price, mark price, P&L, and return from primitive inputs."""
    entry_price = position.cost_basis_inr / position.quantity
    current_price = position.current_value_inr / position.quantity
    pnl = position.current_value_inr - position.cost_basis_inr
    base_return = pnl / position.cost_basis_inr if position.cost_basis_inr else float("nan")
    interest_value = position.cumulative_interest_bera * current_price
    value_with_interest = position.current_value_inr + interest_value
    total_return = ((value_with_interest - position.cost_basis_inr) / position.cost_basis_inr
                    if position.cost_basis_inr else float("nan"))
    return PositionAccounting(
        symbol=position.symbol,
        quantity=position.quantity,
        cost_basis_inr=position.cost_basis_inr,
        current_value_inr=position.current_value_inr,
        average_entry_price_inr=entry_price,
        current_implied_price_inr=current_price,
        unrealized_pnl_inr=pnl,
        unrealized_return=base_return,
        interest_value_inr=interest_value,
        current_value_including_interest_inr=value_with_interest,
        return_including_interest=total_return,
    )


def bera_position() -> PositionInput:
    """Build the BERA position from config, without duplicating derived values."""
    from config.config import (CUMULATIVE_INTEREST_BERA, CURRENT_VALUE_INR,
                               COST_BASIS, EARN_APR, QUANTITY)
    return PositionInput("BERA", QUANTITY, COST_BASIS, CURRENT_VALUE_INR,
                         EARN_APR, CUMULATIVE_INTEREST_BERA)


__all__ = ["PositionInput", "PositionAccounting", "account_position", "bera_position"]
