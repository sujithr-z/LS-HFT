# =============================================================================
# src/features/microstructure.py
#
# Mathematical specification — V0 OHLCV proxies ONLY
# ─────────────────────────────────────────────────────
#
# ┌──────────────────────────────────────────────────────────────────────┐
# │  CRITICAL BOUNDARY                                                   │
# │                                                                      │
# │  V0 data = OHLCV  (daily candles from yfinance)                     │
# │  V0 data ≠ Level-2 order book                                       │
# │                                                                      │
# │  We DO NOT have Q_bid or Q_ask.                                     │
# │  Therefore the true imbalance formula:                               │
# │      I_t = (Q_bid - Q_ask) / (Q_bid + Q_ask)                       │
# │  MUST NOT be implemented here using fabricated quantities.           │
# │                                                                      │
# │  All formulas in this file are PROXIES derived from OHLCV.          │
# └──────────────────────────────────────────────────────────────────────┘
#
# 8.1  Close Location Value (CLV) — candle position proxy
#
#     CLV_t = (2·C_t - H_t - L_t) / (H_t - L_t)
#
#     Derivation:
#         Numerator   = (C_t - L_t) - (H_t - C_t)
#                     = distance from low  minus  distance to high
#         Denominator = H_t - L_t  (candle range)
#
#     Range: -1 ≤ CLV_t ≤ 1  (assuming L_t ≤ C_t ≤ H_t, which holds post-clean)
#     CLV_t ≈ +1  →  close near the high
#     CLV_t ≈ -1  →  close near the low
#     CLV_t = 0   →  close exactly at candle midpoint
#
#     Edge case: if H_t = L_t (doji / no range), CLV_t is undefined → NaN.
#
# 8.2  Relative Volume (RV)
#
#     RV_t = V_t / V̄_t^(n)
#
#     where V̄_t^(n) is the rolling mean volume from volume.py.
#     RV_t > 1  →  volume above recent average.
#     RV_t = 1  →  volume exactly at the rolling mean.
#
#     Edge case: if V̄_t = 0 → undefined → NaN.
#
# What is NOT implemented (reserved for V2 with Level-2 data):
#     I_t = (Q_bid - Q_ask) / (Q_bid + Q_ask)   — true order-book imbalance
#     Bid-ask spread
#     Order flow imbalance
# =============================================================================

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config.config import VOLUME_WINDOW

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def close_location_value(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """
    CLV_t = (2·C_t - H_t - L_t) / (H_t - L_t)

    Inputs:
        high  : pd.Series of H_t  (strictly positive, post-clean)
        low   : pd.Series of L_t  (strictly positive, post-clean)
        close : pd.Series of C_t  (strictly positive, post-clean)

    Output:
        pd.Series of CLV_t ∈ [-1, +1].
        NaN where H_t = L_t (zero-range candle / doji).

    Units: dimensionless.

    Mathematical invariants:
        CLV_t = +1   iff  C_t = H_t
        CLV_t = -1   iff  C_t = L_t
        CLV_t = 0    iff  C_t = (H_t + L_t) / 2

    Numerical test:
        H=110, L=90, C=100  →  CLV = (200-110-90)/(110-90) = 0/20 = 0.0
        H=110, L=90, C=105  →  CLV = (210-110-90)/(20) = 10/20 = 0.5
    """
    candle_range = high - low

    # Zero-range → undefined → NaN
    with np.errstate(invalid="ignore", divide="ignore"):
        clv = (2 * close - high - low) / candle_range.replace(0, float("nan"))

    clv.name = "clv"
    return clv


def relative_volume(
    volume: pd.Series,
    window: int = VOLUME_WINDOW,
) -> pd.Series:
    """
    RV_t = V_t / V̄_t^(n)

    Inputs:
        volume : pd.Series of V_t
        window : n (rolling window for the mean)

    Output:
        pd.Series of RV_t.  NaN for first (window-1) rows.
        NaN where V̄_t = 0.

    Mathematical invariants:
        RV_t = 1   iff  V_t = V̄_t
        RV_t > 1   iff  V_t > V̄_t  (above-average session)
        RV_t < 1   iff  V_t < V̄_t  (below-average session)
        RV_t ≥ 0   always (since V_t ≥ 0 and V̄_t > 0)
    """
    rolling_mean = volume.rolling(window=window, min_periods=window).mean()

    with np.errstate(invalid="ignore", divide="ignore"):
        rv = volume / rolling_mean.replace(0, float("nan"))

    rv.name = f"relative_volume_{window}"
    return rv


# ---------------------------------------------------------------------------
# Reserved for V2 — documented here for clarity
# ---------------------------------------------------------------------------

def _order_book_imbalance_NOT_IMPLEMENTED() -> None:
    """
    TRUE ORDER-BOOK IMBALANCE — reserved for V2.

    Formula (V2, with Level-2 data):
        I_t = (Q_bid - Q_ask) / (Q_bid + Q_ask)

    Requires:
        Q_bid : aggregate bid quantity at best bid
        Q_ask : aggregate ask quantity at best ask

    Why not in V0:
        OHLCV data does not contain Q_bid or Q_ask.
        Any attempt to fabricate these quantities from OHLCV
        would introduce false precision.
    """
    raise NotImplementedError(
        "Order-book imbalance requires Level-2 data (V2+).  "
        "This function must not be called in V0."
    )


# ---------------------------------------------------------------------------
# Convenience aggregator
# ---------------------------------------------------------------------------

def compute_all(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = VOLUME_WINDOW,
) -> pd.DataFrame:
    """
    Compute all V0 microstructure proxies.

    Columns produced:
        clv                        — Close Location Value ∈ [-1, +1]
        relative_volume_{window}   — V_t / V̄_t^(n)
    """
    clv = close_location_value(high, low, close)
    rv  = relative_volume(volume, window)

    df = pd.DataFrame({
        "clv":   clv,
        rv.name: rv,
    })

    logger.info(
        f"Microstructure proxies computed: {len(df)} rows  "
        f"(CLV proxy, RV proxy — true L2 features reserved for V2)"
    )
    return df
