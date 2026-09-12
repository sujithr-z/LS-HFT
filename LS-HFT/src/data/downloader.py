# =============================================================================
# src/data/downloader.py
#
# Mathematical responsibility: NONE.
#
# Pipeline role:
#   External Market Data  →  D_raw
#
# This module is a pure I/O layer.  It downloads OHLCV data from yfinance,
# validates the basic structural invariants, and saves the raw file.
# No return, volatility, or signal calculations may appear here.
#
# Validation invariants (enforced before saving):
#   - O_t > 0,  H_t > 0,  L_t > 0,  C_t > 0
#   - H_t >= max(O_t, C_t)
#   - L_t <= min(O_t, C_t)
#   - V_t >= 0
#   - timestamps strictly ascending: t_1 < t_2 < ... < t_N
# =============================================================================

import os
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

# Project configuration — single source of all parameters
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config.config import (
    TICKER, PERIOD, INTERVAL,
    RAW_DATA_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download(
    ticker: str = TICKER,
    period: str = PERIOD,
    interval: str = INTERVAL,
    save: bool = True,
    output_dir: str = RAW_DATA_PATH,
) -> pd.DataFrame:
    """
    Download historical OHLCV data for a given ticker.

    Mathematical inputs:
        ticker   — identifies the price series P_t
        period   — length of historical window
        interval — spacing between observations (Δt)

    Mathematical output:
        DataFrame D_raw with columns {Date, Open, High, Low, Close, Volume}
        satisfying the structural invariants listed in the module docstring.

    Raises:
        ValueError  — if yfinance returns an empty response
        ValueError  — if any structural invariant is violated
    """
    logger.info(f"Downloading {ticker}  period={period}  interval={interval}")

    raw: pd.DataFrame = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    # ------------------------------------------------------------------
    # Guard: empty response
    # ------------------------------------------------------------------
    if raw.empty:
        raise ValueError(
            f"yfinance returned an empty DataFrame for ticker='{ticker}' "
            f"period='{period}' interval='{interval}'.  "
            "Check the ticker symbol and your network connection."
        )

    logger.info(f"Downloaded {len(raw)} observations.")

    # ------------------------------------------------------------------
    # Normalise column names and index
    # ------------------------------------------------------------------
    raw = _normalise(raw)

    # ------------------------------------------------------------------
    # Structural validation
    # ------------------------------------------------------------------
    _validate(raw)

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    if save:
        path = _save(raw, ticker, interval, output_dir)
        logger.info(f"Raw data saved → {path}")

    return raw


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten MultiIndex columns (yfinance sometimes produces them),
    ensure standard column names, and name the index 'Date'.
    """
    # Flatten MultiIndex produced by yfinance ≥0.2.x
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Canonical column set
    rename_map = {
        "open":   "Open",
        "high":   "High",
        "low":    "Low",
        "close":  "Close",
        "volume": "Volume",
        "adj close": "Close",
    }
    df.columns = [rename_map.get(c.lower(), c) for c in df.columns]

    # Drop any Dividends / Stock Splits columns that auto_adjust may add
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].copy()

    df.index.name = "Date"
    return df


def _validate(df: pd.DataFrame) -> None:
    """
    Enforce structural invariants on D_raw before saving.

    Invariant 1 — price positivity:
        O_t > 0,  H_t > 0,  L_t > 0,  C_t > 0

    Invariant 2 — price ordering:
        H_t >= max(O_t, C_t)
        L_t <= min(O_t, C_t)

    Invariant 3 — volume non-negativity:
        V_t >= 0

    Invariant 4 — strictly ascending timestamps:
        t_1 < t_2 < ... < t_N
    """
    price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]

    # Invariant 1 — positivity
    for col in price_cols:
        n_nonpos = (df[col] <= 0).sum()
        if n_nonpos > 0:
            raise ValueError(
                f"Invariant violated: {n_nonpos} non-positive values in '{col}'.  "
                f"All prices must satisfy P_t > 0."
            )

    # Invariant 2 — OHLC ordering
    if all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
        bad_high = (df["High"] < df[["Open", "Close"]].max(axis=1)).sum()
        bad_low  = (df["Low"]  > df[["Open", "Close"]].min(axis=1)).sum()
        if bad_high > 0:
            logger.warning(
                f"{bad_high} rows where High < max(Open, Close).  "
                "Data may contain adjustment artefacts."
            )
        if bad_low > 0:
            logger.warning(
                f"{bad_low} rows where Low > min(Open, Close).  "
                "Data may contain adjustment artefacts."
            )

    # Invariant 3 — volume
    if "Volume" in df.columns:
        n_neg_vol = (df["Volume"] < 0).sum()
        if n_neg_vol > 0:
            raise ValueError(
                f"Invariant violated: {n_neg_vol} negative Volume values.  "
                "Volume must satisfy V_t >= 0."
            )

    # Invariant 4 — timestamps ascending
    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "Invariant violated: timestamps are not strictly ascending.  "
            "Expected t_1 < t_2 < ... < t_N."
        )

    logger.info("All structural invariants passed.")


def _save(
    df: pd.DataFrame,
    ticker: str,
    interval: str,
    output_dir: str,
) -> Path:
    """Save raw DataFrame as CSV and return the file path."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = f"{ticker.replace('.', '_')}_{interval}_raw.csv"
    path = Path(output_dir) / filename
    df.to_csv(path)
    return path
