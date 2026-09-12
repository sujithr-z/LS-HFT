# =============================================================================
# src/data/loader.py
#
# Pipeline role:
#   raw file  →  validated DataFrame  →  D
#
# This module converts a persisted CSV into the canonical mathematical
# dataset D = {(t_i, O_i, H_i, L_i, C_i, V_i)}_{i=1}^{N}.
#
# The loader fails loudly if the required schema is absent.
# No computations are performed here.
# =============================================================================

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Required columns in the mathematical dataset D
REQUIRED_COLUMNS: list[str] = ["Open", "High", "Low", "Close", "Volume"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load(filepath: str | Path) -> pd.DataFrame:
    """
    Load a raw OHLCV CSV into the canonical DataFrame.

    Mathematical inputs:
        filepath — path to CSV produced by downloader.py

    Mathematical output:
        DataFrame with:
            index  → t_i  (DatetimeIndex, ascending)
            Open   → O_i
            High   → H_i
            Low    → L_i
            Close  → C_i
            Volume → V_i

    Raises:
        FileNotFoundError — if the file does not exist
        ValueError        — if required columns are missing
        ValueError        — if the index cannot be parsed as dates
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {filepath}\n"
            "Run scripts/download_data.py first."
        )

    logger.info(f"Loading  {filepath}")

    df = pd.read_csv(filepath, index_col=0, parse_dates=True)

    _validate_schema(df)
    _validate_index(df)

    logger.info(
        f"Loaded {len(df)} observations  "
        f"[{df.index[0].date()} → {df.index[-1].date()}]"
    )

    return df


def load_latest(
    ticker: str,
    interval: str = "1d",
    raw_dir: str = "data/raw",
) -> pd.DataFrame:
    """
    Convenience wrapper: locate the most recent raw file for a given ticker
    and interval, then delegate to load().
    """
    pattern = f"{ticker.replace('.', '_')}_{interval}_raw.csv"
    candidates = list(Path(raw_dir).glob(pattern))

    if not candidates:
        raise FileNotFoundError(
            f"No raw file matching '{pattern}' found in '{raw_dir}'."
        )

    # If somehow multiple files exist, take the newest by modification time
    path = max(candidates, key=lambda p: p.stat().st_mtime)
    return load(path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_schema(df: pd.DataFrame) -> None:
    """
    Verify that all required columns {O, H, L, C, V} are present.

    Mathematical requirement:
        D must contain at minimum {t, O, H, L, C, V}
        to support all downstream feature calculations.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Schema validation failed.  Missing columns: {missing}\n"
            f"Expected columns: {REQUIRED_COLUMNS}\n"
            f"Found columns:    {list(df.columns)}"
        )


def _validate_index(df: pd.DataFrame) -> None:
    """
    Verify the index represents parseable, ascending timestamps.

    Mathematical requirement:
        t_1 < t_2 < ... < t_N
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "Index could not be parsed as DatetimeIndex.  "
            "Ensure the CSV was created by downloader.py with a valid date index."
        )

    if not df.index.is_monotonic_increasing:
        logger.warning(
            "Timestamps are not strictly ascending.  "
            "cleaner.py will sort them."
        )
