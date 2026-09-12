# =============================================================================
# src/data/cleaner.py
#
# Pipeline role:
#   D_raw  →  D_clean
#
# Mathematical guarantees produced by this module:
#
#   4.1  Duplicate timestamps removed:
#            ∀ i ≠ j : t_i ≠ t_j
#
#   4.2  No NaN in price columns:
#            C_t ≠ NaN  (and likewise for O, H, L)
#
#   4.3  Chronological ordering:
#            t_1 < t_2 < ... < t_N
#
#   4.4  Price positivity:
#            P_t > 0  (required because we later compute ln(P_t / P_{t-1}))
#
# Cleaning policy:
#   - All removals are REPORTED before they happen.
#   - Data is never silently modified.
#   - A cleaning report dict is always returned alongside D_clean.
# =============================================================================

import logging
from pathlib import Path
from typing import NamedTuple

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PRICE_COLS = ["Open", "High", "Low", "Close"]


# ---------------------------------------------------------------------------
# Report object — every cleaning run returns one of these
# ---------------------------------------------------------------------------

class CleaningReport(NamedTuple):
    """
    Transparent accounting of every row removed or modified.

    Attributes:
        rows_input          : N before cleaning
        duplicates_removed  : rows dropped due to t_i = t_j
        missing_before      : NaN count across all price columns before drop
        missing_rows_removed: rows dropped because any price column is NaN
        nonpositive_removed : rows dropped because P_t <= 0
        rows_output         : N after cleaning  (=rows_input minus all drops)
        sort_applied        : True if sort was required to achieve t_1 < ... < t_N
    """
    rows_input: int
    duplicates_removed: int
    missing_before: int
    missing_rows_removed: int
    nonpositive_removed: int
    rows_output: int
    sort_applied: bool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Apply the full cleaning pipeline and return (D_clean, report).

    Step order:
        1. Sort chronologically                → ensures 4.3 before checks
        2. Remove duplicate timestamps         → ensures 4.1
        3. Remove rows with NaN price values   → ensures 4.2
        4. Remove rows with non-positive prices→ ensures 4.4

    Returns:
        (cleaned_df, CleaningReport)
    """
    df = df.copy()
    rows_input = len(df)

    # ------------------------------------------------------------------
    # Step 1 — Chronological ordering
    # ------------------------------------------------------------------
    sort_applied = False
    if not df.index.is_monotonic_increasing:
        logger.warning("Timestamps not ascending — sorting ascending.")
        df = df.sort_index(ascending=True)
        sort_applied = True

    # ------------------------------------------------------------------
    # Step 2 — Duplicate timestamps
    # ------------------------------------------------------------------
    n_before_dup = len(df)
    dup_mask = df.index.duplicated(keep="first")
    n_dups = int(dup_mask.sum())

    if n_dups > 0:
        dup_timestamps = df.index[dup_mask].tolist()
        logger.warning(
            f"Found {n_dups} duplicate timestamp(s) — keeping first occurrence.\n"
            f"  Removed at: {dup_timestamps[:10]}"
            f"{'...' if n_dups > 10 else ''}"
        )
        df = df[~dup_mask]
    else:
        logger.info("Duplicate timestamps: 0")

    # ------------------------------------------------------------------
    # Step 3 — Missing price values
    # ------------------------------------------------------------------
    present_price_cols = [c for c in PRICE_COLS if c in df.columns]
    missing_before = int(df[present_price_cols].isna().sum().sum())

    if missing_before > 0:
        nan_mask = df[present_price_cols].isna().any(axis=1)
        n_nan_rows = int(nan_mask.sum())
        logger.warning(
            f"Missing values: {missing_before} NaN cell(s) across price columns "
            f"→ removing {n_nan_rows} row(s)."
        )
        df = df[~nan_mask]
    else:
        n_nan_rows = 0
        logger.info("Missing price values: 0")

    # ------------------------------------------------------------------
    # Step 4 — Price positivity  (P_t > 0 required for ln(P_t / P_{t-1}))
    # ------------------------------------------------------------------
    nonpos_mask = (df[present_price_cols] <= 0).any(axis=1)
    n_nonpos = int(nonpos_mask.sum())

    if n_nonpos > 0:
        logger.warning(
            f"Non-positive prices: {n_nonpos} row(s) with P_t ≤ 0 — removing."
        )
        df = df[~nonpos_mask]
    else:
        logger.info("Non-positive prices: 0")

    # ------------------------------------------------------------------
    # Volume: coerce negative to NaN and warn (do not silently zero out)
    # ------------------------------------------------------------------
    if "Volume" in df.columns:
        neg_vol = (df["Volume"] < 0).sum()
        if neg_vol > 0:
            logger.warning(
                f"{neg_vol} negative Volume value(s) set to NaN — inspect manually."
            )
            df.loc[df["Volume"] < 0, "Volume"] = float("nan")

    # ------------------------------------------------------------------
    # Build report
    # ------------------------------------------------------------------
    report = CleaningReport(
        rows_input=rows_input,
        duplicates_removed=n_dups,
        missing_before=missing_before,
        missing_rows_removed=n_nan_rows,
        nonpositive_removed=n_nonpos,
        rows_output=len(df),
        sort_applied=sort_applied,
    )

    logger.info(
        f"Cleaning complete: {report.rows_input} → {report.rows_output} rows  "
        f"(removed {report.rows_input - report.rows_output})"
    )

    return df, report


def save_cleaned(
    df: pd.DataFrame,
    ticker: str,
    interval: str = "1d",
    output_dir: str = "data/processed",
) -> Path:
    """Persist D_clean as CSV. Returns the file path."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = f"{ticker.replace('.', '_')}_{interval}_clean.csv"
    path = Path(output_dir) / filename
    df.to_csv(path)
    logger.info(f"Cleaned data saved → {path}")
    return path


def print_report(report: CleaningReport) -> None:
    """Pretty-print the cleaning report to stdout."""
    print("\n" + "=" * 50)
    print("  CLEANING REPORT")
    print("=" * 50)
    print(f"  Rows input              : {report.rows_input:>8,}")
    print(f"  Duplicates removed      : {report.duplicates_removed:>8,}")
    print(f"  NaN cells (price)       : {report.missing_before:>8,}")
    print(f"  NaN rows removed        : {report.missing_rows_removed:>8,}")
    print(f"  Non-positive removed    : {report.nonpositive_removed:>8,}")
    print(f"  Sort applied            : {report.sort_applied!s:>8}")
    print(f"  Rows output             : {report.rows_output:>8,}")
    print("=" * 50 + "\n")
