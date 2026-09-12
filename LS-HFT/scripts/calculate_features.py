# =============================================================================
# scripts/calculate_features.py
#
# Orchestration only — contains NO mathematics.
#
# Pipeline:
#   load D_clean
#       → compute returns        (src/features/returns.py)
#       → compute volatility     (src/features/volatility.py)
#       → compute volume         (src/features/volume.py)
#       → compute microstructure (src/features/microstructure.py)
#       → save feature dataset
# =============================================================================

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    TICKER, INTERVAL,
    PROCESSED_DATA_PATH,
    RETURN_WINDOW, VOLATILITY_WINDOW, VOLUME_WINDOW,
    TRADING_PERIODS_PER_YEAR,
)
import src.features.returns        as ret_mod
import src.features.volatility     as vol_mod
import src.features.volume         as vvol_mod
import src.features.microstructure as micro_mod


def main() -> None:
    print(f"\n{'='*55}")
    print(f"  LS-HFT V0 — Calculate Features")
    print(f"{'='*55}\n")

    # ----------------------------------------------------------------
    # Load D_clean
    # ----------------------------------------------------------------
    clean_path = (
        PROJECT_ROOT / PROCESSED_DATA_PATH
        / f"{TICKER.replace('.', '_')}_{INTERVAL}_clean.csv"
    )
    df = pd.read_csv(clean_path, index_col=0, parse_dates=True)
    print(f"Loaded cleaned data: {len(df)} rows")

    prices = df["Close"]
    volume = df["Volume"]

    # ----------------------------------------------------------------
    # Returns  (src/features/returns.py)
    # ----------------------------------------------------------------
    returns_df = ret_mod.compute_all(prices, window=RETURN_WINDOW)

    # ----------------------------------------------------------------
    # Volatility  (src/features/volatility.py)
    # ----------------------------------------------------------------
    vol_df = vol_mod.compute_all(
        returns_df["log_return"],
        window=VOLATILITY_WINDOW,
        periods_per_year=TRADING_PERIODS_PER_YEAR,
    )

    # ----------------------------------------------------------------
    # Volume features  (src/features/volume.py)
    # ----------------------------------------------------------------
    volume_df = vvol_mod.compute_all(volume, window=VOLUME_WINDOW)

    # ----------------------------------------------------------------
    # Microstructure proxies  (src/features/microstructure.py)
    # ----------------------------------------------------------------
    micro_df = micro_mod.compute_all(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        volume=df["Volume"],
        window=VOLUME_WINDOW,
    )

    # ----------------------------------------------------------------
    # Assemble and save
    # ----------------------------------------------------------------
    feature_df = pd.concat([df, returns_df, vol_df, volume_df, micro_df], axis=1)

    out_path = (
        PROJECT_ROOT / PROCESSED_DATA_PATH
        / f"{TICKER.replace('.', '_')}_{INTERVAL}_features.csv"
    )
    feature_df.to_csv(out_path)

    print(f"\nFeature dataset saved: {len(feature_df)} rows × {len(feature_df.columns)} columns")
    print(f"  → {out_path}")
    print(f"\nColumns:\n  {list(feature_df.columns)}")


if __name__ == "__main__":
    main()
