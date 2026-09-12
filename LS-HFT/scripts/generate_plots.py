# =============================================================================
# scripts/generate_plots.py
#
# Orchestration only — contains NO mathematics.
#
# Pipeline:
#   load feature dataset
#       → run statistical analysis   (src/analysis/statistics.py)
#       → run risk analysis          (src/analysis/risk.py)
#       → generate all plots         (src/visualization/)
#       → print summaries
# =============================================================================

import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for script execution

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    TICKER, INTERVAL,
    PROCESSED_DATA_PATH, FIGURES_PATH,
)
import src.analysis.statistics  as stat_mod
import src.analysis.risk        as risk_mod
import src.visualization.price      as viz_price
import src.visualization.returns    as viz_returns
import src.visualization.volatility as viz_vol
import src.visualization.volume     as viz_volume


def main() -> None:
    print(f"\n{'='*55}")
    print(f"  LS-HFT V0 — Generate Plots & Analysis")
    print(f"{'='*55}\n")

    # ----------------------------------------------------------------
    # Load feature dataset
    # ----------------------------------------------------------------
    feat_path = (
        PROJECT_ROOT / PROCESSED_DATA_PATH
        / f"{TICKER.replace('.', '_')}_{INTERVAL}_features.csv"
    )
    df = pd.read_csv(feat_path, index_col=0, parse_dates=True)
    print(f"Feature dataset loaded: {len(df)} rows")

    out_dir = str(PROJECT_ROOT / FIGURES_PATH)

    # ----------------------------------------------------------------
    # Statistical analysis
    # ----------------------------------------------------------------
    stats_dict = stat_mod.describe(df["log_return"])
    stat_mod.print_summary(stats_dict, label=f"{TICKER} Log Returns")

    # ----------------------------------------------------------------
    # Risk analysis
    # ----------------------------------------------------------------
    report = risk_mod.risk_report(
        simple_returns=df["simple_return"],
        log_returns=df["log_return"],
    )
    risk_mod.print_report(report)

    # ----------------------------------------------------------------
    # Visualisations
    # ----------------------------------------------------------------

    print("Generating plots...")

    # (t, P_t)
    viz_price.plot_price(df["Close"], ticker=TICKER, output_dir=out_dir)

    # (t, r_t)
    viz_returns.plot_returns(df["log_return"], ticker=TICKER, output_dir=out_dir)

    # f(R) histogram
    viz_returns.plot_return_histogram(df["log_return"], ticker=TICKER, output_dir=out_dir)

    # (t, CR_t)
    viz_returns.plot_cumulative_return(df["cumulative_return"], ticker=TICKER, output_dir=out_dir)

    # Identify rolling vol and annualised vol columns dynamically
    from config.config import VOLATILITY_WINDOW, TRADING_PERIODS_PER_YEAR
    rv_col  = f"rolling_vol_{VOLATILITY_WINDOW}"
    ann_col = f"{rv_col}_annualised"

    # (t, σ_t)
    viz_vol.plot_rolling_volatility(
        df[rv_col], df[ann_col] if ann_col in df.columns else None,
        ticker=TICKER, output_dir=out_dir,
    )

    # (t, D_t) — drawdown
    viz_vol.plot_drawdown(report["drawdown_series"], ticker=TICKER, output_dir=out_dir)

    # Regime chart
    if "vol_regime" in df.columns:
        viz_vol.plot_volatility_regime(df[rv_col], df["vol_regime"],
                                       ticker=TICKER, output_dir=out_dir)

    # (t, V_t)
    viz_volume.plot_volume(df["Volume"], ticker=TICKER, output_dir=out_dir)

    # (t, Z_t^V)
    zv_col = f"volume_zscore_{VOLATILITY_WINDOW}"
    if zv_col in df.columns:
        viz_volume.plot_volume_zscore(df[zv_col], ticker=TICKER, output_dir=out_dir)

    print(f"\nAll plots saved → {out_dir}")


if __name__ == "__main__":
    main()
