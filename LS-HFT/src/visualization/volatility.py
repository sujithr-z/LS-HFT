# =============================================================================
# src/visualization/volatility.py
#
# Maps:
#   (t, σ_t)          →  rolling volatility chart
#   (t, σ_t_annual)   →  annualised volatility chart
#   vol_regime        →  regime overlay
#   (t, D_t)          →  drawdown chart
# =============================================================================

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config.config import FIGURES_PATH, FIGURE_DPI, FIGURE_SIZE, TICKER


def plot_rolling_volatility(
    rolling_vol: pd.Series,
    annualised_vol: pd.Series | None = None,
    ticker: str = TICKER,
    save: bool = True,
    output_dir: str = FIGURES_PATH,
    filename: str = "volatility.png",
) -> plt.Figure:
    """
    Plot (t, σ_t) and optionally (t, σ_t_annual).

    x-axis : t  (date)
    y-axis : volatility (dimensionless std of log returns)
    """
    n_panels = 2 if annualised_vol is not None else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] * n_panels),
                             sharex=True)
    if n_panels == 1:
        axes = [axes]

    # Panel 1 — rolling volatility
    ax = axes[0]
    ax.plot(rolling_vol.index, rolling_vol.values,
            linewidth=1.4, color="#FF9800", label=rolling_vol.name)
    ax.fill_between(rolling_vol.index, rolling_vol.values, alpha=0.2, color="#FF9800")
    ax.set_title(f"{ticker} — Rolling Volatility  σ_t", fontsize=13, fontweight="bold")
    ax.set_ylabel("σ_t  (daily std)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2 — annualised (optional)
    if annualised_vol is not None:
        ax2 = axes[1]
        ax2.plot(annualised_vol.index, annualised_vol.values,
                 linewidth=1.4, color="#E91E63", label=annualised_vol.name)
        ax2.fill_between(annualised_vol.index, annualised_vol.values, alpha=0.2, color="#E91E63")
        ax2.set_title(f"{ticker} — Annualised Volatility", fontsize=13, fontweight="bold")
        ax2.set_ylabel("σ_annual")
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate()
    fig.tight_layout()

    if save:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(output_dir) / filename, dpi=FIGURE_DPI)

    return fig


def plot_drawdown(
    drawdown: pd.Series,
    ticker: str = TICKER,
    save: bool = True,
    output_dir: str = FIGURES_PATH,
    filename: str = "drawdown.png",
) -> plt.Figure:
    """
    Plot (t, D_t)  where  D_t = 1 - W_t / M_t ∈ [0, 1].

    MDD is annotated.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    ax.fill_between(drawdown.index, -drawdown.values, 0,
                    color="#F44336", alpha=0.5, label="Drawdown")
    ax.plot(drawdown.index, -drawdown.values, color="#B71C1C", linewidth=0.8)

    mdd = float(drawdown.max())
    mdd_date = drawdown.idxmax()
    ax.annotate(
        f"MDD = {mdd:.2%}",
        xy=(mdd_date, -mdd),
        xytext=(mdd_date, -mdd * 0.7),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=10,
    )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{ticker} — Drawdown  D_t = 1 - W_t/M_t", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{-y:.0%}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate()
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(output_dir) / filename, dpi=FIGURE_DPI)

    return fig


def plot_volatility_regime(
    rolling_vol: pd.Series,
    regime: pd.Series,
    ticker: str = TICKER,
    save: bool = True,
    output_dir: str = FIGURES_PATH,
    filename: str = "vol_regime.png",
) -> plt.Figure:
    """
    Plot volatility with regime background shading.

    Shading:
        low    → green
        medium → yellow
        high   → red
    """
    colour_map = {"low": "#C8E6C9", "medium": "#FFF9C4", "high": "#FFCDD2"}

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Background regime shading
    prev_regime = None
    start_idx = None
    for i, (idx, reg) in enumerate(regime.items()):
        if pd.isna(reg):
            continue
        if reg != prev_regime:
            if prev_regime is not None and start_idx is not None:
                ax.axvspan(start_idx, idx, alpha=0.3,
                           color=colour_map.get(prev_regime, "white"), label=None)
            start_idx = idx
            prev_regime = reg
    if prev_regime and start_idx is not None:
        ax.axvspan(start_idx, rolling_vol.index[-1], alpha=0.3,
                   color=colour_map.get(prev_regime, "white"))

    ax.plot(rolling_vol.index, rolling_vol.values,
            linewidth=1.4, color="#FF9800", label="Rolling σ_t", zorder=5)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#C8E6C9", alpha=0.6, label="Low vol"),
        Patch(facecolor="#FFF9C4", alpha=0.6, label="Medium vol"),
        Patch(facecolor="#FFCDD2", alpha=0.6, label="High vol"),
    ]
    ax.legend(handles=legend_elements + ax.get_legend_handles_labels()[0][:1])

    ax.set_title(f"{ticker} — Volatility Regime (Q₀.₃₃ / Q₀.₆₇)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("σ_t")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(output_dir) / filename, dpi=FIGURE_DPI)

    return fig
