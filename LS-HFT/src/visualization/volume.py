# =============================================================================
# src/visualization/volume.py
#
# Maps:
#   (t, V_t)      →  volume bar chart
#   (t, Z_t^V)    →  volume z-score chart with spike markers
# =============================================================================

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config.config import FIGURES_PATH, FIGURE_DPI, FIGURE_SIZE, TICKER, VOLUME_SPIKE_Z


def plot_volume(
    volume: pd.Series,
    ticker: str = TICKER,
    save: bool = True,
    output_dir: str = FIGURES_PATH,
    filename: str = "volume.png",
) -> plt.Figure:
    """
    Plot (t, V_t) — raw volume bar chart.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    ax.bar(volume.index, volume.values, color="#42A5F5", alpha=0.7, width=1, label="Volume V_t")

    ax.set_title(f"{ticker} — Trading Volume  V_t", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y/1e6:.0f}M" if y >= 1e6 else f"{y:.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate()
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    if save:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(output_dir) / filename, dpi=FIGURE_DPI)

    return fig


def plot_volume_zscore(
    zscore: pd.Series,
    spike_threshold: float = VOLUME_SPIKE_Z,
    ticker: str = TICKER,
    save: bool = True,
    output_dir: str = FIGURES_PATH,
    filename: str = "volume_zscore.png",
) -> plt.Figure:
    """
    Plot (t, Z_t^V)  with horizontal spike threshold and coloured bars.

    Bars above the threshold are highlighted in red.
    Z_t^V = 0 at normal volume.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    normal  = zscore.copy()
    spikes  = zscore.copy()
    normal[normal > spike_threshold]   = float("nan")
    spikes[spikes <= spike_threshold]  = float("nan")

    ax.bar(zscore.index, normal.values,  color="#42A5F5", alpha=0.7, width=1, label="Normal")
    ax.bar(zscore.index, spikes.values,  color="#F44336", alpha=0.85, width=1,
           label=f"Spike (Z > {spike_threshold})")

    ax.axhline(spike_threshold, color="#B71C1C", linestyle="--", linewidth=1.2,
               label=f"Threshold Z={spike_threshold}")
    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_title(f"{ticker} — Volume Z-score  Z_t^V = (V_t − V̄_t) / s_V,t", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Z-score")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate()
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    if save:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(output_dir) / filename, dpi=FIGURE_DPI)

    return fig
