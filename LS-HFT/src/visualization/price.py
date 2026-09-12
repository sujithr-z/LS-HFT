# =============================================================================
# src/visualization/price.py
#
# Maps:  (t, P_t)  →  chart
#
# This module does not compute any mathematics.
# It consumes the output of the data pipeline and renders it.
# =============================================================================

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config.config import FIGURES_PATH, FIGURE_DPI, FIGURE_SIZE, TICKER


def plot_price(
    prices: pd.Series,
    ticker: str = TICKER,
    save: bool = True,
    output_dir: str = FIGURES_PATH,
    filename: str = "price.png",
) -> plt.Figure:
    """
    Plot the price series  P_t = C_t  over time.

    x-axis : t  (date)
    y-axis : P_t  (closing price)

    No mathematical transformation is applied — raw closing prices only.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    ax.plot(prices.index, prices.values, linewidth=1.2, color="#2196F3", label=ticker)
    ax.fill_between(prices.index, prices.values, alpha=0.08, color="#2196F3")

    ax.set_title(f"{ticker} — Closing Price  (P_t = C_t)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
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
