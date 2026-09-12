# =============================================================================
# src/visualization/returns.py
#
# Maps:
#   (t, R_t)   →  time-series chart
#   f(R)       →  return histogram
#   (t, CR_t)  →  cumulative return chart
# =============================================================================

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config.config import FIGURES_PATH, FIGURE_DPI, FIGURE_SIZE, TICKER


def plot_returns(
    returns: pd.Series,
    ticker: str = TICKER,
    save: bool = True,
    output_dir: str = FIGURES_PATH,
    filename: str = "returns_timeseries.png",
) -> plt.Figure:
    """
    Plot (t, R_t) — return time series.

    Colours positive returns green, negative returns red.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    pos = returns.copy()
    neg = returns.copy()
    pos[pos < 0] = float("nan")
    neg[neg >= 0] = float("nan")

    ax.bar(returns.index, pos.values, color="#4CAF50", alpha=0.7, width=1, label="Positive")
    ax.bar(returns.index, neg.values, color="#F44336", alpha=0.7, width=1, label="Negative")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    ax.set_title(f"{ticker} — Daily Log Returns  r_t", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log Return")
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


def plot_return_histogram(
    returns: pd.Series,
    ticker: str = TICKER,
    bins: int = 60,
    save: bool = True,
    output_dir: str = FIGURES_PATH,
    filename: str = "returns_histogram.png",
) -> plt.Figure:
    """
    Plot the empirical return distribution  f(R).

    Overlays a fitted normal curve for comparison.
    Vertical lines at Q_{0.05} and Q_{0.95}.
    """
    valid = returns.dropna()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    ax.hist(valid, bins=bins, density=True, color="#2196F3", alpha=0.6,
            edgecolor="white", linewidth=0.3, label="Empirical")

    # Fitted normal overlay
    mu, sigma = valid.mean(), valid.std()
    x = np.linspace(valid.min(), valid.max(), 500)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), "r-", linewidth=2, label=f"N(μ={mu:.4f}, σ={sigma:.4f})")

    # Tail quantile markers
    q05 = float(valid.quantile(0.05))
    q95 = float(valid.quantile(0.95))
    ax.axvline(q05, color="#FF5722", linestyle="--", linewidth=1.5, label=f"Q₀.₀₅ = {q05:.4f}")
    ax.axvline(q95, color="#4CAF50", linestyle="--", linewidth=1.5, label=f"Q₀.₉₅ = {q95:.4f}")
    ax.axvline(0, color="black", linestyle="-", linewidth=0.8)

    ax.set_title(f"{ticker} — Return Distribution  f(R)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Log Return")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(output_dir) / filename, dpi=FIGURE_DPI)

    return fig


def plot_cumulative_return(
    cumulative_returns: pd.Series,
    ticker: str = TICKER,
    save: bool = True,
    output_dir: str = FIGURES_PATH,
    filename: str = "cumulative_return.png",
) -> plt.Figure:
    """
    Plot (t, CR_t)  where  CR_t = Π(1+R_i) - 1.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    ax.plot(cumulative_returns.index, cumulative_returns.values,
            linewidth=1.5, color="#9C27B0", label="Cumulative Return")
    ax.fill_between(cumulative_returns.index, cumulative_returns.values,
                    where=cumulative_returns.values >= 0,
                    alpha=0.15, color="#4CAF50", label="Positive")
    ax.fill_between(cumulative_returns.index, cumulative_returns.values,
                    where=cumulative_returns.values < 0,
                    alpha=0.15, color="#F44336", label="Negative")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    ax.set_title(f"{ticker} — Cumulative Return  CR_t", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
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
