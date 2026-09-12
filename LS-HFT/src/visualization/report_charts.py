# =============================================================================
# src/visualization/report_charts.py
#
# Pipeline role:
#   (PositionState, FittedDistribution, [EVResult], [EarnResult],
#    [ScenarioSet], [DecisionResult])  ->  matplotlib figure dashboard
#
# Called automatically after print_full_report() completes.
# Uses the interactive backend so the window pops up immediately.
# This module contains NO mathematics.
# =============================================================================

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.position.portfolio              import PositionState
from src.position.earn                   import EarnResult
from src.forecasting.return_distribution import FittedDistribution
from src.forecasting.price_scenarios     import ScenarioSet
from src.decision.expected_value         import EVResult
from src.decision.hold_decision          import DecisionResult, Decision

# ---------------------------------------------------------------------------
# Colour palette (consistent with ANSI colours in position_report.py)
# ---------------------------------------------------------------------------
_CLR_HOLD     = "#4CAF50"   # green
_CLR_REDUCE   = "#FFC107"   # amber
_CLR_EXIT     = "#F44336"   # red
_CLR_INSUF    = "#FFC107"   # amber
_CLR_PRICE    = "#2196F3"   # blue
_CLR_BG       = "#0f1117"   # dark background
_CLR_PANEL    = "#1a1d27"   # panel background
_CLR_TEXT     = "#e0e6f0"   # primary text
_CLR_MUTED    = "#6b7280"   # muted text
_CLR_ACCENT   = "#00e5ff"   # cyan accent (matching _CYAN in position_report)
_CLR_POSITIVE = "#4CAF50"
_CLR_NEGATIVE = "#F44336"

_DECISION_CLR = {
    Decision.HOLD:              _CLR_HOLD,
    Decision.REDUCE:            _CLR_REDUCE,
    Decision.EXIT:              _CLR_EXIT,
    Decision.INSUFFICIENT_DATA: _CLR_INSUF,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pnl_colour(v: float) -> str:
    return _CLR_POSITIVE if v >= 0 else _CLR_NEGATIVE


def _apply_dark_style(fig: plt.Figure) -> None:
    """Apply dark background to a figure and all its axes."""
    fig.patch.set_facecolor(_CLR_BG)
    for ax in fig.get_axes():
        ax.set_facecolor(_CLR_PANEL)
        ax.tick_params(colors=_CLR_TEXT, labelsize=8)
        ax.xaxis.label.set_color(_CLR_TEXT)
        ax.yaxis.label.set_color(_CLR_TEXT)
        ax.title.set_color(_CLR_ACCENT)
        for spine in ax.spines.values():
            spine.set_edgecolor(_CLR_MUTED)


# ---------------------------------------------------------------------------
# Panel 1 — Position P&L summary
# ---------------------------------------------------------------------------

def _plot_pnl_summary(ax: plt.Axes, pos: PositionState) -> None:
    labels  = ["Cost Basis", "Current Value"]
    values  = [pos.cost_basis_inr, pos.current_value_inr]
    colours = [_CLR_MUTED, _pnl_colour(pos.unrealised_pnl_inr)]

    bars = ax.bar(labels, values, color=colours, width=0.45,
                  edgecolor=_CLR_BG, linewidth=1.5)

    top = max(values)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + top * 0.015,
            f"Rs{val:,.2f}",
            ha="center", va="bottom",
            color=_CLR_TEXT, fontsize=9, fontweight="bold",
        )

    pnl_col = _pnl_colour(pos.unrealised_pnl_inr)
    ax.annotate(
        f"Unrealised P&L\nRs{pos.unrealised_pnl_inr:+,.2f}  ({pos.return_pct:+.2%})",
        xy=(0.5, 0.97), xycoords="axes fraction",
        ha="center", va="top", color=pnl_col, fontsize=9, fontweight="bold",
    )
    ax.set_title("Position P&L Summary", fontsize=10, fontweight="bold")
    ax.set_ylabel("INR (Rs)", color=_CLR_TEXT)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"Rs{x:,.0f}")
    )
    ax.grid(axis="y", alpha=0.15, color=_CLR_MUTED)


# ---------------------------------------------------------------------------
# Panel 2 — Horizon EV analysis
# ---------------------------------------------------------------------------

def _plot_horizon_ev(
    ax: plt.Axes,
    ev_results: list,
    decisions: list,
) -> None:
    labels     = [ev.horizon_label for ev in ev_results]
    x          = np.arange(len(labels))
    width      = 0.35
    exp_ret    = [ev.expected_earn_return_pct * 100 for ev in ev_results]
    delta_ev   = [ev.delta_ev_inr for ev in ev_results]
    dec_clrs   = [_DECISION_CLR.get(dr.decision, _CLR_MUTED) for dr in decisions]

    ax.bar(x - width / 2, exp_ret, width,
           label="E[Earn+Return] %",
           color=[_pnl_colour(v) for v in exp_ret],
           alpha=0.75, edgecolor=_CLR_BG)

    ax2 = ax.twinx()
    ax2.bar(x + width / 2, delta_ev, width,
            label="DeltaEV (Rs)",
            color=dec_clrs, alpha=0.85, edgecolor=_CLR_BG)
    ax2.set_ylabel("DeltaEV (Rs)", color=_CLR_MUTED, fontsize=8)
    ax2.tick_params(colors=_CLR_MUTED, labelsize=7)
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"Rs{x:,.0f}")
    )
    ax2.spines["right"].set_edgecolor(_CLR_MUTED)
    ax2.set_facecolor(_CLR_PANEL)

    ax.set_title("Horizon EV Analysis", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Expected Return (%)", fontsize=8)
    ax.axhline(0, color=_CLR_MUTED, linewidth=0.8, linestyle="--")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.1f}%")
    )
    legend_patches = [
        mpatches.Patch(color=_CLR_HOLD,   label="HOLD"),
        mpatches.Patch(color=_CLR_REDUCE, label="REDUCE"),
        mpatches.Patch(color=_CLR_EXIT,   label="EXIT"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=7,
              facecolor=_CLR_PANEL, edgecolor=_CLR_MUTED, labelcolor=_CLR_TEXT)


# ---------------------------------------------------------------------------
# Panel 3 — VaR & ES
# ---------------------------------------------------------------------------

def _plot_risk_metrics(ax: plt.Axes, ev_results: list) -> None:
    labels   = [ev.horizon_label for ev in ev_results]
    var_vals = [ev.var_inr for ev in ev_results]
    es_vals  = [ev.es_inr  for ev in ev_results]
    x        = np.arange(len(labels))

    ax.plot(x, var_vals, marker="o", color=_CLR_REDUCE, linewidth=2,
            markersize=5, label="VaR (Rs)")
    ax.plot(x, es_vals,  marker="s", color=_CLR_EXIT,   linewidth=2,
            markersize=5, label="ES (Rs)", linestyle="--")
    ax.fill_between(x, var_vals, es_vals, alpha=0.12, color=_CLR_EXIT)

    ax.set_title("Downside Risk: VaR & ES", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("INR (Rs)", fontsize=8)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"Rs{x:,.0f}")
    )
    ax.legend(fontsize=7, facecolor=_CLR_PANEL,
              edgecolor=_CLR_MUTED, labelcolor=_CLR_TEXT)
    ax.grid(alpha=0.12, color=_CLR_MUTED)


# ---------------------------------------------------------------------------
# Panel 4 — Earn yield
# ---------------------------------------------------------------------------

def _plot_earn_yield(ax: plt.Axes, earn_results: list) -> None:
    labels = [er.horizon_label for er in earn_results]
    values = [er.earn_value_inr for er in earn_results]
    x      = np.arange(len(labels))

    bars = ax.bar(x, values, color=_CLR_ACCENT, alpha=0.8,
                  edgecolor=_CLR_BG, linewidth=1)

    top = max(values) if values else 1.0
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + top * 0.02,
            f"Rs{val:.4f}",
            ha="center", va="bottom", color=_CLR_ACCENT, fontsize=7,
        )

    ax.set_title("Earn Yield per Horizon", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Earn (Rs)", fontsize=8)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"Rs{x:.4f}")
    )
    ax.grid(axis="y", alpha=0.12, color=_CLR_MUTED)


# ---------------------------------------------------------------------------
# Panel 5 — Price scenarios
# ---------------------------------------------------------------------------

def _plot_price_scenarios(
    ax: plt.Axes,
    scenario_sets: list,
    pos: PositionState,
) -> None:
    labels = [ss.horizon_label for ss in scenario_sets]
    x      = np.arange(len(labels))
    width  = 0.22

    bear_p   = [ss.bear.price_usd for ss in scenario_sets]
    base_p   = [ss.base.price_usd for ss in scenario_sets]
    bull_p   = [ss.bull.price_usd for ss in scenario_sets]

    ax.bar(x - width, bear_p, width, label="Bear (Q10)",
           color=_CLR_EXIT, alpha=0.8, edgecolor=_CLR_BG)
    ax.bar(x,          base_p, width, label="Base (Q50)",
           color=_CLR_MUTED, alpha=0.8, edgecolor=_CLR_BG)
    ax.bar(x + width,  bull_p, width, label="Bull (Q90)",
           color=_CLR_HOLD, alpha=0.8, edgecolor=_CLR_BG)

    ax.axhline(pos.entry_price_usd,   color=_CLR_REDUCE, linewidth=1.5,
               linestyle="--", label=f"Entry ${pos.entry_price_usd:.5f}")
    ax.axhline(pos.current_price_usd, color=_CLR_ACCENT,  linewidth=1.5,
               linestyle="-.", label=f"Current ${pos.current_price_usd:.5f}")

    ax.set_title("Price Scenarios (USD)", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Price (USD)", fontsize=8)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:.4f}")
    )
    ax.legend(fontsize=7, facecolor=_CLR_PANEL,
              edgecolor=_CLR_MUTED, labelcolor=_CLR_TEXT)
    ax.grid(axis="y", alpha=0.12, color=_CLR_MUTED)


# ---------------------------------------------------------------------------
# Panel 6 — Decision per horizon
# ---------------------------------------------------------------------------

def _plot_decision_summary(ax: plt.Axes, decisions: list) -> None:
    labels  = [dr.horizon_label for dr in decisions]
    colours = [_DECISION_CLR.get(dr.decision, _CLR_MUTED) for dr in decisions]
    dec_lbl = [dr.decision.value for dr in decisions]

    y    = np.arange(len(labels))
    bars = ax.barh(y, [1] * len(labels), color=colours,
                   edgecolor=_CLR_BG, height=0.6)

    for bar, label in zip(bars, dec_lbl):
        ax.text(
            0.5, bar.get_y() + bar.get_height() / 2,
            label,
            ha="center", va="center",
            color=_CLR_BG, fontsize=10, fontweight="bold",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks([])
    ax.set_title("Decision per Horizon", fontsize=10, fontweight="bold")
    ax.invert_yaxis()

    legend_patches = [
        mpatches.Patch(color=_CLR_HOLD,   label="HOLD"),
        mpatches.Patch(color=_CLR_REDUCE, label="REDUCE"),
        mpatches.Patch(color=_CLR_EXIT,   label="EXIT"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=7,
              facecolor=_CLR_PANEL, edgecolor=_CLR_MUTED, labelcolor=_CLR_TEXT)


# ---------------------------------------------------------------------------
# Master dashboard — public API
# ---------------------------------------------------------------------------

def show_report_dashboard(
    pos          : PositionState,
    dist         : FittedDistribution,
    ev_results   : list,
    earn_results : list,
    decisions    : list,
    scenario_sets: list,
    save         : bool = True,
    output_dir   : str  = "outputs/figures",
    filename     : str  = "position_report_dashboard.png",
    show         : bool = True,
) -> plt.Figure:
    """
    Render a 6-panel matplotlib dashboard summarising the full
    position intelligence report.  Called after print_full_report().

    Parameters
    ----------
    show : bool
        If True  -> plt.show() is called (interactive window pops up).
        If False -> figure is returned without displaying.
    save : bool
        If True  -> figure is saved as *output_dir/filename*.
    """
    # ----------------------------------------------------------------
    # Switch to an interactive backend if we are currently on Agg
    # ----------------------------------------------------------------
    current_backend = matplotlib.get_backend().lower()
    if "agg" in current_backend:
        for backend in ("TkAgg", "Qt5Agg", "WXAgg"):
            try:
                matplotlib.use(backend)
                break
            except Exception:
                continue

    plt.rcParams.update({
        "font.family":   "DejaVu Sans",
        "font.size":      9,
        "axes.titlepad":  8,
    })

    # ----------------------------------------------------------------
    # Figure layout
    # ----------------------------------------------------------------
    fig = plt.figure(figsize=(18, 12), facecolor=_CLR_BG)
    fig.suptitle(
        f"LS-HFT V0.1  —  Position Intelligence Dashboard  |  {pos.ticker}",
        fontsize=15, fontweight="bold", color=_CLR_ACCENT, y=0.98,
    )

    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.45, wspace=0.35,
        left=0.06, right=0.97,
        top=0.93,  bottom=0.07,
    )

    ax_pnl      = fig.add_subplot(gs[0, 0])
    ax_horizon  = fig.add_subplot(gs[0, 1])
    ax_risk     = fig.add_subplot(gs[0, 2])
    ax_earn     = fig.add_subplot(gs[1, 0])
    ax_scenario = fig.add_subplot(gs[1, 1])
    ax_decision = fig.add_subplot(gs[1, 2])

    # ----------------------------------------------------------------
    # Render each panel
    # ----------------------------------------------------------------
    _plot_pnl_summary(ax_pnl, pos)
    _plot_horizon_ev(ax_horizon, ev_results, decisions)
    _plot_risk_metrics(ax_risk, ev_results)
    _plot_earn_yield(ax_earn, earn_results)
    _plot_price_scenarios(ax_scenario, scenario_sets, pos)
    _plot_decision_summary(ax_decision, decisions)

    _apply_dark_style(fig)

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------
    if save:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_path = out / filename
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_CLR_BG)
        print(f"\n  [Dashboard] Saved to: {save_path}")

    # ----------------------------------------------------------------
    # Display
    # ----------------------------------------------------------------
    if show:
        print("  [Dashboard] Opening interactive matplotlib window ...")
        plt.show()

    return fig
