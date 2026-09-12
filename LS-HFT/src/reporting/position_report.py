# =============================================================================
# src/reporting/position_report.py
#
# Pipeline role:
#   (PositionState, FittedDistribution, [EVResult], [EarnResult],
#    [ScenarioSet])  ->  formatted terminal report
#
# This module contains NO mathematics.
# =============================================================================

import sys
from pathlib import Path

# Force UTF-8 on Windows cp1252 console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.position.portfolio              import PositionState
from src.position.earn                   import EarnResult, breakeven_price_inr
from src.forecasting.return_distribution import FittedDistribution
from src.forecasting.price_scenarios     import ScenarioSet
from src.decision.expected_value         import EVResult
from src.decision.hold_decision          import DecisionResult, Decision, consensus_decision
from config.config                       import VAR_CONFIDENCE, TRANSACTION_COST_PCT

# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_WHITE  = "\033[97m"

_DECISION_COLOUR = {
    Decision.HOLD:              _GREEN,
    Decision.REDUCE:            _YELLOW,
    Decision.EXIT:              _RED,
    Decision.INSUFFICIENT_DATA: _YELLOW,
}

def _col(text: str, colour: str) -> str:
    return f"{colour}{_BOLD}{text}{_RESET}"

def _pct(v: float, decimals: int = 3) -> str:
    fmt = f"+.{decimals}%"
    return f"{v:{fmt}}"

def _rs(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"Rs{sign}{v:.2f}"

def _usd(v: float) -> str:
    return f"${v:.6f}"


# ---------------------------------------------------------------------------
# Section header
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print(f"\n{_col(title, _CYAN)}")
    print("-" * 68)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def print_header(ticker: str) -> None:
    w = 68
    print()
    print("+" + "=" * (w - 2) + "+")
    print(f"|{'LS-HFT V0.1 -- POSITION INTELLIGENCE REPORT'.center(w - 2)}|")
    print(f"|{ticker.center(w - 2)}|")
    print("+" + "=" * (w - 2) + "+")


# ---------------------------------------------------------------------------
# Position summary
# ---------------------------------------------------------------------------

def print_position_summary(pos: PositionState) -> None:
    _section("POSITION ACCOUNTING  (all values derived from raw inputs)")
    pnl_col = _GREEN if pos.unrealised_pnl_inr >= 0 else _RED

    print(f"  Raw inputs")
    print(f"    Quantity (Q)        : {pos.quantity:.8f} BERA  (principal)")
    print(f"    Cost basis          : Rs{pos.cost_basis_inr:.2f}  (total INR paid)")
    print(f"    Current value       : Rs{pos.current_value_inr:.2f}  (from platform)")
    print(f"    Earn APR            : {pos.earn_apr:.4%}")
    print(f"    Accrued interest    : {pos.cumulative_interest_bera:.8f} BERA")
    print()
    print(f"  Derived prices  (computed = never hard-coded)")
    print(f"    Entry price (P0)    : {_usd(pos.entry_price_usd)}  "
          f"(= Rs{pos.cost_basis_inr:.2f} / {pos.quantity:.4f} / {pos.inr_usd_rate})")
    print(f"    Current price (Pt)  : {_usd(pos.current_price_usd)}  "
          f"(= Rs{pos.current_value_inr:.2f} / {pos.quantity:.4f} / {pos.inr_usd_rate})")
    print(f"    INR per USD rate    : {pos.inr_usd_rate}")
    print()
    print(f"  P&L")
    print(f"    Unrealised P&L      : "
          f"{_col(_rs(pos.unrealised_pnl_inr), pnl_col)}  "
          f"({_col(_pct(pos.return_pct), pnl_col)})")
    print(f"    Exit transaction cost: Rs{pos.tx_cost_inr:.4f}  ({pos.tx_cost_pct:.2%} one-way)")
    print(f"    Net P&L if exit now : {_col(_rs(pos.exit_pnl_inr), pnl_col)}")
    print()
    print(f"  Earn accrued")
    print(f"    Interest earned     : {pos.cumulative_interest_bera:.8f} BERA  "
          f"(= Rs{pos.accrued_value_inr:.4f})")
    print(f"    Total position value: Rs{pos.total_value_inr:.4f}  (principal + accrued)")


# ---------------------------------------------------------------------------
# Distribution summary
# ---------------------------------------------------------------------------

def print_distribution_summary(dist: FittedDistribution) -> None:
    _section("RETURN DISTRIBUTION  (empirical, 60-day hourly BERA data)")
    ann_vol = dist.sigma_period * (8760 ** 0.5)
    print(f"  Observations fitted : {dist.n_obs:,}")
    print(f"  Period              : {dist.period_label}")
    print(f"  Mean (mu) per period: {_pct(dist.mu_period, 4)}")
    print(f"  Std  (sg) per period: {_pct(dist.sigma_period, 4)}")
    print(f"  Annualised vol      : {ann_vol:.1%}  (= sg x sqrt(8760))")
    print(f"  Price model         : Log-normal  P(t+h) = P(t) x exp(r_h)")
    print(f"  r_h ~ N(mu x h,  sg x sqrt(h))")


# ---------------------------------------------------------------------------
# Earn summary
# ---------------------------------------------------------------------------

def print_earn_summary(
    earn_results: list,
    pos: PositionState,
) -> None:
    _section("EARN OPPORTUNITY COST  (Y_h = Q x P_t x ((1+APR)^(h/N) - 1))")
    print(f"  Earn APR            : {pos.earn_apr:.4%}  per year")
    print(f"  Compounding base    : 8760 periods/year (hourly)")
    print(f"  Current price (Pt)  : Rs{pos.current_price_inr:.4f} per BERA")
    print()
    print(f"  {'Horizon':<12}  {'Earn BERA':>12}  {'Earn Rs':>10}  {'Earn %':>8}")
    print("  " + "-" * 50)
    for er in earn_results:
        print(f"  {er.horizon_label:<12}  "
              f"{er.earn_tokens:>12.8f}  "
              f"Rs{er.earn_value_inr:>8.4f}  "
              f"{_pct(er.effective_rate_h, 4):>8}")
    print()
    print(f"  Already accrued: {pos.cumulative_interest_bera:.8f} BERA = "
          f"Rs{pos.accrued_value_inr:.6f}")


# ---------------------------------------------------------------------------
# Horizon table (full)
# ---------------------------------------------------------------------------

def print_horizon_table(
    ev_results : list,
    earn_results: list,
    decisions  : list,
    alpha      : float = VAR_CONFIDENCE,
) -> None:
    _section(f"HORIZON ANALYSIS  (alpha={alpha:.0%}, n={ev_results[0].n_sim:,} paths)")

    # Column header
    hdr = (f"  {'Horizon':<12}  {'E[Price]':>9}  {'Earn':>7}  "
           f"{'E[Earn+]':>9}  {'P(+)':>6}  "
           f"{'VaR Rs':>8}  {'ES Rs':>8}  "
           f"{'DeltaEV':>9}  {'Decision':>10}")
    print(hdr)
    print("  " + "-" * 105)

    for ev, er, dr in zip(ev_results, earn_results, decisions):
        d_col  = _DECISION_COLOUR.get(dr.decision, _WHITE)
        d_str  = _col(f"{dr.decision.value:<8}", d_col)
        dv_col = _GREEN if ev.delta_ev_inr >= 0 else _RED

        row = (
            f"  {ev.horizon_label:<12}  "
            f"{_pct(ev.expected_return_pct, 2):>9}  "
            f"Rs{er.earn_value_inr:>5.3f}  "
            f"{_pct(ev.expected_earn_return_pct, 2):>9}  "
            f"{ev.prob_profit_earn:>6.1%}  "
            f"Rs{ev.var_inr:>6.2f}  "
            f"Rs{ev.es_inr:>6.2f}  "
            f"{_col(_rs(ev.delta_ev_inr), dv_col):>9}  "
            f"{d_str}"
        )
        print(row)

    print()
    print("  Columns:")
    print("    E[Price]  = expected return from price movement alone")
    print("    Earn      = deterministic earn yield (opportunity cost of exit)")
    print("    E[Earn+]  = earn-adjusted expected return")
    print("    P(+)      = P(earn-adjusted P&L > 0)")
    print("    VaR/ES    = downside risk on earn-adjusted P&L")
    print("    DeltaEV   = E[earn-adjusted P&L] - exit_pnl_now  (>0 = HOLD better)")


# ---------------------------------------------------------------------------
# EV decomposition for one horizon
# ---------------------------------------------------------------------------

def print_ev_decomposition(
    ev: EVResult,
    er: EarnResult,
) -> None:
    _section(f"EV DECOMPOSITION -- {ev.horizon_label}")
    p   = ev.prob_profit_earn
    q   = ev.prob_loss_earn
    G   = ev.cond_gain_inr
    L   = ev.cond_loss_inr

    print(f"  --- Price component ---")
    print(f"    E[price P&L]        : {_rs(ev.expected_pnl_inr)}")
    print(f"    P(price profit)     : {ev.prob_profit_price:.1%}")
    print()
    print(f"  --- Earn yield (deterministic) ---")
    print(f"    Y_h                 : Rs{er.earn_value_inr:.6f}  ({_pct(er.effective_rate_h, 4)})")
    print(f"    Breakeven lowered by: Rs{er.earn_value_inr:.6f}  (earn reduces bar)")
    print()
    print(f"  --- Earn-adjusted P&L ---")
    print(f"    E[earn+price P&L]   : {_col(_rs(ev.expected_earn_pnl_inr), _GREEN if ev.expected_earn_pnl_inr >= 0 else _RED)}")
    print(f"    P(earn-adj profit)  : {p:.1%}")
    print(f"    P(earn-adj loss)    : {q:.1%}")
    print()
    print(f"  --- EV = P(+) * E[gain|+]  -  P(-) * E[loss|-] ---")
    print(f"     = {p:.3f} * Rs{G:.2f}  -  {q:.3f} * Rs{L:.2f}")
    print(f"     = Rs{p*G:.2f}  -  Rs{q*L:.2f}")
    print(f"     = {_col(_rs(ev.expected_earn_pnl_inr), _GREEN if ev.expected_earn_pnl_inr >= 0 else _RED)}")
    print()
    print(f"  --- Hold vs Exit now ---")
    print(f"    Exit P&L (certain)  : {_rs(ev.exit_pnl_now_inr)}")
    print(f"    E[HOLD earn+ P&L]   : {_rs(ev.expected_earn_pnl_inr)}")
    better = "HOLD" if ev.delta_ev_inr >= 0 else "EXIT"
    col    = _GREEN if ev.delta_ev_inr >= 0 else _RED
    print(f"    Delta EV            : {_col(_rs(ev.delta_ev_inr), col)}  -> {_col(better, col)} has higher EV")


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def print_scenarios(
    scenario_sets: list,
    pos: PositionState,
    earn_results: list,
) -> None:
    _section("PRICE SCENARIOS  (quantiles of return distribution)")
    print(f"  {'Horizon':<12}  {'Bear Q10':>16}  {'Base Q50':>16}  {'Bull Q90':>16}")
    print("  " + "-" * 66)
    for ss, er in zip(scenario_sets, earn_results):
        bear_s = f"${ss.bear.price_usd:.4f} ({_pct(ss.bear.return_pct, 1)})"
        base_s = f"${ss.base.price_usd:.4f} ({_pct(ss.base.return_pct, 1)})"
        bull_s = f"${ss.bull.price_usd:.4f} ({_pct(ss.bull.return_pct, 1)})"
        print(f"  {ss.horizon_label:<12}  {bear_s:>16}  {base_s:>16}  {bull_s:>16}")

    # Breakeven prices for each horizon
    print()
    print(f"  Earn-adjusted breakeven prices (below Pt = hold still rational):")
    print(f"  {'Horizon':<12}  {'P0 (entry)':>12}  {'Pt (current)':>14}  {'P_break':>12}  {'vs Pt':>10}")
    print("  " + "-" * 60)
    for ss, er in zip(scenario_sets, earn_results):
        p_break = breakeven_price_inr(
            entry_price_inr=pos.entry_price_inr,
            earn_value_inr=er.earn_value_inr,
            quantity=pos.quantity,
            tx_cost_pct=pos.tx_cost_pct,
            current_price_inr=pos.current_price_inr,
        )
        diff    = p_break - pos.current_price_inr
        diff_s  = _col(f"Rs{diff:+.4f}", _GREEN if diff < 0 else _RED)
        print(f"  {ss.horizon_label:<12}  "
              f"Rs{pos.entry_price_inr:>10.4f}  "
              f"Rs{pos.current_price_inr:>12.4f}  "
              f"Rs{p_break:>10.4f}  "
              f"{diff_s}")
    print("  (P_break < Pt means earn yield alone makes holding rational even if price flat)")


# ---------------------------------------------------------------------------
# Final decision
# ---------------------------------------------------------------------------

def print_final_decision(decisions: list) -> None:
    top = consensus_decision(decisions)
    col = _DECISION_COLOUR.get(top, _WHITE)
    w   = 68

    print()
    print("+" + "=" * (w - 2) + "+")
    print(f"|{'OVERALL DECISION'.center(w - 2)}|")
    label = f"**  {top.value}  **"
    print(f"|{_col(label, col).center(w - 2 + 10)}|")
    print("+" + "=" * (w - 2) + "+")

    primary = next(
        (d for d in decisions if d.decision in (Decision.EXIT, Decision.REDUCE)),
        decisions[-1],
    )
    print(f"\n  Primary driver ({primary.horizon_label} horizon):")
    for line in primary.reason.split("  "):
        if line.strip():
            print(f"    >> {line.strip()}")

    print()
    print("  --- ANTI-BIAS STATEMENT ------------------------------------")
    print("  Decision derived from: E[Pi_earn], P(Pi<0), VaR, ES, DeltaEV")
    print("  The system will say EXIT if the math says EXIT.")
    print("  It is NOT influenced by your existing emotional attachment.")
    print("  -------------------------------------------------------------")


# ---------------------------------------------------------------------------
# Master report
# ---------------------------------------------------------------------------

def print_full_report(
    pos          : PositionState,
    dist         : FittedDistribution,
    ev_results   : list,
    earn_results : list,
    decisions    : list,
    scenario_sets: list,
) -> None:
    print_header(pos.ticker)
    print_position_summary(pos)
    print_distribution_summary(dist)
    print_earn_summary(earn_results, pos)
    print_horizon_table(ev_results, earn_results, decisions)

    # EV decomposition for 7-day horizon
    idx = 4 if len(ev_results) > 4 else -1
    print_ev_decomposition(ev_results[idx], earn_results[idx])

    print_scenarios(scenario_sets, pos, earn_results)
    print_final_decision(decisions)
    print()
