# =============================================================================
# scripts/bera_analysis.py
#
# LS-HFT V0.1 — BERA Position Intelligence
#
# Orchestration only — contains NO mathematics.
#
# Full pipeline:
#   1. Download BERA-USD hourly data          (yfinance)
#   2. Clean it                               (cleaner.py)
#   3. Compute log returns                    (features/returns.py)
#   4. Fit return distribution                (forecasting/return_distribution.py)
#   5. Build position state from raw inputs   (position/portfolio.py)
#   6. Compute earn yield across horizons     (position/earn.py)
#   7. Compute earn-adjusted EV               (decision/expected_value.py)
#   8. Build price scenarios                  (forecasting/price_scenarios.py)
#   9. Apply decision engine                  (decision/hold_decision.py)
#  10. Print full report                      (reporting/position_report.py)
# =============================================================================

import sys
import io
# Force UTF-8 on Windows cp1252 console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    BERA_TICKER, BERA_PERIOD, BERA_INTERVAL,
    COST_BASIS, QUANTITY, CURRENT_VALUE_INR, INR_USD_RATE,
    EARN_APR, CUMULATIVE_INTEREST_BERA, EARN_PERIODS_PER_YEAR,
    HORIZONS_HOURS, HORIZON_LABELS,
    N_SIMULATIONS, RANDOM_SEED,
    VAR_CONFIDENCE,
    EV_THRESHOLD_PCT, PROB_LOSS_MAX, ES_FRACTION_MAX,
    TRANSACTION_COST_PCT,
    RAW_DATA_PATH,
)

from src.data.cleaner                    import clean
from src.features.returns                import log_return
from src.position.portfolio              import build_position
from src.position.earn                   import compute_earn_all_horizons
from src.forecasting.return_distribution import fit as fit_distribution
from src.forecasting.price_scenarios     import build_scenarios
from src.decision.expected_value         import compute_all_horizons
from src.decision.hold_decision          import evaluate_all
from src.reporting.position_report       import print_full_report
from src.visualization.report_charts     import show_report_dashboard

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")


# ---------------------------------------------------------------------------
# Fetch BERA data
# ---------------------------------------------------------------------------

def _fetch_bera(ticker: str, period: str, interval: str) -> pd.DataFrame:
    print(f"  Fetching {ticker}  period={period}  interval={interval} ...", end=" ", flush=True)
    try:
        raw = yf.download(
            tickers=ticker, period=period, interval=interval,
            auto_adjust=True, progress=False,
        )
        if raw.empty:
            raise ValueError("Empty response from yfinance.")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.index.name = "Date"
        print(f"OK  ({len(raw)} rows)")
        return raw
    except Exception as e:
        saved = list((PROJECT_ROOT / RAW_DATA_PATH).glob(
            f"{ticker.replace('-', '_').replace('.', '_')}*.csv"
        ))
        if saved:
            path = max(saved, key=lambda p: p.stat().st_mtime)
            print(f"FALLBACK -> {path.name}")
            return pd.read_csv(path, index_col=0, parse_dates=True)
        raise RuntimeError(
            f"Could not fetch {ticker} from yfinance and no cache found.  Error: {e}"
        ) from e


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 68)
    print("  LS-HFT V0.1 -- BERA Position Intelligence")
    print("=" * 68)

    # ----------------------------------------------------------------
    # Step 1 — Build position from raw user inputs
    #   All derived values (entry price, P&L, return) computed in portfolio.py
    # ----------------------------------------------------------------
    print("\n[1/7] Building position from raw inputs")
    pos = build_position(
        ticker="BERA-USD",
        quantity=QUANTITY,
        cost_basis_inr=COST_BASIS,
        current_value_inr=CURRENT_VALUE_INR,
        earn_apr=EARN_APR,
        cumulative_interest_bera=CUMULATIVE_INTEREST_BERA,
        inr_usd_rate=INR_USD_RATE,
        tx_cost_pct=TRANSACTION_COST_PCT,
    )
    print(f"  Q        = {pos.quantity:.8f} BERA")
    print(f"  P0 (INR) = Rs{pos.entry_price_inr:.4f}  "
          f"(= Rs{COST_BASIS:.2f} / {QUANTITY:.8f} / {INR_USD_RATE})")
    print(f"  Pt (INR) = Rs{pos.current_price_inr:.4f}  "
          f"(= Rs{CURRENT_VALUE_INR:.2f} / {QUANTITY:.8f} / {INR_USD_RATE})")
    print(f"  P0 (USD) = ${pos.entry_price_usd:.6f}")
    print(f"  Pt (USD) = ${pos.current_price_usd:.6f}")
    print(f"  Unrealised P&L = Rs{pos.unrealised_pnl_inr:.4f}  ({pos.return_pct:+.4%})")

    # ----------------------------------------------------------------
    # Step 2 — Download historical data for distribution fitting
    # ----------------------------------------------------------------
    print("\n[2/7] Fetching historical BERA data for distribution")
    raw = _fetch_bera(BERA_TICKER, BERA_PERIOD, BERA_INTERVAL)

    # ----------------------------------------------------------------
    # Step 3 — Clean
    # ----------------------------------------------------------------
    print("[3/7] Cleaning data")
    df, report = clean(raw)
    print(f"  {report.rows_output:,} rows retained  (removed {report.rows_input - report.rows_output})")

    # ----------------------------------------------------------------
    # Step 4 — Fit return distribution
    # ----------------------------------------------------------------
    print("[4/7] Fitting return distribution")
    lr   = log_return(df["Close"])
    dist = fit_distribution(lr.dropna(), period_label=BERA_INTERVAL)
    ann_vol = dist.sigma_period * (8760 ** 0.5)
    print(f"  n={dist.n_obs:,}  mu={dist.mu_period:.6f}  sg={dist.sigma_period:.6f}  "
          f"ann.vol={ann_vol:.1%}")

    # Use position's derived Pt (USD) as current price for distribution
    # This is more accurate than live market price for position accounting.
    p_current_usd = pos.current_price_usd
    p_entry_usd   = pos.entry_price_usd
    print(f"  Using P_current = ${p_current_usd:.6f}  (derived from V_inr / Q / rate)")

    # ----------------------------------------------------------------
    # Step 5 — Earn yield across all horizons (deterministic)
    # ----------------------------------------------------------------
    print("[5/7] Computing earn yield per horizon")
    earn_results = compute_earn_all_horizons(
        quantity=QUANTITY,
        current_price_inr=pos.current_price_inr,
        horizons_h=HORIZONS_HOURS,
        horizon_labels=HORIZON_LABELS,
        earn_apr=EARN_APR,
        periods_per_year=EARN_PERIODS_PER_YEAR,
        cumulative_interest=CUMULATIVE_INTEREST_BERA,
        inr_usd_rate=INR_USD_RATE,
    )
    for er in earn_results:
        print(f"  {er.horizon_label:<12}: {er.earn_tokens:.8f} BERA  "
              f"(Rs{er.earn_value_inr:.6f}  {er.effective_rate_h:.4%})")

    # ----------------------------------------------------------------
    # Step 6 — Monte Carlo EV across all horizons
    # ----------------------------------------------------------------
    print("\n[6/7] Running Monte Carlo EV analysis ...")
    ev_results = compute_all_horizons(
        dist=dist,
        horizons_h=HORIZONS_HOURS,
        horizon_labels=HORIZON_LABELS,
        p_current_usd=p_current_usd,
        p_entry_usd=p_entry_usd,
        quantity=QUANTITY,
        inr_usd_rate=INR_USD_RATE,
        tx_cost_pct=TRANSACTION_COST_PCT,
        earn_results=earn_results,
        alpha=VAR_CONFIDENCE,
        n_sim=N_SIMULATIONS,
        seed=RANDOM_SEED,
    )

    # ----------------------------------------------------------------
    # Step 7 — Price scenarios + decision engine
    # ----------------------------------------------------------------
    print("[7/7] Building scenarios and decision ...")
    scenario_sets = [
        build_scenarios(
            dist=dist, h=h, horizon_label=label,
            p_current=p_current_usd, p_entry=p_entry_usd,
            quantity=QUANTITY, inr_usd_rate=INR_USD_RATE,
        )
        for h, label in zip(HORIZONS_HOURS, HORIZON_LABELS)
    ]

    from src.decision.expected_value import EVResult as _EVR

    # Wrap earn-adjusted metrics into the shape hold_decision.evaluate() expects.
    # evaluate() reads: .expected_return, .prob_loss, .es_fraction, .horizon_label, .prob_profit
    # We supply the EARN-ADJUSTED values so the decision is made on the full picture.
    adapted = []
    for ev in ev_results:
        dummy = _EVR.__new__(_EVR)
        dummy.__dict__.update(ev.__dict__)
        # Override the fields evaluate() will read with earn-adjusted versions:
        dummy.expected_return = ev.expected_earn_return_pct
        dummy.prob_loss       = ev.prob_loss_earn
        dummy.prob_profit     = ev.prob_profit_earn
        adapted.append(dummy)

    decisions = evaluate_all(
        ev_results=adapted,
        ev_threshold_pct=EV_THRESHOLD_PCT,
        prob_loss_max=PROB_LOSS_MAX,
        es_fraction_max=ES_FRACTION_MAX,
        n_obs=dist.n_obs,
    )


    # ----------------------------------------------------------------
    # Print full report
    # ----------------------------------------------------------------
    print_full_report(
        pos=pos,
        dist=dist,
        ev_results=ev_results,
        earn_results=earn_results,
        decisions=decisions,
        scenario_sets=scenario_sets,
    )

    # ----------------------------------------------------------------
    # Visual dashboard (matplotlib) — triggered after report completes
    # ----------------------------------------------------------------
    from config.config import FIGURES_PATH
    show_report_dashboard(
        pos=pos,
        dist=dist,
        ev_results=ev_results,
        earn_results=earn_results,
        decisions=decisions,
        scenario_sets=scenario_sets,
        save=True,
        output_dir=str(PROJECT_ROOT / FIGURES_PATH),
        filename="position_report_dashboard.png",
        show=True,
    )


if __name__ == "__main__":
    main()
