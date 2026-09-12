# LS-HFT V0 — Low-Speed HFT Research Framework

> **Status:** V0 — Historical OHLCV research foundation. No trading decisions.

---

## Architecture

```
D → D_clean → F → S → R → V → G
```

| Symbol | Meaning |
|--------|---------|
| D | Raw OHLCV dataset |
| D_clean | Cleaned dataset |
| F | Feature set (returns, volatility, volume, microstructure) |
| S | Statistical analysis |
| R | Risk analysis |
| V | Visualisations |
| G | Saved outputs |

---

## Versioning Roadmap

| Version | Data | Focus |
|---------|------|-------|
| **V0** | Historical OHLCV | Returns, Volatility, Risk, Statistics |
| V1 | Real-time OHLCV | Event engine |
| V2 | Real bid/ask L2 | Spread, Order-flow, Imbalance |
| V3 | Signal engine | Expected value |
| V4 | Execution simulator | Costs, slippage, latency |
| V5 | Paper trading | — |
| V6 | Controlled real-money | — |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download raw data  (edit config/config.py first)
python scripts/download_data.py

# 3. Clean data
python scripts/process_data.py

# 4. Calculate features
python scripts/calculate_features.py

# 5. Generate plots & analysis
python scripts/generate_plots.py

# 6. Run tests
pytest tests/ -v

#7. if you want to analysis bera coin
cd d:\HFT_trading\LS-HFT
python scripts/bera_analysis.py

```

---

## Configuration

All parameters live in [`config/config.py`](config/config.py).
**No mathematical constants are hard-coded elsewhere.**

Key parameters:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `TICKER` | `RELIANCE.NS` | Yahoo Finance ticker |
| `PERIOD` | `10y` | Historical lookback |
| `INTERVAL` | `1d` | Observation frequency |
| `RETURN_WINDOW` | `20` | Rolling return window (n) |
| `VOLATILITY_WINDOW` | `20` | Rolling volatility window (n) |
| `VOLUME_WINDOW` | `20` | Volume z-score window (n) |
| `VAR_CONFIDENCE` | `0.95` | α for VaR and ES |
| `TRADING_PERIODS_PER_YEAR` | `252` | Annualisation factor (auto-set from INTERVAL) |

---

## Mathematical Dependency Graph

```
                  PRICE (P_t = C_t)
                        │
                        ▼
                ┌───────────────┐
                │    RETURNS    │
                └───────┬───────┘
                        │
            ┌───────────┼───────────┐
            ▼           ▼           ▼
          MEAN      VARIANCE     QUANTILES
            │           │           │
            ▼           ▼           ▼
         RETURN     VOLATILITY     VaR
       STATISTICS       │           │
                        │           ▼
                        │           ES
                        ▼
                  RISK ANALYSIS
                        │
               ┌────────┴────────┐
               ▼                 ▼
           WEALTH CURVE      DRAWDOWN
               └────────┬────────┘
                        ▼
                  RISK REPORT

VOLUME ──► rolling mean ──► rolling std ──► z-score ──► spike

OHLCV ──► CLV proxy, Relative Volume proxy
          (true L2 microstructure reserved for V2)
```

---

## Module Responsibilities

| Module | Mathematics | Role |
|--------|------------|------|
| `config/config.py` | None — parameters only | Single source of truth |
| `src/data/downloader.py` | None | External data → D_raw |
| `src/data/loader.py` | None | CSV → validated DataFrame |
| `src/data/cleaner.py` | None | D_raw → D_clean |
| `src/features/returns.py` | R_t, r_t, CR_t, R_t^(n) | Return calculations |
| `src/features/volatility.py` | σ_t^(n), σ_annual, regime | Volatility |
| `src/features/volume.py` | Z_t^V, spike detection | Volume features |
| `src/features/microstructure.py` | CLV, RV (proxies only) | OHLCV microstructure |
| `src/analysis/statistics.py` | μ, s, skew, excess kurtosis, quantiles | Distribution |
| `src/analysis/risk.py` | VaR, ES, MDD, σ_down | Risk |
| `src/visualization/` | None | Charts |
| `scripts/` | None | Orchestration |
| `tests/` | Mathematical invariants | Verification |

---

## Key Design Rules

1. **Every formula lives in one module.** Scripts orchestrate; modules compute.
2. **No silent data modification.** The cleaner reports everything it removes.
3. **Annualisation is not hard-coded.** `config.TRADING_PERIODS_PER_YEAR` is derived from `INTERVAL`.
4. **Excess kurtosis is explicitly labelled.** Normal distribution = 0 is stated in output.
5. **V0 has no trading decisions.** All features are descriptive.
6. **No fabricated L2 data.** `microstructure.py` contains OHLCV proxies and an explicit `NOT_IMPLEMENTED` guard for true order-book imbalance.
