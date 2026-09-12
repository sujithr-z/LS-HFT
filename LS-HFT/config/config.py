# =============================================================================
# LS-HFT V0 — Global Configuration
#
# Mathematical role:
#   This file is the single source of truth for all parameters.
#   No formula elsewhere in the codebase should hard-code a numeric
#   constant that lives here.
#
# Rule: every module imports from this file; no module duplicates a value.
# =============================================================================

# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------

TICKER: str = "RELIANCE.NS"   # Yahoo Finance ticker
PERIOD: str = "10y"            # historical lookback passed to yfinance
INTERVAL: str = "1d"           # observation frequency: "1d" | "1wk" | "1mo"

# ---------------------------------------------------------------------------
# Feature windows
#
#   N_return   = 20  →  rolling window for return aggregation
#   N_vol      = 20  →  rolling window for volatility
#   N_volume   = 20  →  rolling window for volume z-score
# ---------------------------------------------------------------------------

RETURN_WINDOW: int   = 20    # n in R_t^(n) = P_t/P_{t-n} - 1
VOLATILITY_WINDOW: int = 20  # n in rolling σ_t^(n)
VOLUME_WINDOW: int   = 20    # n in rolling V̄_t^(n)

# ---------------------------------------------------------------------------
# Annualisation factor
#
#   Depends on INTERVAL:
#       "1d"  → 252 trading days / year
#       "1wk" → 52  weeks / year
#       "1mo" → 12  months / year
#
#   σ_annual = σ_period × sqrt(TRADING_PERIODS_PER_YEAR)
# ---------------------------------------------------------------------------

_ANNUALIZATION_MAP: dict = {
    "1d":  252,
    "1wk": 52,
    "1mo": 12,
}

TRADING_PERIODS_PER_YEAR: int = _ANNUALIZATION_MAP.get(INTERVAL, 252)

# ---------------------------------------------------------------------------
# Risk parameters
#
#   alpha  = 0.95  →  confidence level for VaR and ES
#   T      = 0.0   →  target return for downside deviation
#   W0     = 1.0   →  normalised starting capital (dimensionless)
# ---------------------------------------------------------------------------

VAR_CONFIDENCE: float = 0.95   # α: P( R > VaR ) = α
ES_CONFIDENCE: float  = 0.95   # same level used for Expected Shortfall

DOWNSIDE_TARGET: float = 0.0   # T in min(R_t - T, 0)
INITIAL_WEALTH: float  = 1.0   # W_0, normalised

# ---------------------------------------------------------------------------
# Volume spike threshold
#
#   Z_t^V > VOLUME_SPIKE_Z  →  spike
# ---------------------------------------------------------------------------

VOLUME_SPIKE_Z: float = 2.0

# ---------------------------------------------------------------------------
# File paths  (relative to project root)
# ---------------------------------------------------------------------------

RAW_DATA_PATH: str       = "data/raw"
PROCESSED_DATA_PATH: str = "data/processed"
METADATA_PATH: str       = "data/metadata"

FIGURES_PATH: str    = "outputs/figures"
STATISTICS_PATH: str = "outputs/statistics"
REPORTS_PATH: str    = "outputs/reports"

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

FIGURE_DPI: int = 150
FIGURE_SIZE: tuple = (14, 6)


# ===========================================================================
# V0.1 — BERA Position Intelligence
# ===========================================================================

# ---------------------------------------------------------------------------
# BERA market data
# ---------------------------------------------------------------------------

BERA_TICKER: str   = "BERA-USD"
BERA_PERIOD: str   = "60d"       # short lookback; BERA is a new asset
BERA_INTERVAL: str = "1h"        # hourly bars for crypto

# ---------------------------------------------------------------------------
# Position definition
#
# COST_BASIS_MODE controls how COST_BASIS is interpreted:
#
#   "total_inr"   : COST_BASIS is total INR invested (e.g., ₹350 total)
#                   P_0_usd = COST_BASIS / (QUANTITY × INR_USD_RATE)
#
#   "total_usd"   : COST_BASIS is total USD invested
#                   P_0_usd = COST_BASIS / QUANTITY
#
#   "per_unit_usd": COST_BASIS is the USD price you paid per BERA token
#                   P_0_usd = COST_BASIS  (QUANTITY × P_0_usd = total cost)
#
# Example:
#   You invested ₹350 total.
#   At ₹84/USD rate, that is $4.17.
#   At $0.193/BERA entry price, you hold ≈ 21.6 BERA.
#   COST_BASIS_MODE = "total_inr"
#   COST_BASIS = 350.0
#   QUANTITY   = 21.6
# ---------------------------------------------------------------------------

COST_BASIS_MODE: str  = "total_inr"
COST_BASIS: float     = 350.0      # units depend on COST_BASIS_MODE
QUANTITY: float       = 19.36757634  # number of BERA tokens held
CURRENT_VALUE_INR: float = 358.42   # current marked value of the position
CUMULATIVE_INTEREST_BERA: float = 0.00009685  # Earn interest already received
EARN_APR: float       = 0.1382     # annual Earn opportunity rate
INR_USD_RATE: float   = 84.0       # ₹ per USD  (update to live rate)

# ---------------------------------------------------------------------------
# Decision thresholds (rule-based — must be explicit, not arbitrary AI)
#
#   EV_THRESHOLD_PCT  : minimum expected return % at horizon to consider HOLD
#   PROB_LOSS_MAX     : maximum acceptable P(Π_T < 0) for a pure HOLD
#   ES_FRACTION_MAX   : maximum ES as a fraction of current position value for HOLD
#                       (e.g., 0.30 → willing to tolerate up to 30% tail loss)
# ---------------------------------------------------------------------------

EV_THRESHOLD_PCT: float = 0.0     # 0% → any positive EV qualifies for HOLD
PROB_LOSS_MAX: float    = 0.60    # 60% → still HOLD if P(loss) < 60%
ES_FRACTION_MAX: float  = 0.30    # 30% → EXIT/REDUCE if ES > 30% of position

# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------

N_SIMULATIONS: int  = 50_000     # number of future price paths per horizon
RANDOM_SEED: int    = 42

# ---------------------------------------------------------------------------
# Analysis horizons
#
# Unit: hours (matches BERA_INTERVAL = "1h")
# ---------------------------------------------------------------------------

HORIZONS_HOURS: list = [1, 6, 24, 72, 168, 720]
HORIZON_LABELS: list = ["1 hour", "6 hours", "24 hours", "3 days", "7 days", "30 days"]

# ---------------------------------------------------------------------------
# Earn parameters
#
# earn_apr                  : annual percentage rate from the Earn/staking product
# cumulative_interest_bera  : BERA already accrued (not yet claimed)
#
# Earn yield for horizon h hours:
#   Y_h = Q x P_t x ((1 + APR)^(h/N) - 1)
#   where N = EARN_PERIODS_PER_YEAR (8760 for hourly compounding)
#
# Opportunity cost of exiting at time 0:
#   OC_h = -Y_h   (the yield that would have been earned but is now forfeited)
# ---------------------------------------------------------------------------

EARN_APR: float                  = 0.1382       # 13.82% p.a.
CUMULATIVE_INTEREST_BERA: float  = 0.00009685   # BERA earned so far
EARN_PERIODS_PER_YEAR: int       = 8760          # hourly compounding base

# ---------------------------------------------------------------------------
# Transaction cost model
# TRANSACTION_COST_PCT : one-way cost as fraction (0.002 = 0.2%)
# ---------------------------------------------------------------------------

TRANSACTION_COST_PCT: float = 0.002   # 0.2% one-way exit cost
