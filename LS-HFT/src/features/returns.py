# =============================================================================
# src/features/returns.py
#
# Mathematical specification
# ──────────────────────────
#
# Primary price variable:
#     P_t = C_t   (closing price)
#
# 5.1  Simple return
#     R_t = (P_t - P_{t-1}) / P_{t-1}   =   P_t / P_{t-1} - 1
#     Units    : dimensionless (fraction)
#     Domain   : R_t ∈ (-1, +∞)   (prices are positive, so R_t > -1)
#     Edge case: P_{t-1} = 0 → undefined; guaranteed away by cleaner.py
#
# 5.2  Log return  (primary representation for quant analysis)
#     r_t = ln(P_t / P_{t-1})
#     Relationship to simple return:
#         r_t = ln(1 + R_t)   ≈ R_t  for |R_t| << 1
#     Additive over time:
#         r_{1..n} = Σ r_t       (vs. compounding for simple returns)
#     Units    : dimensionless (nats)
#     Domain   : r_t ∈ (-∞, +∞)
#
# 5.3  Cumulative return
#     W_0 = 1  (normalised starting wealth)
#     W_t = W_0 × Π_{i=1}^{t} (1 + R_i)
#     CR_t = W_t / W_0 - 1  =  Π_{i=1}^{t} (1 + R_i) - 1
#
# 5.4  Rolling n-period return
#     Simple :  R_t^(n) = P_t / P_{t-n} - 1
#     Log    :  r_t^(n) = ln(P_t / P_{t-n})
#     For t < n, value is NaN (insufficient history).
# =============================================================================

import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simple_return(prices: pd.Series) -> pd.Series:
    """
    R_t = P_t / P_{t-1} - 1

    Inputs:
        prices : pd.Series of P_t values (strictly positive, ascending index)

    Output:
        pd.Series of R_t.  First element is NaN (no prior observation).

    Invariant:
        If P_t == P_{t-1}, then R_t == 0.
        If P_t > P_{t-1}, then R_t > 0.
        If P_t < P_{t-1}, then -1 < R_t < 0  (since prices are positive).

    Numerical test:
        P_{t-1} = 100, P_t = 110  →  R_t = 0.10  (10%)
    """
    r = prices.pct_change()
    r.name = "simple_return"
    return r


def log_return(prices: pd.Series) -> pd.Series:
    """
    r_t = ln(P_t / P_{t-1})

    Inputs:
        prices : pd.Series of P_t values (strictly positive)

    Output:
        pd.Series of r_t.  First element is NaN.

    Relationship to simple return:
        r_t = ln(1 + R_t)

    Invariant:
        If P_t == P_{t-1}, then r_t == 0.
        r_t is real-valued for all P_t > 0.

    Numerical test:
        P_{t-1} = 100, P_t = 110  →  r_t = ln(1.1) ≈ 0.09531
    """
    r = np.log(prices / prices.shift(1))
    r.name = "log_return"
    return r


def cumulative_return(simple_returns: pd.Series) -> pd.Series:
    """
    CR_t = Π_{i=1}^{t} (1 + R_i) - 1

    Starting from a normalised wealth W_0 = 1.

    Inputs:
        simple_returns : pd.Series of R_t (NaN for t=0 is handled).

    Output:
        pd.Series of CR_t.  CR_0 = 0 (no change yet).

    Mathematical derivation:
        W_t = W_0 × (1+R_1) × (1+R_2) × ... × (1+R_t)
        CR_t = W_t / W_0 - 1

    Equivalent log-space computation (numerically stable):
        CR_t = exp( Σ r_i ) - 1

    Invariant:
        If all R_t = 0, then CR_t = 0 for all t.
        CR_t >= -1 (total capital cannot drop below zero).
    """
    valid = simple_returns.dropna()
    cr = (1 + valid).cumprod() - 1
    # Re-index to original index so that NaN rows appear at the top
    cr = cr.reindex(simple_returns.index)
    cr.name = "cumulative_return"
    return cr


def rolling_return(prices: pd.Series, window: int) -> pd.Series:
    """
    R_t^(n) = P_t / P_{t-n} - 1

    Inputs:
        prices : pd.Series of P_t
        window : n (integer ≥ 1)

    Output:
        pd.Series.  Values are NaN for t < n (insufficient history).

    Invariant:
        rolling_return(prices, 1) == simple_return(prices)
        (modulo the index alignment).

    Numerical test:
        prices = [100, 102, 101, 105, 110]  window=2
        At t=2: P_t=101, P_{t-2}=100  →  R_t^(2) = 0.01  (1%)
    """
    r = prices / prices.shift(window) - 1
    r.name = f"rolling_return_{window}"
    return r


def rolling_log_return(prices: pd.Series, window: int) -> pd.Series:
    """
    r_t^(n) = ln(P_t / P_{t-n})

    Inputs:
        prices : pd.Series of P_t
        window : n (integer ≥ 1)

    Output:
        pd.Series.  Values are NaN for t < n.
    """
    r = np.log(prices / prices.shift(window))
    r.name = f"rolling_log_return_{window}"
    return r


def compute_all(prices: pd.Series, window: int = 20) -> pd.DataFrame:
    """
    Convenience: compute all return features and return as a single DataFrame.

    Columns produced:
        simple_return
        log_return
        cumulative_return
        rolling_return_{window}
        rolling_log_return_{window}
    """
    sr = simple_return(prices)
    lr = log_return(prices)
    cr = cumulative_return(sr)
    rr = rolling_return(prices, window)
    rlr = rolling_log_return(prices, window)

    df = pd.DataFrame({
        "simple_return":          sr,
        "log_return":             lr,
        "cumulative_return":      cr,
        f"rolling_return_{window}":     rr,
        f"rolling_log_return_{window}": rlr,
    })

    logger.info(
        f"Returns computed: {len(df)} rows, window={window}  "
        f"(first {window} rows have NaN rolling values)"
    )
    return df
