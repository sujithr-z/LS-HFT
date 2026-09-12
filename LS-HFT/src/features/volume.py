# =============================================================================
# src/features/volume.py
#
# Mathematical specification
# ──────────────────────────
#
# 7.1  Volume change:
#     ΔV_t  = V_t - V_{t-1}
#     g_t^V = (V_t - V_{t-1}) / V_{t-1}    (percentage volume change)
#
# 7.2  Rolling mean volume:
#     V̄_t^(n) = (1/n) × Σ_{i=t-n+1}^{t} V_i
#
# 7.3  Rolling volume standard deviation (ddof=1):
#     s_{V,t} = sqrt( (1/(n-1)) × Σ_{i=t-n+1}^{t} (V_i - V̄_t)² )
#
# 7.4  Volume z-score:
#     Z_t^V = (V_t - V̄_t) / s_{V,t}
#     Interpretation:
#         Z_t^V = 0  → normal volume
#         Z_t^V = 2  → volume ~2 std above recent mean
#
# 7.5  Volume spike (feature, NOT a trading signal):
#     spike_t = True  iff  Z_t^V > VOLUME_SPIKE_Z
#     Default VOLUME_SPIKE_Z = 2.0  (from config)
# =============================================================================

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config.config import VOLUME_WINDOW, VOLUME_SPIKE_Z

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def volume_change(volume: pd.Series) -> pd.DataFrame:
    """
    ΔV_t  = V_t - V_{t-1}
    g_t^V = (V_t - V_{t-1}) / V_{t-1}

    Inputs:
        volume : pd.Series of V_t (non-negative integers / floats)

    Output:
        DataFrame with columns:
            volume_change  (ΔV_t)
            volume_pct_change  (g_t^V)

    Edge case:
        If V_{t-1} = 0, g_t^V is undefined → NaN (not infinite).

    Invariant:
        ΔV_t = 0  iff  V_t = V_{t-1}.
    """
    dv = volume.diff()
    dv.name = "volume_change"

    gv = volume.pct_change()
    gv.name = "volume_pct_change"

    return pd.DataFrame({"volume_change": dv, "volume_pct_change": gv})


def rolling_mean_volume(
    volume: pd.Series,
    window: int = VOLUME_WINDOW,
) -> pd.Series:
    """
    V̄_t^(n) = (1/n) × Σ_{i=t-n+1}^{t} V_i

    Inputs:
        volume : pd.Series of V_t
        window : n

    Output:
        pd.Series.  First (window-1) values are NaN.
    """
    mv = volume.rolling(window=window, min_periods=window).mean()
    mv.name = f"volume_mean_{window}"
    return mv


def rolling_std_volume(
    volume: pd.Series,
    window: int = VOLUME_WINDOW,
) -> pd.Series:
    """
    s_{V,t} = sqrt( (1/(n-1)) × Σ_{i=t-n+1}^{t} (V_i - V̄_t)² )   ddof=1

    Inputs:
        volume : pd.Series of V_t
        window : n (≥ 2 for ddof=1)

    Output:
        pd.Series.  First (window-1) values are NaN.

    Invariant:
        If all V_i in the window are identical, s_{V,t} = 0.
    """
    if window < 2:
        raise ValueError(f"Window must be ≥ 2 for ddof=1 std.  Got {window}.")
    sv = volume.rolling(window=window, min_periods=window).std(ddof=1)
    sv.name = f"volume_std_{window}"
    return sv


def volume_zscore(
    volume: pd.Series,
    window: int = VOLUME_WINDOW,
) -> pd.Series:
    """
    Z_t^V = (V_t - V̄_t^(n)) / s_{V,t}^(n)

    Inputs:
        volume : pd.Series of V_t
        window : n

    Output:
        pd.Series of Z_t^V.  NaN where rolling std is 0 or undefined.

    Invariant:
        If V_t = V̄_t, then Z_t^V = 0.

    Edge case:
        If s_{V,t} = 0  (all volumes in window are identical),
        Z_t^V is undefined → NaN  (we do NOT divide by zero).

    Numerical test:
        V_t = V̄_t  →  Z_t^V = 0   (exactly, by definition)
    """
    mv = rolling_mean_volume(volume, window)
    sv = rolling_std_volume(volume, window)

    # Avoid division by zero: set to NaN where std = 0
    with np.errstate(invalid="ignore"):
        zv = (volume - mv) / sv.replace(0, float("nan"))

    zv.name = f"volume_zscore_{window}"
    return zv


def volume_spike(
    zscore: pd.Series,
    threshold: float = VOLUME_SPIKE_Z,
) -> pd.Series:
    """
    spike_t = True  iff  Z_t^V > threshold

    Inputs:
        zscore    : pd.Series of Z_t^V  (from volume_zscore)
        threshold : spike boundary (default from config: 2.0)

    Output:
        pd.Series of bool.  NaN rows → False.

    Note:
        This is a FEATURE — it describes an empirical observation.
        It is NOT a buy/sell signal.
    """
    spike = zscore > threshold
    spike.name = f"volume_spike_z{threshold}"
    return spike


def compute_all(
    volume: pd.Series,
    window: int = VOLUME_WINDOW,
    spike_threshold: float = VOLUME_SPIKE_Z,
) -> pd.DataFrame:
    """
    Compute all volume features and return as a single DataFrame.

    Columns produced:
        volume_change
        volume_pct_change
        volume_mean_{window}
        volume_std_{window}
        volume_zscore_{window}
        volume_spike_z{spike_threshold}
    """
    changes = volume_change(volume)
    mv   = rolling_mean_volume(volume, window)
    sv   = rolling_std_volume(volume, window)
    zv   = volume_zscore(volume, window)
    spk  = volume_spike(zv, spike_threshold)

    df = pd.concat([changes, mv, sv, zv, spk], axis=1)

    n_spikes = int(spk.sum())
    logger.info(
        f"Volume features computed: {len(df)} rows  window={window}  "
        f"spikes(Z>{spike_threshold})={n_spikes}"
    )
    return df
