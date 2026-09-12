# =============================================================================
# src/decision/hold_decision.py
#
# Mathematical specification
# ──────────────────────────
#
# Decision function:
#   Decision = f(E[Π_T], P(Π_T < 0), VaR, ES, es_fraction, trend, vol)
#
# Rules (explicit thresholds from config — no opaque AI):
#
#   HOLD:
#       E[Π_T] > 0  (expected return > EV_THRESHOLD_PCT)
#       AND P(Π_T < 0) < PROB_LOSS_MAX
#       AND es_fraction < ES_FRACTION_MAX
#
#   REDUCE:
#       E[Π_T] > 0
#       BUT es_fraction >= ES_FRACTION_MAX  (downside too large)
#
#   EXIT:
#       E[Π_T] ≤ 0  (negative expected value after costs)
#       OR P(Π_T < 0) >= PROB_LOSS_MAX  AND E[Π_T] barely positive
#
#   INSUFFICIENT_DATA:
#       Not enough historical data to fit a reliable distribution
#       (n_obs < minimum threshold)
#
# Anti-confirmation-bias rule:
#   The algorithm MUST be allowed to output EXIT even if the user
#   personally wants BERA to rise.
#   The objective is: maximise E[risk-adjusted future value]
#   NOT:              prove that BERA will go up.
# =============================================================================

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config.config import EV_THRESHOLD_PCT, PROB_LOSS_MAX, ES_FRACTION_MAX
from src.decision.expected_value import EVResult


# ---------------------------------------------------------------------------
# Decision enum
# ---------------------------------------------------------------------------

class Decision(str, Enum):
    HOLD             = "HOLD"
    REDUCE           = "REDUCE"
    EXIT             = "EXIT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


# ---------------------------------------------------------------------------
# Decision result
# ---------------------------------------------------------------------------

@dataclass
class DecisionResult:
    """
    Output of the decision engine for one horizon.

    Attributes:
        decision        : HOLD | REDUCE | EXIT | INSUFFICIENT_DATA
        horizon_label   : e.g., "7 days"
        ev_result       : the underlying EVResult
        reason          : plain-English explanation of the decision
        ev_threshold    : EV_THRESHOLD_PCT used
        prob_loss_max   : PROB_LOSS_MAX used
        es_fraction_max : ES_FRACTION_MAX used
    """
    decision        : Decision
    horizon_label   : str
    ev_result       : EVResult
    reason          : str
    ev_threshold    : float
    prob_loss_max   : float
    es_fraction_max : float


# ---------------------------------------------------------------------------
# Decision engine
# ---------------------------------------------------------------------------

def evaluate(
    ev: EVResult,
    ev_threshold_pct: float  = EV_THRESHOLD_PCT,
    prob_loss_max: float     = PROB_LOSS_MAX,
    es_fraction_max: float   = ES_FRACTION_MAX,
    min_obs: int             = 50,
    n_obs: int               = 9999,
) -> DecisionResult:
    """
    Apply the rule-based decision function to one EVResult.

    Rule logic (evaluated in order):

    1. INSUFFICIENT_DATA
       if n_obs < min_obs — distribution estimate is unreliable.

    2. EXIT
       if E[R_h] ≤ ev_threshold_pct  — expected value is non-positive.

    3. EXIT
       if P(loss) >= prob_loss_max AND E[R_h] ≤ 2 × ev_threshold_pct
       — barely positive EV with very high loss probability.

    4. REDUCE
       if E[R_h] > ev_threshold_pct
       AND es_fraction >= es_fraction_max
       — positive EV but tail risk too large; reduce position size.

    5. HOLD
       otherwise — positive EV and acceptable downside.

    Parameters:
        ev               : EVResult for this horizon
        ev_threshold_pct : minimum acceptable E[R_h] (fraction, e.g., 0.0)
        prob_loss_max    : maximum acceptable P(loss) for pure HOLD
        es_fraction_max  : maximum ES/position_value for HOLD
        min_obs          : minimum historical observations required
        n_obs            : actual observations in the fitted distribution
    """
    er = ev.expected_return
    pl = ev.prob_loss
    es = ev.es_fraction

    reasons = []

    # ------------------------------------------------------------------
    # Rule 1 — Data sufficiency
    # ------------------------------------------------------------------
    if n_obs < min_obs:
        return DecisionResult(
            decision=Decision.INSUFFICIENT_DATA,
            horizon_label=ev.horizon_label,
            ev_result=ev,
            reason=(
                f"Only {n_obs} historical observations available.  "
                f"Need ≥ {min_obs} for a reliable distribution estimate.  "
                "Collect more data before acting."
            ),
            ev_threshold=ev_threshold_pct,
            prob_loss_max=prob_loss_max,
            es_fraction_max=es_fraction_max,
        )

    # ------------------------------------------------------------------
    # Rule 2 — Non-positive expected value → EXIT
    # ------------------------------------------------------------------
    if er <= ev_threshold_pct:
        reasons.append(
            f"Expected return E[R] = {er:.4%} ≤ threshold {ev_threshold_pct:.4%}."
        )
        reasons.append(
            f"After transaction costs, this position has non-positive expected value."
        )
        return DecisionResult(
            decision=Decision.EXIT,
            horizon_label=ev.horizon_label,
            ev_result=ev,
            reason="  ".join(reasons),
            ev_threshold=ev_threshold_pct,
            prob_loss_max=prob_loss_max,
            es_fraction_max=es_fraction_max,
        )

    # ------------------------------------------------------------------
    # Rule 3 — Barely positive EV + very high loss probability → EXIT
    # ------------------------------------------------------------------
    if pl >= prob_loss_max and er <= 2 * abs(ev_threshold_pct) + 0.005:
        reasons.append(
            f"P(loss) = {pl:.1%} ≥ threshold {prob_loss_max:.1%}  "
            f"while expected return is only {er:.4%}."
        )
        reasons.append(
            "Risk/reward is unfavourable: high probability of loss "
            "with minimal positive expected value."
        )
        return DecisionResult(
            decision=Decision.EXIT,
            horizon_label=ev.horizon_label,
            ev_result=ev,
            reason="  ".join(reasons),
            ev_threshold=ev_threshold_pct,
            prob_loss_max=prob_loss_max,
            es_fraction_max=es_fraction_max,
        )

    # ------------------------------------------------------------------
    # Rule 4 — Positive EV but tail risk too large → REDUCE
    # ------------------------------------------------------------------
    if es >= es_fraction_max:
        reasons.append(
            f"Expected return E[R] = {er:.4%} is positive."
        )
        reasons.append(
            f"However, ES₉₅ / position = {es:.1%} ≥ threshold {es_fraction_max:.1%}."
        )
        reasons.append(
            "Tail risk exceeds tolerance.  "
            "Consider reducing position size to lower absolute downside exposure."
        )
        return DecisionResult(
            decision=Decision.REDUCE,
            horizon_label=ev.horizon_label,
            ev_result=ev,
            reason="  ".join(reasons),
            ev_threshold=ev_threshold_pct,
            prob_loss_max=prob_loss_max,
            es_fraction_max=es_fraction_max,
        )

    # ------------------------------------------------------------------
    # Rule 5 — HOLD
    # ------------------------------------------------------------------
    reasons.append(
        f"Expected return E[R] = {er:.4%} > threshold {ev_threshold_pct:.4%}."
    )
    reasons.append(
        f"P(profit) = {ev.prob_profit:.1%}.  "
        f"ES₉₅ / position = {es:.1%} < threshold {es_fraction_max:.1%}."
    )
    reasons.append(
        "Expected value is positive after costs and downside risk is within tolerance."
    )

    return DecisionResult(
        decision=Decision.HOLD,
        horizon_label=ev.horizon_label,
        ev_result=ev,
        reason="  ".join(reasons),
        ev_threshold=ev_threshold_pct,
        prob_loss_max=prob_loss_max,
        es_fraction_max=es_fraction_max,
    )


def evaluate_all(
    ev_results: list[EVResult],
    ev_threshold_pct: float = EV_THRESHOLD_PCT,
    prob_loss_max: float    = PROB_LOSS_MAX,
    es_fraction_max: float  = ES_FRACTION_MAX,
    n_obs: int              = 9999,
) -> list[DecisionResult]:
    """
    Apply evaluate() across all horizon EVResults.
    """
    return [
        evaluate(ev, ev_threshold_pct, prob_loss_max, es_fraction_max, n_obs=n_obs)
        for ev in ev_results
    ]


def consensus_decision(decisions: list[DecisionResult]) -> Decision:
    """
    Summarise across all horizons into one top-level decision.

    Rule:
        If majority of horizons are EXIT  → EXIT
        If any horizon is EXIT            → REDUCE  (caution)
        If majority are REDUCE            → REDUCE
        Otherwise                         → HOLD

    This is a simple voting rule for V0.1.
    A more sophisticated aggregation can be added in V1.
    """
    counts = {d: 0 for d in Decision}
    for dr in decisions:
        counts[dr.decision] += 1

    n = len(decisions)
    if counts[Decision.EXIT] > n / 2:
        return Decision.EXIT
    if counts[Decision.EXIT] > 0:
        return Decision.REDUCE
    if counts[Decision.REDUCE] > n / 2:
        return Decision.REDUCE
    return Decision.HOLD
