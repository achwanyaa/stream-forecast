"""
Evaluation metrics — Spec §4.

• Non-crossing quantile repair
• Pinball (quantile) loss
• Skill score vs. naive baseline
• Calibration coverage (cov_10, cov_80, cov_90)
• Conditional directional accuracy
• Residual autopsy payload
"""

from __future__ import annotations

from typing import Any

import numpy as np

from config import QUANTILES


# =========================================================================
#  Non-crossing repair
# =========================================================================

def repair_quantile_crossings(
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort (q10, q50, q90) ascending per row to enforce monotonicity.

    Parameters
    ----------
    q10, q50, q90 : np.ndarray  shape (N,)

    Returns
    -------
    tuple of np.ndarray
        (q10_repaired, q50_repaired, q90_repaired)
    """
    stacked = np.stack([q10, q50, q90], axis=-1)  # (N, 3)
    stacked.sort(axis=-1)
    return stacked[:, 0], stacked[:, 1], stacked[:, 2]


# =========================================================================
#  Pinball (quantile) loss
# =========================================================================

def pinball_loss(
    y: np.ndarray,
    q_hat: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Element-wise pinball loss L_α(y, q̂).

    L_α(y, q̂) = max(α·(y − q̂), (α − 1)·(y − q̂))
    """
    diff = y - q_hat
    return np.maximum(alpha * diff, (alpha - 1.0) * diff)


def pinball_loss_mean(
    y: np.ndarray,
    q_hat: np.ndarray,
    alpha: float,
) -> float:
    """Scalar mean pinball loss (ignoring NaN)."""
    losses = pinball_loss(y, q_hat, alpha)
    return float(np.nanmean(losses))


def compute_pinball_metrics(
    y: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
) -> dict[str, float]:
    """Compute pinball_10, pinball_50, pinball_90, pinball_mean.

    *y* is the actual (unclipped) raw log-delta.
    Quantiles are assumed **already repaired** (non-crossing).
    """
    p10 = pinball_loss_mean(y, q10, 0.10)
    p50 = pinball_loss_mean(y, q50, 0.50)
    p90 = pinball_loss_mean(y, q90, 0.90)
    p_mean = (p10 + p50 + p90) / 3.0
    return {
        "pinball_10": p10,
        "pinball_50": p50,
        "pinball_90": p90,
        "pinball_mean": p_mean,
    }


# =========================================================================
#  Skill score vs. naive
# =========================================================================

def skill_score(
    pinball_mean_model: float,
    pinball_mean_naive: float,
) -> float:
    """Skill = 1 − pinball_mean_model / pinball_mean_naive.

    > 0 means improvement over naive.
    """
    if pinball_mean_naive == 0.0:
        return 0.0
    return 1.0 - pinball_mean_model / pinball_mean_naive


# =========================================================================
#  Calibration coverage
# =========================================================================

def compute_calibration(
    y: np.ndarray,
    q10: np.ndarray,
    q90: np.ndarray,
) -> dict[str, float]:
    """Coverage fractions.

    cov_10 = P(y < q10)   ideal 0.10
    cov_80 = P(q10 ≤ y ≤ q90)  ideal 0.80
    cov_90 = P(y < q90)   ideal 0.90
    """
    valid = ~(np.isnan(y) | np.isnan(q10) | np.isnan(q90))
    yv, q10v, q90v = y[valid], q10[valid], q90[valid]
    n = len(yv)
    if n == 0:
        return {"cov_10": np.nan, "cov_80": np.nan, "cov_90": np.nan}

    cov_10 = float(np.mean(yv < q10v))
    cov_90 = float(np.mean(yv < q90v))
    cov_80 = float(np.mean((yv >= q10v) & (yv <= q90v)))
    return {"cov_10": cov_10, "cov_80": cov_80, "cov_90": cov_90}


# =========================================================================
#  Conditional directional accuracy
# =========================================================================

def cond_dir_acc_conf10(
    y: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    train_q10: np.ndarray,
    train_q50: np.ndarray,
    train_q90: np.ndarray,
) -> float:
    """Sign accuracy on rows where signal-to-width ratio exceeds
    training 90th percentile.

    signal_to_width = |q50| / (q90 - q10 + 1e-12)
    """
    # Training threshold
    train_width = train_q90 - train_q10 + 1e-12
    train_ratio = np.abs(train_q50) / train_width
    threshold = float(np.nanpercentile(train_ratio, 90))

    # Test
    test_width = q90 - q10 + 1e-12
    test_ratio = np.abs(q50) / test_width

    mask = (test_ratio > threshold) & ~np.isnan(y) & ~np.isnan(q50)
    if mask.sum() == 0:
        return np.nan

    correct = np.sign(q50[mask]) == np.sign(y[mask])
    return float(np.mean(correct))


def cond_dir_acc_tension10(
    y: np.ndarray,
    q50: np.ndarray,
    tension_test: np.ndarray,
    tension_train: np.ndarray,
) -> float:
    """Sign accuracy on rows where crowding tension exceeds training
    90th percentile.

    tension = abs(crowding_penalty_z) or abs(skew_top20_z)
    """
    threshold = float(np.nanpercentile(tension_train, 90))

    mask = (tension_test > threshold) & ~np.isnan(y) & ~np.isnan(q50)
    if mask.sum() == 0:
        return np.nan

    correct = np.sign(q50[mask]) == np.sign(y[mask])
    return float(np.mean(correct))


# =========================================================================
#  Residual autopsy
# =========================================================================

def _autocorrelation(x: np.ndarray, max_lag: int = 10) -> list[float]:
    """Lag 1..max_lag sample autocorrelation of *x* (NaN-aware)."""
    x = x.copy()
    valid = ~np.isnan(x)
    if valid.sum() < max_lag + 2:
        return [np.nan] * max_lag
    mean = np.nanmean(x)
    x[~valid] = mean  # fill for lagged product — conservative
    var = np.nanvar(x[valid])
    if var == 0:
        return [0.0] * max_lag
    acf = []
    for lag in range(1, max_lag + 1):
        cov = np.nanmean((x[lag:] - mean) * (x[:-lag] - mean))
        acf.append(float(cov / var))
    return acf


def residual_autopsy(
    y: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    utc_hour: np.ndarray,
    regime: np.ndarray,
) -> dict[str, Any]:
    """Residual autopsy payload (Spec §4).

    Parameters
    ----------
    y       : actual unclipped log-delta
    q10, q50, q90 : predicted quantiles (repaired)
    utc_hour : int array 0..23
    regime   : int array 0/1/2 (dominant regime label)

    Returns
    -------
    dict  with keys:
        residual_acf   – list[float] lag 1..10
        pinball_by_hour – dict[int, dict]  hour → {pinball_10, pinball_50, pinball_90, pinball_mean}
        by_regime       – dict[int, dict]  regime → {pinball_10..90, pinball_mean, cov_10, cov_80, cov_90}
    """
    residuals = y - q50

    # ACF of residuals
    acf = _autocorrelation(residuals, max_lag=10)

    # Pinball by UTC hour
    pinball_by_hour: dict[int, dict[str, float]] = {}
    for h in range(24):
        mask = utc_hour == h
        if mask.sum() == 0:
            continue
        pinball_by_hour[h] = compute_pinball_metrics(
            y[mask], q10[mask], q50[mask], q90[mask]
        )

    # Pinball + calibration by regime
    by_regime: dict[int, dict[str, float]] = {}
    for r in [0, 1, 2]:
        mask = regime == r
        if mask.sum() == 0:
            continue
        pbl = compute_pinball_metrics(y[mask], q10[mask], q50[mask], q90[mask])
        cal = compute_calibration(y[mask], q10[mask], q90[mask])
        by_regime[r] = {**pbl, **cal}

    return {
        "residual_acf": acf,
        "pinball_by_hour": pinball_by_hour,
        "by_regime": by_regime,
    }


# =========================================================================
#  Aggregate fold-level metrics
# =========================================================================

def compute_fold_metrics(
    y: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    naive_q10: np.ndarray | None = None,
    naive_q50: np.ndarray | None = None,
    naive_q90: np.ndarray | None = None,
    utc_hour: np.ndarray | None = None,
    regime: np.ndarray | None = None,
    train_q10: np.ndarray | None = None,
    train_q50: np.ndarray | None = None,
    train_q90: np.ndarray | None = None,
    tension_test: np.ndarray | None = None,
    tension_train: np.ndarray | None = None,
) -> dict[str, Any]:
    """Full metric dict for a single fold.

    Returns all spec §4 metrics: pinball, skill, calibration,
    conditional directional accuracy, and residual autopsy (if
    utc_hour / regime are provided).
    """
    # Repair crossings
    q10, q50, q90 = repair_quantile_crossings(q10, q50, q90)

    # Pinball
    pbl = compute_pinball_metrics(y, q10, q50, q90)

    # Skill vs. naive
    skill = None
    if naive_q10 is not None and naive_q50 is not None and naive_q90 is not None:
        nq10, nq50, nq90 = repair_quantile_crossings(naive_q10, naive_q50, naive_q90)
        naive_pbl = compute_pinball_metrics(y, nq10, nq50, nq90)
        skill = skill_score(pbl["pinball_mean"], naive_pbl["pinball_mean"])

    # Calibration
    cal = compute_calibration(y, q10, q90)

    # Conditional dir acc
    cda_conf = None
    if train_q10 is not None and train_q50 is not None and train_q90 is not None:
        cda_conf = cond_dir_acc_conf10(y, q10, q50, q90, train_q10, train_q50, train_q90)

    cda_tension = None
    if tension_test is not None and tension_train is not None:
        cda_tension = cond_dir_acc_tension10(y, q50, tension_test, tension_train)

    # Autopsy
    autopsy = None
    if utc_hour is not None and regime is not None:
        autopsy = residual_autopsy(y, q10, q50, q90, utc_hour, regime)

    result: dict[str, Any] = {
        **pbl,
        "skill_vs_naive": skill,
        **cal,
        "cond_dir_acc_conf10": cda_conf,
        "cond_dir_acc_tension10": cda_tension,
    }
    if autopsy is not None:
        result["residual_acf1"] = autopsy["residual_acf"][0] if autopsy["residual_acf"] else None
        result["autopsy"] = autopsy

    return result
