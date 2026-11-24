"""
Multi-horizon target definitions — Spec §4.

For every horizon H ∈ {1, 5, 15, 60}:
  • Causal 60-min realized vol  σ̂_{t,60}
  • Raw log-delta               y_{t,H}
  • Vol-normalized delta        ỹ_{t,H}
  • Cascade magnitude           m_{t,H}
  • Binary cascade flag         is_cascade_H
"""

from __future__ import annotations

import numpy as np
import polars as pl

from config import (
    HORIZONS_MIN,
    SEED,
    VOLATILITY_FLOOR,
    TARGET_CLIP_LOWER,
    TARGET_CLIP_UPPER,
    CASCADE_THRESHOLD_MULTIPLIER,
    LOG_MAG_FLOOR,
)

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------
VOL_FLOOR: float = VOLATILITY_FLOOR
TARGET_CLIP: float = TARGET_CLIP_UPPER
CASCADE_THRESHOLD_SIGMA: float = CASCADE_THRESHOLD_MULTIPLIER
MAG_EPSILON: float = LOG_MAG_FLOOR


# =========================================================================
#  Realized volatility (causal, trailing 60 steps)
# =========================================================================

def _causal_realized_vol_60(
    r: np.ndarray,
    steps_in_segment: np.ndarray,
) -> np.ndarray:
    """Rolling std of r_t over trailing 60 valid steps, floored at VOL_FLOOR.

    Parameters
    ----------
    r : np.ndarray  (N,)
        1-min log-returns (NaN where invalid).
    steps_in_segment : np.ndarray (N,)
        1-based counter within each valid segment; 0 inside gaps.

    Returns
    -------
    np.ndarray (N,)
        σ̂_{t,60} — NaN where steps_in_segment < 60.
    """
    n = len(r)
    vol = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        if steps_in_segment[i] < 60:
            continue
        window = r[i - 59 : i + 1]  # trailing 60 (inclusive)
        if np.any(np.isnan(window)):
            continue
        s = np.std(window, ddof=1)
        vol[i] = max(s, VOL_FLOOR)

    return vol


def compute_realized_vol_60(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``sigma_60`` column to *df*.

    *df* must contain ``r_1`` (1-min log-return) and ``steps_in_segment``.
    """
    r = np.asarray(df["r_1"], dtype=np.float64)
    sis = np.asarray(df["steps_in_segment"], dtype=np.int64)
    vol = _causal_realized_vol_60(r, sis)
    return df.with_columns(pl.Series("sigma_60", vol))


# =========================================================================
#  Raw log-deltas
# =========================================================================

def compute_raw_log_deltas(
    df: pl.DataFrame,
    horizons: list[int] | None = None,
) -> pl.DataFrame:
    """Add ``y_H`` columns for each horizon.

    y_{t,H} = ln(x_close[t+H] / x_close[t]),
    NULL if segment_id[t+H] != segment_id[t].
    """
    horizons = horizons or HORIZONS_MIN
    x_close = np.asarray(df["x_close"], dtype=np.float64)
    seg_id  = np.asarray(df["segment_id"])
    n = len(x_close)

    new_cols: dict[str, np.ndarray] = {}
    for h in horizons:
        y = np.full(n, np.nan, dtype=np.float64)
        for t in range(n - h):
            if seg_id[t + h] != seg_id[t]:
                continue
            if seg_id[t] < 0:
                continue
            if x_close[t] <= 0 or x_close[t + h] <= 0:
                continue
            if np.isnan(x_close[t]) or np.isnan(x_close[t + h]):
                continue
            y[t] = np.log(x_close[t + h] / x_close[t])
        new_cols[f"y_{h}"] = y

    return df.with_columns([pl.Series(k, v) for k, v in new_cols.items()])


# =========================================================================
#  Vol-normalized deltas (primary training target)
# =========================================================================

def compute_vol_normalized_deltas(
    df: pl.DataFrame,
    horizons: list[int] | None = None,
    clip_train: bool = False,
    train_mask: np.ndarray | None = None,
) -> pl.DataFrame:
    """Add ``y_norm_H`` columns.

    ỹ_{t,H} = y_{t,H} / (σ̂_{t,60} · √H)

    Parameters
    ----------
    clip_train : bool
        If True, clip to [-15, 15] inside training rows only.
    train_mask : np.ndarray, optional
        Boolean array marking training rows.  Required if clip_train=True.
    """
    horizons = horizons or HORIZONS_MIN
    sigma = np.asarray(df["sigma_60"], dtype=np.float64)
    n = len(sigma)

    new_cols: dict[str, np.ndarray] = {}
    for h in horizons:
        y = np.asarray(df[f"y_{h}"], dtype=np.float64)
        y_norm = y / (sigma * np.sqrt(h))

        if clip_train and train_mask is not None:
            clipped = np.clip(y_norm, -TARGET_CLIP, TARGET_CLIP)
            y_norm = np.where(train_mask, clipped, y_norm)

        new_cols[f"y_norm_{h}"] = y_norm

    return df.with_columns([pl.Series(k, v) for k, v in new_cols.items()])


# =========================================================================
#  Cascade / realized magnitude target  (auxiliary)
# =========================================================================

def compute_cascade_targets(
    df: pl.DataFrame,
    horizons: list[int] | None = None,
) -> pl.DataFrame:
    """Add ``m_H`` and ``is_cascade_H`` columns for each horizon.

    m_{t,H}  = ln(sqrt(Σ r²_{t+1..t+H}) + ε)
    is_cascade_H = 1 if max extreme deviation > 4·σ̂·√H
    """
    horizons = horizons or HORIZONS_MIN
    r       = np.asarray(df["r_1"], dtype=np.float64)
    x_close = np.asarray(df["x_close"], dtype=np.float64)
    x_high  = np.asarray(df["x_high"], dtype=np.float64)
    x_low   = np.asarray(df["x_low"], dtype=np.float64)
    sigma   = np.asarray(df["sigma_60"], dtype=np.float64)
    seg_id  = np.asarray(df["segment_id"])
    n = len(r)

    new_cols: dict[str, np.ndarray] = {}

    for h in horizons:
        mag      = np.full(n, np.nan, dtype=np.float64)
        cascade  = np.full(n, np.nan, dtype=np.float64)

        for t in range(n - h):
            if seg_id[t] < 0:
                continue
            # Check segment continuity for all t+1..t+H
            if seg_id[t + h] != seg_id[t]:
                continue

            # Verify full continuity within window
            valid = True
            for k in range(1, h + 1):
                if seg_id[t + k] != seg_id[t]:
                    valid = False
                    break
            if not valid:
                continue

            if np.isnan(sigma[t]) or sigma[t] <= 0:
                continue

            # Realized magnitude: ln(sqrt(Σ r²_{t+1..t+H}) + ε)
            sum_r2 = 0.0
            for k in range(1, h + 1):
                rk = r[t + k]
                if np.isnan(rk):
                    valid = False
                    break
                sum_r2 += rk * rk
            if not valid:
                continue
            mag[t] = np.log(np.sqrt(sum_r2) + MAG_EPSILON)

            # Binary cascade: max extreme deviation > 4·σ̂·√H
            threshold = CASCADE_THRESHOLD_SIGMA * sigma[t] * np.sqrt(h)
            max_dev = 0.0
            base_price = x_close[t]
            if base_price <= 0 or np.isnan(base_price):
                cascade[t] = 0.0
                continue
            for k in range(1, h + 1):
                hi = x_high[t + k]
                lo = x_low[t + k]
                if not np.isnan(hi):
                    dev_hi = abs(hi / base_price - 1.0)
                    if dev_hi > max_dev:
                        max_dev = dev_hi
                if not np.isnan(lo):
                    dev_lo = abs(lo / base_price - 1.0)
                    if dev_lo > max_dev:
                        max_dev = dev_lo

            cascade[t] = 1.0 if max_dev > threshold else 0.0

        new_cols[f"m_{h}"] = mag
        new_cols[f"is_cascade_{h}"] = cascade

    return df.with_columns([pl.Series(k, v) for k, v in new_cols.items()])


# =========================================================================
#  Convenience: compute all targets at once
# =========================================================================

def compute_all_targets(
    df: pl.DataFrame,
    horizons: list[int] | None = None,
) -> pl.DataFrame:
    """Compute 1-min log-returns, σ̂_{t,60}, and all multi-horizon targets.

    *df* must contain at minimum:
    ``x_open``, ``x_high``, ``x_low``, ``x_close``,
    ``segment_id``, ``steps_in_segment``.

    Returns the input dataframe augmented with:
    ``r_1``, ``sigma_60``,
    ``y_H``, ``y_norm_H``, ``m_H``, ``is_cascade_H`` for each H.
    """
    horizons = horizons or HORIZONS_MIN

    # 1-min log-return (null if steps_in_segment <= 1)
    x_close = np.asarray(df["x_close"], dtype=np.float64)
    sis = np.asarray(df["steps_in_segment"], dtype=np.int64)
    n = len(x_close)
    r_1 = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        if sis[i] <= 1:
            continue
        if x_close[i] > 0 and x_close[i - 1] > 0:
            r_1[i] = np.log(x_close[i] / x_close[i - 1])
    df = df.with_columns(pl.Series("r_1", r_1))

    # Realized vol
    df = compute_realized_vol_60(df)

    # Raw log-deltas
    df = compute_raw_log_deltas(df, horizons)

    # Vol-normalized deltas (no train clipping here — that's per-fold)
    df = compute_vol_normalized_deltas(df, horizons)

    # Cascade targets
    df = compute_cascade_targets(df, horizons)

    return df
