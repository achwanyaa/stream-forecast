"""
Layer 2 — Causal Behavioral Feature Engine
===========================================
Spec §0, §6: Zero lookahead leakage, scale-invariant, gap-masked feature pipeline.

All features are computed in Polars with Numba @njit kernels for Permutation
Entropy, streak computation, steps since rolling high/low, and rolling percentile rank.
Every window W outputs null whenever steps_in_segment < W.

Feature Groups:
  1. Kinematics (multi-scale returns, acceleration)
  2. Intrabar Structure (Garman-Klass, intrabar close pos, shadow ratios)
  3. Tension & Stretch (rolling Z-scores, percentile ranks, distance from extrema)
  4. Streaks & Fatigue (signed run length, steps since extrema, positive return frac)
  5. Aggression & Intensity (1m imbalance, volume-weighted imbalance, log-intensity Z)
  6. Tier 2 Crowd State (commitment velocity, tension product, skew Z, crowding Z)
  7. Loudness vs. Disorder (realized vol, GK vol, vol ratios, vol-of-vol, Permutation Entropy)
  8. Cyclical Time (diurnal & weekly sine / cosine encodings)

Public API:
  compute_all_features(master_df: pl.DataFrame) -> pl.DataFrame
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import polars as pl
from numba import njit

# ---------------------------------------------------------------------------
# Config & Constants (§4, §6, §7)
# ---------------------------------------------------------------------------
try:
    from compute.config import (
        EPSILON,
        VOLATILITY_FLOOR,
        HMM_OBS_FEATURES,
    )
except ImportError:
    EPSILON = 1e-12
    VOLATILITY_FLOOR = 1e-6
    HMM_OBS_FEATURES = [
        "log_rv_60",
        "vol_ratio_15_240",
        "perm_entropy_60",
        "abs_agg_imbalance_15",
    ]

_EPS = EPSILON
_VOL_FLOOR = VOLATILITY_FLOOR

# Window specifications
_RET_WINDOWS: list[int] = [1, 5, 15, 60, 240]
_TENSION_WINDOWS: list[int] = [60, 240, 1440]
_VOL_WINDOWS: list[int] = [15, 60, 240, 1440]
_COMMITMENT_WINDOWS: list[int] = [15, 60, 240]

# Permutation Entropy constants (m=4, tau=1)
_PE_M: int = 4
_PE_TAU: int = 1
_PE_WINDOWS: list[int] = [60, 240]
_PE_NORM: float = math.log(24.0)  # ln(4!) = ln(24)


# ===================================================================
# SECTION A — High-Performance Numba @njit Kernels
# ===================================================================

@njit(cache=True)
def _streak_kernel(signs: np.ndarray, seg_ids: np.ndarray, steps_in_seg: np.ndarray) -> np.ndarray:
    """Compute signed consecutive directional run length.

    Parameters
    ----------
    signs : int8 array
        +1 for up return, -1 for down return, 0 for flat.
    seg_ids : int32 array
        Segment id per row (-1 for gap rows).
    steps_in_seg : int64 array
        1-based step counter within segment (0 for gap rows).

    Returns
    -------
    streak : float64 array
        Positive for up-streaks, negative for down-streaks, 0 for flat.
        NaN where steps_in_seg < 2 or seg_ids == -1.
    """
    n = signs.shape[0]
    streak = np.full(n, np.nan, dtype=np.float64)
    prev_sign = np.int8(0)
    run = 0.0
    prev_seg = np.int32(-999)

    for i in range(n):
        sid = seg_ids[i]
        sis = steps_in_seg[i]

        if sid == -1 or sis < 2:
            prev_sign = np.int8(0)
            run = 0.0
            prev_seg = sid
            continue

        # Reset streak on segment boundary
        if sid != prev_seg:
            prev_sign = np.int8(0)
            run = 0.0

        s = signs[i]
        if s == 0:
            run = 0.0
            prev_sign = np.int8(0)
            streak[i] = 0.0
        elif s == prev_sign:
            run += 1.0
            streak[i] = run * float(s)
        else:
            run = 1.0
            prev_sign = s
            streak[i] = float(s)

        prev_seg = sid

    return streak


@njit(cache=True)
def _steps_since_hilo_kernel(
    close: np.ndarray,
    seg_ids: np.ndarray,
    steps_in_seg: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal steps since rolling high and rolling low over trailing window.

    Scans backwards up to `window` steps within the same segment.
    Returns (hi_since, lo_since) as float64 arrays.
    NaN where steps_in_seg < window or seg_ids == -1.
    """
    n = close.shape[0]
    hi_since = np.full(n, np.nan, dtype=np.float64)
    lo_since = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        if seg_ids[i] == -1 or steps_in_seg[i] < window:
            continue

        hi_val = close[i]
        lo_val = close[i]
        hi_idx = i
        lo_idx = i

        # Scan backwards up to window steps
        for offset in range(1, window):
            j = i - offset
            if close[j] > hi_val:
                hi_val = close[j]
                hi_idx = j
            if close[j] < lo_val:
                lo_val = close[j]
                lo_idx = j

        hi_since[i] = float(i - hi_idx)
        lo_since[i] = float(i - lo_idx)

    return hi_since, lo_since


@njit(cache=True)
def _rolling_percentile_rank_kernel(
    vals: np.ndarray,
    seg_ids: np.ndarray,
    steps_in_seg: np.ndarray,
    window: int,
) -> np.ndarray:
    """Causal rolling empirical percentile rank of vals over trailing window.

    Fraction of points in [t - window + 1, t] that are <= vals[t].
    NaN where steps_in_seg < window or seg_ids == -1.
    """
    n = vals.shape[0]
    ranks = np.full(n, np.nan, dtype=np.float64)
    inv_w = 1.0 / float(window)

    for i in range(n):
        if seg_ids[i] == -1 or steps_in_seg[i] < window:
            continue

        curr = vals[i]
        count = 0
        for offset in range(window):
            if vals[i - offset] <= curr:
                count += 1

        ranks[i] = float(count) * inv_w

    return ranks


@njit(cache=True)
def _perm_entropy_kernel(
    vals: np.ndarray,
    seg_ids: np.ndarray,
    steps_in_seg: np.ndarray,
    window: int,
    m: int,
    tau: int,
) -> np.ndarray:
    """O(N) causal rolling Permutation Entropy using incremental factorial number system.

    Parameters
    ----------
    vals : float64 array (e.g. 1-minute log-returns)
    seg_ids : int32 array (-1 for gaps)
    steps_in_seg : int64 array
    window : int (rolling window size: 60 or 240)
    m : int (embedding dimension: 4)
    tau : int (embedding delay: 1)

    Returns
    -------
    pe : float64 array (normalized to [0, 1]), NaN where steps_in_seg < window.
    """
    n = vals.shape[0]
    pe = np.full(n, np.nan, dtype=np.float64)

    # Number of possible permutations = m!
    n_perms = 1
    for k in range(1, m + 1):
        n_perms *= k

    # Pre-compute factorials for Lehmer code
    factorials = np.empty(m, dtype=np.int64)
    factorials[0] = 1
    for k in range(1, m):
        factorials[k] = factorials[k - 1] * k

    ln_norm = math.log(float(n_perms))  # ln(m!)
    pat_span = (m - 1) * tau + 1        # steps needed for one pattern
    max_pats = window - pat_span + 1    # max patterns in sliding window

    if max_pats <= 0:
        return pe

    hist = np.zeros(n_perms, dtype=np.int64)
    ring_buf = np.zeros(max_pats, dtype=np.int64)
    buf_pos = 0
    pat_count = 0
    prev_seg = np.int32(-999)

    pv = np.empty(m, dtype=np.float64)

    for i in range(n):
        sid = seg_ids[i]
        sis = steps_in_seg[i]

        if sid == -1 or sid != prev_seg:
            # Segment boundary or gap — reset sliding window state
            hist[:] = 0
            ring_buf[:] = 0
            buf_pos = 0
            pat_count = 0
            prev_seg = sid
            if sid == -1:
                continue

        # Need enough points within the segment to extract length-m pattern
        if sis < pat_span:
            continue

        # Extract pattern values
        for j in range(m):
            pv[j] = vals[i - (m - 1 - j) * tau]

        # Compute Lehmer code (rank in 0..m!-1)
        code = np.int64(0)
        for j in range(m):
            c = np.int64(0)
            for k in range(j + 1, m):
                if pv[k] < pv[j]:
                    c += 1
            code += c * factorials[m - 1 - j]

        # Slide window: remove pattern that drops out of the trailing window
        if pat_count == max_pats:
            old_code = ring_buf[buf_pos]
            hist[old_code] -= 1
            pat_count -= 1

        # Add new pattern
        ring_buf[buf_pos] = code
        hist[code] += 1
        pat_count += 1
        buf_pos = (buf_pos + 1) % max_pats

        # Only emit feature value when the full trailing window is inside the segment
        if sis < window or pat_count < max_pats:
            continue

        # Shannon entropy
        tot = float(pat_count)
        h = 0.0
        for p_idx in range(n_perms):
            cnt = hist[p_idx]
            if cnt > 0:
                p = float(cnt) / tot
                h -= p * math.log(p)

        pe[i] = h / ln_norm

    return pe


# ===================================================================
# SECTION B — Polars Expression Helpers
# ===================================================================

def _mask_gap(expr: pl.Expr, window: int) -> pl.Expr:
    """Mask expression to null where steps_in_segment < window or is_gap is true."""
    return (
        pl.when((pl.col("steps_in_segment") >= window) & (~pl.col("is_gap")))
        .then(expr)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def _rolling_zscore(col_name: str, window: int, out_name: str) -> pl.Expr:
    """Causal rolling Z-score over trailing window."""
    mean_e = pl.col(col_name).rolling_mean(window_size=window, min_periods=window)
    std_e = pl.col(col_name).rolling_std(window_size=window, min_periods=window)
    raw_z = (pl.col(col_name) - mean_e) / (std_e + _EPS)
    return _mask_gap(raw_z, window).alias(out_name)


# ===================================================================
# SECTION C — Feature Computation Groups
# ===================================================================

def _compute_base_series(df: pl.DataFrame) -> pl.DataFrame:
    """Compute base returns, log prices, and baseline causal volatility σ̂_{t,60}."""
    # 1-minute log-return
    r1_raw = (pl.col("x_close") / pl.col("x_close").shift(1)).log()
    r1 = (
        pl.when((pl.col("steps_in_segment") >= 2) & (~pl.col("is_gap")))
        .then(r1_raw)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )

    df = df.with_columns([
        r1.alias("r_1"),
        r1.alias("r_1m"),
        (r1 ** 2).alias("_r1_sq"),
        pl.col("x_close").log().alias("_ln_close"),
    ])

    # Multi-scale raw returns (r_{t,w} = ln(close_t / close_{t-w}))
    for w in [5, 15, 60, 240]:
        rw_raw = (pl.col("x_close") / pl.col("x_close").shift(w)).log()
        df = df.with_columns(
            pl.when((pl.col("steps_in_segment") >= w + 1) & (~pl.col("is_gap")))
            .then(rw_raw)
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias(f"r_{w}m")
        )

    # Causal 60m realized volatility σ̂_{t,60} = sqrt(rolling_mean(r1^2, 60)), floored at VOLATILITY_FLOOR
    sigma_60_raw = (
        pl.col("_r1_sq")
        .rolling_mean(window_size=60, min_periods=60)
        .sqrt()
        .clip(lower_bound=_VOL_FLOOR)
    )
    df = df.with_columns(
        _mask_gap(sigma_60_raw, 60).alias("sigma_60")
    )

    return df


def _group1_kinematics(df: pl.DataFrame) -> pl.DataFrame:
    """Group 1: Kinematics.

    - Multi-scale vol-normalized log-returns:
        r_{t,w} / (σ̂_{t,60} · √w) for w ∈ {1, 5, 15, 60, 240}
    - Acceleration:
        (r_{t,5} - r_{t-5,5}) / (σ̂_{t,60} · √5)
        (r_{t,15} - r_{t-15,15}) / (σ̂_{t,60} · √15)
    """
    exprs: list[pl.Expr] = []

    for w in _RET_WINDOWS:
        r_col = f"r_{w}m"
        scale = pl.col("sigma_60") * math.sqrt(w) + _EPS
        normed = pl.col(r_col) / scale
        min_steps = max(w + 1, 60)
        exprs.append(_mask_gap(normed, min_steps).alias(f"ret_{w}m"))

    # Acceleration: r_{t,5} - r_{t-5,5}
    scale_5 = pl.col("sigma_60") * math.sqrt(5) + _EPS
    accel_5_raw = (pl.col("r_5m") - pl.col("r_5m").shift(5)) / scale_5
    exprs.append(_mask_gap(accel_5_raw, 60).alias("accel_5m"))

    # Acceleration: r_{t,15} - r_{t-15,15}
    scale_15 = pl.col("sigma_60") * math.sqrt(15) + _EPS
    accel_15_raw = (pl.col("r_15m") - pl.col("r_15m").shift(15)) / scale_15
    exprs.append(_mask_gap(accel_15_raw, 60).alias("accel_15m"))

    return df.with_columns(exprs)


def _group2_intrabar(df: pl.DataFrame) -> pl.DataFrame:
    """Group 2: Intrabar Structure.

    - Garman-Klass 1m variance:
        v_GK = max(0, 0.5·(ln(H/L))² - (2·ln2 - 1)·(ln(C/O))²)
    - Intrabar close position:
        (C - L) / (H - L + 1e-12)
    - Upper shadow ratio:
        (H - max(C, O)) / (H - L + 1e-12)
    - Lower shadow ratio:
        (min(C, O) - L) / (H - L + 1e-12)
    """
    hl_log = (pl.col("x_high") / pl.col("x_low")).log()
    co_log = (pl.col("x_close") / pl.col("x_open")).log()
    c2 = 2.0 * math.log(2.0) - 1.0

    gk_raw = (0.5 * (hl_log ** 2) - c2 * (co_log ** 2)).clip(lower_bound=0.0)
    gk_var = _mask_gap(gk_raw, 1).alias("gk_var")

    bar_range = pl.col("x_high") - pl.col("x_low") + _EPS
    intrabar_pos = _mask_gap((pl.col("x_close") - pl.col("x_low")) / bar_range, 1).alias("intrabar_pos")

    c_max_o = pl.max_horizontal("x_close", "x_open")
    c_min_o = pl.min_horizontal("x_close", "x_open")
    upper_shadow = _mask_gap((pl.col("x_high") - c_max_o) / bar_range, 1).alias("upper_shadow_ratio")
    lower_shadow = _mask_gap((c_min_o - pl.col("x_low")) / bar_range, 1).alias("lower_shadow_ratio")

    return df.with_columns([gk_var, intrabar_pos, upper_shadow, lower_shadow])


def _group3_tension(df: pl.DataFrame) -> pl.DataFrame:
    """Group 3: Tension & Stretch.

    - Rolling Z-scores of ln(x_close) over w ∈ {60, 240, 1440}
    - Rolling percentile rank of x_close over w ∈ {60, 240, 1440}
    - Distance from rolling max/min over 240m & 1440m divided by σ̂_{t,60}·√w
    """
    exprs: list[pl.Expr] = []

    # Rolling Z-score of ln(x_close)
    for w in _TENSION_WINDOWS:
        exprs.append(_rolling_zscore("_ln_close", w, f"zscore_lnclose_{w}m"))

    df = df.with_columns(exprs)

    # Rolling percentile rank via Numba @njit
    close_np = df["x_close"].to_numpy().astype(np.float64)
    seg_ids_np = df["segment_id"].to_numpy().astype(np.int32)
    steps_np = df["steps_in_segment"].to_numpy().astype(np.int64)

    rank_cols: list[pl.Series] = []
    for w in _TENSION_WINDOWS:
        rk = _rolling_percentile_rank_kernel(close_np, seg_ids_np, steps_np, w)
        rank_cols.append(pl.Series(f"pctrank_close_{w}m", rk, dtype=pl.Float64))
    df = df.with_columns(rank_cols)

    # Distance from rolling max / min over 240m and 1440m, vol-normalized
    dist_exprs: list[pl.Expr] = []
    for w in [240, 1440]:
        scale = pl.col("sigma_60") * math.sqrt(w) + _EPS
        roll_max = pl.col("x_close").rolling_max(window_size=w, min_periods=w)
        roll_min = pl.col("x_close").rolling_min(window_size=w, min_periods=w)

        dist_hi = _mask_gap((pl.col("x_close") - roll_max) / scale, w).alias(f"dist_from_hi_{w}m")
        dist_lo = _mask_gap((pl.col("x_close") - roll_min) / scale, w).alias(f"dist_from_lo_{w}m")
        dist_exprs.extend([dist_hi, dist_lo])

    return df.with_columns(dist_exprs)


def _group4_streaks(df: pl.DataFrame) -> pl.DataFrame:
    """Group 4: Streaks & Fatigue.

    - Signed consecutive directional run length (streak_len, Numba @njit)
    - Steps since 60m & 240m rolling high/low (Numba @njit)
    - Fraction of positive 1m steps over w ∈ {15, 60}
    """
    # 1. Signed streak length
    r1_np = df["r_1"].fill_null(0.0).to_numpy().astype(np.float64)
    signs = np.sign(r1_np).astype(np.int8)
    seg_ids = df["segment_id"].to_numpy().astype(np.int32)
    steps = df["steps_in_segment"].to_numpy().astype(np.int64)

    streak_arr = _streak_kernel(signs, seg_ids, steps)
    df = df.with_columns(pl.Series("streak_len", streak_arr, dtype=pl.Float64))

    # 2. Steps since rolling high/low
    close_np = df["x_close"].to_numpy().astype(np.float64)
    hilo_cols: list[pl.Series] = []
    for w in [60, 240]:
        hi_s, lo_s = _steps_since_hilo_kernel(close_np, seg_ids, steps, w)
        hilo_cols.extend([
            pl.Series(f"steps_since_hi_{w}", hi_s, dtype=pl.Float64),
            pl.Series(f"steps_since_lo_{w}", lo_s, dtype=pl.Float64),
            pl.Series(f"steps_since_hi_{w}m", hi_s, dtype=pl.Float64),
            pl.Series(f"steps_since_lo_{w}m", lo_s, dtype=pl.Float64),
        ])
    df = df.with_columns(hilo_cols)

    # 3. Fraction of positive 1m steps over w ∈ {15, 60}
    pos_flag = (pl.col("r_1") > 0.0).cast(pl.Float64)
    frac_exprs: list[pl.Expr] = []
    for w in [15, 60]:
        roll_pos = pos_flag.rolling_mean(window_size=w, min_periods=w)
        frac_exprs.append(_mask_gap(roll_pos, w).alias(f"frac_pos_{w}m"))
    df = df.with_columns(frac_exprs)

    return df


def _group5_aggression(df: pl.DataFrame) -> pl.DataFrame:
    """Group 5: Aggression & Intensity.

    - 1m aggression imbalance: I_t = (agg_up - agg_down) / (agg_up + agg_down + 1e-12)
    - Rolling volume-weighted aggression imbalance over w ∈ {5, 15, 60}
    - Log-intensity Z-scores over trailing 1440 steps:
        Z(ln(activity_mag + 1))
        Z(ln(event_count + 1))
        Z(ln(activity_mag / (event_count + 1) + 1))
    """
    agg_sum = pl.col("aggression_up") + pl.col("aggression_down") + _EPS
    i_t_raw = (pl.col("aggression_up") - pl.col("aggression_down")) / agg_sum
    i_t = _mask_gap(i_t_raw, 1)

    df = df.with_columns([
        i_t.alias("agg_imb_1m"),
        i_t.alias("I_t"),
        (i_t * pl.col("activity_mag")).alias("_iv"),
    ])

    # Rolling volume-weighted aggression imbalance
    vw_exprs: list[pl.Expr] = []
    for w in [5, 15, 60]:
        sum_iv = pl.col("_iv").rolling_sum(window_size=w, min_periods=w)
        sum_v = pl.col("activity_mag").rolling_sum(window_size=w, min_periods=w)
        vw_imb = sum_iv / (sum_v + _EPS)
        vw_exprs.append(_mask_gap(vw_imb, w).alias(f"agg_imb_vw_{w}m"))
    df = df.with_columns(vw_exprs)

    # abs_agg_imbalance_15 (required by HMM_OBS_FEATURES)
    df = df.with_columns(
        pl.col("agg_imb_vw_15m").abs().alias("abs_agg_imbalance_15")
    )

    # Log-intensity Z-scores over trailing 1440 steps
    df = df.with_columns([
        (pl.col("activity_mag") + 1.0).log().alias("_ln_act"),
        (pl.col("event_count").cast(pl.Float64) + 1.0).log().alias("_ln_evcount"),
        ((pl.col("activity_mag") / (pl.col("event_count").cast(pl.Float64) + 1.0)) + 1.0).log().alias("_ln_avg_evsize"),
    ])

    z_exprs = [
        _rolling_zscore("_ln_act", 1440, "z_ln_activity_1440m"),
        _rolling_zscore("_ln_evcount", 1440, "z_ln_evcount_1440m"),
        _rolling_zscore("_ln_avg_evsize", 1440, "z_avg_evsize_1440m"),
    ]
    df = df.with_columns(z_exprs)

    return df


def _group6_crowd(df: pl.DataFrame) -> pl.DataFrame:
    """Group 6: Tier 2 Crowd State.

    - Commitment log-velocity: ln(commitment_t / commitment_{t-w}) for w ∈ {15, 60, 240}
    - Commitment-price tension product: comm_vel_60 · r_{t,60}
    - Rolling 7-day (10080) Z-scores of skew_general, skew_top20, skew_div
    - Rolling 7-day Z-score of crowding_penalty, interaction term crowding_penalty_z · comm_vel_60
    """
    exprs: list[pl.Expr] = []

    for w in _COMMITMENT_WINDOWS:
        vel_raw = (pl.col("commitment") / (pl.col("commitment").shift(w) + _EPS)).log()
        vel_masked = _mask_gap(vel_raw, w + 1)
        exprs.extend([
            vel_masked.alias(f"comm_vel_{w}m"),
            vel_masked.alias(f"comm_vel_{w}"),
        ])
    df = df.with_columns(exprs)

    # Commitment-price tension product: comm_vel_60 · r_{t,60}
    tension_raw = pl.col("comm_vel_60m") * pl.col("r_60m")
    df = df.with_columns(
        _mask_gap(tension_raw, 61).alias("comm_price_tension")
    )

    # Rolling 7-day (10080) Z-scores
    df = df.with_columns(
        (pl.col("skew_top20") - pl.col("skew_general")).alias("_skew_div")
    )

    z_exprs = [
        _rolling_zscore("skew_general", 10080, "skew_general_z"),
        _rolling_zscore("skew_top20", 10080, "skew_top20_z"),
        _rolling_zscore("_skew_div", 10080, "skew_div_z"),
        _rolling_zscore("crowding_penalty", 10080, "crowding_penalty_z"),
    ]
    df = df.with_columns(z_exprs)

    # Interaction term: crowding_penalty_z · comm_vel_60
    interaction = pl.col("crowding_penalty_z") * pl.col("comm_vel_60m")
    df = df.with_columns(
        interaction.alias("crowding_comm_interaction")
    )

    return df


def _group7_vol_disorder(df: pl.DataFrame) -> pl.DataFrame:
    """Group 7: Loudness vs. Disorder.

    - Realized vol σ̂_{t,w} = sqrt(rolling_mean(r1^2, w)) for w ∈ {15, 60, 240, 1440}
    - Log realized vol log_rv_w = ln(σ̂_{t,w})
    - Garman-Klass realized vol: sqrt(rolling_mean(v_GK, w))
    - Compression/expansion ratios: σ̂_15 / σ̂_240, σ̂_60 / σ̂_1440
    - Vol-of-vol: rolling std of ln(σ̂_15) over 240m
    - Causal rolling Permutation Entropy (m=4, tau=1) over w ∈ {60, 240} (Numba @njit)
    """
    vol_exprs: list[pl.Expr] = []

    # Realized volatility from returns
    for w in _VOL_WINDOWS:
        rv_raw = (
            pl.col("_r1_sq")
            .rolling_mean(window_size=w, min_periods=w)
            .sqrt()
            .clip(lower_bound=_VOL_FLOOR)
        )
        vol_exprs.extend([
            _mask_gap(rv_raw, w).alias(f"rvol_{w}m"),
            _mask_gap(rv_raw, w).alias(f"rvol_{w}"),
            _mask_gap(rv_raw.log(), w).alias(f"log_rv_{w}"),
        ])

    # Realized volatility from Garman-Klass
    for w in _VOL_WINDOWS:
        gk_rv_raw = (
            pl.col("gk_var")
            .rolling_mean(window_size=w, min_periods=w)
            .sqrt()
            .clip(lower_bound=_VOL_FLOOR)
        )
        vol_exprs.append(_mask_gap(gk_rv_raw, w).alias(f"rvol_gk_{w}m"))

    df = df.with_columns(vol_exprs)

    # Compression / expansion ratios
    df = df.with_columns([
        _mask_gap(pl.col("rvol_15m") / (pl.col("rvol_240m") + _EPS), 240).alias("vol_ratio_15_240"),
        _mask_gap(pl.col("rvol_60m") / (pl.col("rvol_1440m") + _EPS), 1440).alias("vol_ratio_60_1440"),
    ])

    # Vol-of-vol: rolling std of ln(σ̂_15) over 240m
    vov_raw = pl.col("log_rv_15").rolling_std(window_size=240, min_periods=240)
    df = df.with_columns(
        _mask_gap(vov_raw, 240).alias("vol_of_vol_240m")
    )

    # Causal Permutation Entropy (m=4, tau=1) via Numba
    r1_np = df["r_1"].fill_null(0.0).to_numpy().astype(np.float64)
    seg_ids = df["segment_id"].to_numpy().astype(np.int32)
    steps = df["steps_in_segment"].to_numpy().astype(np.int64)

    pe_cols: list[pl.Series] = []
    for w in _PE_WINDOWS:
        pe_arr = _perm_entropy_kernel(r1_np, seg_ids, steps, w, _PE_M, _PE_TAU)
        pe_cols.extend([
            pl.Series(f"perm_entropy_{w}", pe_arr, dtype=pl.Float64),
            pl.Series(f"perm_entropy_{w}m", pe_arr, dtype=pl.Float64),
        ])
    df = df.with_columns(pe_cols)

    return df


def _group8_cyclical(df: pl.DataFrame) -> pl.DataFrame:
    """Group 8: Cyclical Time.

    - sin(2π·min_of_day / 1440), cos(2π·min_of_day / 1440)
    - sin(2π·min_of_week / 10080), cos(2π·min_of_week / 10080)
    """
    two_pi = 2.0 * math.pi
    dt = pl.col("close_time_ms").cast(pl.Datetime("ms", time_zone="UTC"))

    min_of_day = dt.dt.hour() * 60 + dt.dt.minute()
    min_of_week = dt.dt.weekday() * 1440 + min_of_day

    return df.with_columns([
        (min_of_day.cast(pl.Float64) / 1440.0 * two_pi).sin().alias("sin_day"),
        (min_of_day.cast(pl.Float64) / 1440.0 * two_pi).cos().alias("cos_day"),
        (min_of_week.cast(pl.Float64) / 10080.0 * two_pi).sin().alias("sin_week"),
        (min_of_week.cast(pl.Float64) / 10080.0 * two_pi).cos().alias("cos_week"),
    ])


# ===================================================================
# SECTION D — Public Entry Point
# ===================================================================

_CLEANUP_COLS = [
    "_r1_sq",
    "_ln_close",
    "_iv",
    "_ln_act",
    "_ln_evcount",
    "_ln_avg_evsize",
    "_skew_div",
]


def compute_all_features(master_df: pl.DataFrame) -> pl.DataFrame:
    """Compute all 8 causal behavioral feature groups and return master DataFrame
    with all feature columns appended.

    Guarantees:
      - Zero lookahead leakage: all operations backward-looking.
      - Segment awareness: any rolling window W outputs null where steps_in_segment < W.
      - Gap masking: all gap rows output null for rolling features.
      - Scale invariance: all outputs normalized (log-ratios, Z-scores, ranks, vol-scaled).

    Parameters
    ----------
    master_df : pl.DataFrame
        Aligned master dataset containing Tier 1, backward-joined Tier 2,
        and gap/segment indicator columns.

    Returns
    -------
    pl.DataFrame
        Master DataFrame with all feature columns appended.
    """
    # 1. Enforce strict ascending chronological sort
    df = master_df.sort("close_time_ms")

    # 2. Base returns, log prices, and baseline vol σ̂_{t,60}
    df = _compute_base_series(df)

    # 3. Group 2: Intrabar Structure (produces gk_var needed by Group 7)
    df = _group2_intrabar(df)

    # 4. Group 1: Kinematics
    df = _group1_kinematics(df)

    # 5. Group 3: Tension & Stretch
    df = _group3_tension(df)

    # 6. Group 4: Streaks & Fatigue
    df = _group4_streaks(df)

    # 7. Group 5: Aggression & Intensity
    df = _group5_aggression(df)

    # 8. Group 6: Tier 2 Crowd State
    df = _group6_crowd(df)

    # 9. Group 7: Loudness vs. Disorder
    df = _group7_vol_disorder(df)

    # 10. Group 8: Cyclical Time
    df = _group8_cyclical(df)

    # 11. Drop internal intermediate helper columns
    drop_targets = [c for c in _CLEANUP_COLS if c in df.columns]
    if drop_targets:
        df = df.drop(drop_targets)

    return df
