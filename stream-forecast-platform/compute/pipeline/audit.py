"""
pipeline/audit.py — Spec §3: Phase 1 Randomness & Structure Audit.

Runs on master.parquet (filtered to valid contiguous segments) before training
any models. All results are serialized into an audit_report dict (JSON-safe).
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import polars as pl
from scipy import signal as sp_signal
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy_clean(series: pl.Series) -> np.ndarray:
    """Drop nulls and convert a Polars series to a 1-D float64 numpy array."""
    return series.drop_nulls().to_numpy().astype(np.float64)


def _acf(x: np.ndarray, max_lag: int = 60) -> list[float]:
    """Compute sample autocorrelation for lags 1..max_lag."""
    n = len(x)
    if n < max_lag + 1:
        return [float("nan")] * max_lag
    xm = x - x.mean()
    c0 = np.dot(xm, xm) / n
    if c0 == 0:
        return [0.0] * max_lag
    result = []
    for lag in range(1, max_lag + 1):
        ck = np.dot(xm[: n - lag], xm[lag:]) / n
        result.append(float(ck / c0))
    return result


def _ljung_box(acf_vals: list[float], n: int, lags: list[int]) -> dict[int, float]:
    """Ljung-Box Q statistic p-values at specified lags."""
    result = {}
    for m in lags:
        if m > len(acf_vals):
            result[m] = float("nan")
            continue
        q = 0.0
        for k in range(1, m + 1):
            rho_k = acf_vals[k - 1]
            q += (rho_k ** 2) / (n - k)
        q *= n * (n + 2)
        pval = 1.0 - float(sp_stats.chi2.cdf(q, df=m))
        result[m] = pval
    return result


def _permutation_entropy(x: np.ndarray, m: int = 4, tau: int = 1) -> float:
    """
    Compute normalized permutation entropy.
    m=4, tau=1, normalized by ln(m!) = ln(24).
    """
    n = len(x)
    if n < m:
        return float("nan")

    factorial_m = math.factorial(m)
    counts: dict[tuple, int] = {}
    total = 0

    for i in range(n - (m - 1) * tau):
        motif = tuple(np.argsort(x[i: i + m * tau: tau]).tolist())
        counts[motif] = counts.get(motif, 0) + 1
        total += 1

    if total == 0:
        return float("nan")

    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            h -= p * math.log(p)

    # Normalize by ln(m!)
    return float(h / math.log(factorial_m))


def _rolling_permutation_entropy(
    x: np.ndarray, window: int = 240, m: int = 4, tau: int = 1
) -> np.ndarray:
    """Rolling permutation entropy over a 1-D array with a given window."""
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        seg = x[i - window + 1: i + 1]
        result[i] = _permutation_entropy(seg, m=m, tau=tau)
    return result


def _deterministic_sample(arr1: np.ndarray, arr2: np.ndarray, n_samples: int = 15_000, seed: int = 42) -> dict:
    """
    Deterministic reservoir sample of n_samples from (arr1, arr2),
    clipped to 0.5%–99.5% percentiles.
    """
    # Remove entries where either is NaN
    mask = ~(np.isnan(arr1) | np.isnan(arr2))
    a1 = arr1[mask]
    a2 = arr2[mask]

    if len(a1) == 0:
        return {"x": [], "y": []}

    # Clip to 0.5%–99.5% percentiles
    lo1, hi1 = np.percentile(a1, [0.5, 99.5])
    lo2, hi2 = np.percentile(a2, [0.5, 99.5])
    clip_mask = (a1 >= lo1) & (a1 <= hi1) & (a2 >= lo2) & (a2 <= hi2)
    a1 = a1[clip_mask]
    a2 = a2[clip_mask]

    if len(a1) <= n_samples:
        return {"x": a1.tolist(), "y": a2.tolist()}

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(a1), size=n_samples, replace=False)
    idx.sort()
    return {"x": a1[idx].tolist(), "y": a2[idx].tolist()}


def _safe_float(v: Any) -> Any:
    """Make a value JSON-serializable (handle numpy scalars, NaN, Inf)."""
    if isinstance(v, (np.floating, np.integer)):
        v = float(v)
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
    return v


def _sanitize(obj: Any) -> Any:
    """Recursively sanitize an object for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_sanitize(v) for v in obj.tolist()]
    return _safe_float(obj)


# ---------------------------------------------------------------------------
# Main audit entry point
# ---------------------------------------------------------------------------

def run_audit(master_parquet_path: str) -> dict:
    """
    Execute the Phase 1 Randomness & Structure Audit on the aligned master
    dataset.  Returns a fully JSON-serializable audit_report dict.

    Parameters
    ----------
    master_parquet_path : path to master.parquet produced by ingest.py

    Returns
    -------
    audit_report : dict  (all values JSON-safe)
    """

    df = pl.read_parquet(master_parquet_path)
    report: dict[str, Any] = {}

    # ==================================================================
    # A. Gap & Data Integrity Summary
    # ==================================================================
    total_rows = len(df)
    ct_min = int(df["close_time_ms"].min())
    ct_max = int(df["close_time_ms"].max())
    total_gap_rows = int(df.filter(pl.col("is_gap")).height)
    total_gap_minutes = total_gap_rows  # each row = 1 minute

    # Number of gap outages = number of contiguous gap stretches
    gap_flag = df["is_gap"].to_numpy()
    gap_starts = 0
    longest_outage = 0
    current_outage = 0
    for g in gap_flag:
        if g:
            current_outage += 1
        else:
            if current_outage > 0:
                gap_starts += 1
                longest_outage = max(longest_outage, current_outage)
            current_outage = 0
    if current_outage > 0:
        gap_starts += 1
        longest_outage = max(longest_outage, current_outage)

    # Tier 2 missingness
    t2_5m_null_rate = float(df["t2_5m_staleness_min"].is_null().mean()) if "t2_5m_staleness_min" in df.columns else None
    t2_8h_null_rate = float(df["t2_8h_staleness_min"].is_null().mean()) if "t2_8h_staleness_min" in df.columns else None

    # Tier 2 value missingness (after staleness null-out)
    commitment_null_rate = float(df["commitment"].is_null().mean()) if "commitment" in df.columns else None
    crowding_null_rate = float(df["crowding_penalty"].is_null().mean()) if "crowding_penalty" in df.columns else None

    report["gap_summary"] = {
        "total_rows": total_rows,
        "date_span_ms": ct_max - ct_min,
        "total_gap_minutes": total_gap_minutes,
        "num_gap_outages": gap_starts,
        "longest_outage_minutes": longest_outage,
        "tier2_5m_join_null_rate": t2_5m_null_rate,
        "tier2_8h_join_null_rate": t2_8h_null_rate,
        "commitment_null_rate": commitment_null_rate,
        "crowding_penalty_null_rate": crowding_null_rate,
    }

    # ==================================================================
    # B. Core Derived Series (computed on valid, non-gap rows)
    # ==================================================================
    valid = df.filter(~pl.col("is_gap")).sort("close_time_ms")

    # r_t = ln(x_close_t / x_close_{t-1}), null when steps_in_segment <= 1
    valid = valid.with_columns(
        pl.when(pl.col("steps_in_segment") > 1)
        .then((pl.col("x_close") / pl.col("x_close").shift(1)).log())
        .otherwise(pl.lit(None))
        .alias("r_t")
    )

    # |r_t| and r_t^2
    valid = valid.with_columns([
        pl.col("r_t").abs().alias("abs_r_t"),
        (pl.col("r_t") ** 2).alias("r_t_sq"),
    ])

    # Aggression imbalance I_t
    valid = valid.with_columns(
        (
            (pl.col("aggression_up") - pl.col("aggression_down"))
            / (pl.col("aggression_up") + pl.col("aggression_down") + 1e-12)
        ).alias("I_t")
    )

    # Numpy arrays for analysis
    r_t = _to_numpy_clean(valid["r_t"])
    abs_r_t = _to_numpy_clean(valid["abs_r_t"])
    r_t_sq = _to_numpy_clean(valid["r_t_sq"])
    I_t = _to_numpy_clean(valid["I_t"])

    # ==================================================================
    # C. Autocorrelation & Ljung-Box Suite
    # ==================================================================
    lb_lags = [1, 5, 15, 60]
    acf_report: dict[str, Any] = {}

    for name, arr in [("r_t", r_t), ("abs_r_t", abs_r_t), ("r_t_sq", r_t_sq), ("I_t", I_t)]:
        acf_vals = _acf(arr, max_lag=60)
        lb = _ljung_box(acf_vals, len(arr), lb_lags)
        acf_report[name] = {
            "acf_1_60": acf_vals,
            "ljung_box": {str(k): v for k, v in lb.items()},
        }

    report["autocorrelation"] = acf_report

    # ==================================================================
    # D. Spectral Analysis — Welch PSD
    # ==================================================================
    # Use segments ≥ 20160 minutes
    psd_report: dict[str, Any] = {}
    segment_ids = valid["segment_id"].unique().sort().to_list()

    for series_name, series_col in [("abs_r_t", "abs_r_t"), ("activity_mag", "activity_mag")]:
        psd_segments = []
        for sid in segment_ids:
            if sid < 0:
                continue
            seg = valid.filter(pl.col("segment_id") == sid)
            if len(seg) < 20_160:
                continue
            arr = _to_numpy_clean(seg[series_col])
            if len(arr) < 20_160:
                continue
            psd_segments.append(arr)

        if psd_segments:
            # Average Welch periodograms across segments
            all_freqs = None
            all_psd = None
            for arr in psd_segments:
                freqs, psd_vals = sp_signal.welch(
                    arr, fs=1.0, nperseg=min(8192, len(arr)), noverlap=None
                )
                if all_freqs is None:
                    all_freqs = freqs
                    all_psd = psd_vals
                else:
                    # Interpolate to common freq grid
                    all_psd = all_psd + np.interp(all_freqs, freqs, psd_vals)
            all_psd /= len(psd_segments)

            # Peak power near diurnal (1/1440) and weekly (1/10080)
            diurnal_freq = 1.0 / 1440.0
            weekly_freq = 1.0 / 10080.0

            diurnal_idx = int(np.argmin(np.abs(all_freqs - diurnal_freq)))
            weekly_idx = int(np.argmin(np.abs(all_freqs - weekly_freq)))

            psd_report[series_name] = {
                "num_segments_used": len(psd_segments),
                "freq_resolution": float(all_freqs[1] - all_freqs[0]) if len(all_freqs) > 1 else None,
                "diurnal_peak_power": float(all_psd[diurnal_idx]),
                "weekly_peak_power": float(all_psd[weekly_idx]),
                "freqs": all_freqs.tolist(),
                "psd": all_psd.tolist(),
            }
        else:
            psd_report[series_name] = {
                "num_segments_used": 0,
                "note": "No segments >= 20160 minutes",
            }

    report["welch_psd"] = psd_report

    # ==================================================================
    # E. Tail & Asymmetry Profile
    # ==================================================================
    if len(r_t) > 10:
        excess_kurt = float(sp_stats.kurtosis(r_t, fisher=True))
        skewness = float(sp_stats.skew(r_t))

        # Standardize to z-scores
        mu, sigma = r_t.mean(), r_t.std()
        z_t = (r_t - mu) / (sigma + 1e-15)

        p_abs_z_gt5 = float(np.mean(np.abs(z_t) > 5))
        p_abs_z_gt10 = float(np.mean(np.abs(z_t) > 10))

        # Gaussian expectations
        gauss_p5 = float(2 * (1 - sp_stats.norm.cdf(5)))   # ~5.73e-7
        gauss_p10 = float(2 * (1 - sp_stats.norm.cdf(10)))  # ~1.52e-23

        # |q01| / q99 ratio
        q01 = float(np.percentile(r_t, 1))
        q99 = float(np.percentile(r_t, 99))
        tail_ratio = abs(q01) / (q99 + 1e-15) if q99 != 0 else None

        report["tail_profile"] = {
            "excess_kurtosis": excess_kurt,
            "skewness": skewness,
            "p_abs_z_gt_5": p_abs_z_gt5,
            "p_abs_z_gt_5_gaussian": gauss_p5,
            "p_abs_z_gt_10": p_abs_z_gt10,
            "p_abs_z_gt_10_gaussian": gauss_p10,
            "q01": q01,
            "q99": q99,
            "abs_q01_over_q99": tail_ratio,
        }
    else:
        report["tail_profile"] = {"note": "Insufficient data"}

    # ==================================================================
    # F. Permutation Entropy — Global & Rolling (240-min)
    # ==================================================================
    pe_global = _permutation_entropy(r_t, m=4, tau=1)
    pe_rolling = _rolling_permutation_entropy(r_t, window=240, m=4, tau=1)

    pe_all_nan = bool(np.all(np.isnan(pe_rolling)))
    report["permutation_entropy"] = {
        "global": pe_global,
        "rolling_240_min": {
            "mean": None if pe_all_nan else _safe_float(np.nanmean(pe_rolling)),
            "std": None if pe_all_nan else _safe_float(np.nanstd(pe_rolling)),
            "min": None if pe_all_nan else _safe_float(np.nanmin(pe_rolling)),
            "max": None if pe_all_nan else _safe_float(np.nanmax(pe_rolling)),
        },
    }

    # ==================================================================
    # G. Diurnal & Weekly Seasonality Grid (7x24 heatmap)
    # ==================================================================
    # Derive dow (0=Monday..6=Sunday) and hour from close_time_ms (UTC)
    valid_with_time = valid.with_columns([
        (pl.col("close_time_ms") * 1_000).cast(pl.Datetime("us")).dt.weekday().sub(1).alias("dow"),
        (pl.col("close_time_ms") * 1_000).cast(pl.Datetime("us")).dt.hour().alias("hour_utc"),
    ])

    heatmap_report: dict[str, Any] = {}
    for metric_name in ["abs_r_t", "activity_mag"]:
        grid_data = (
            valid_with_time
            .filter(pl.col(metric_name).is_not_null())
            .group_by(["dow", "hour_utc"])
            .agg([
                pl.col(metric_name).mean().alias("mean"),
                pl.col(metric_name).median().alias("median"),
            ])
            .sort(["dow", "hour_utc"])
        )
        heatmap_report[metric_name] = {
            "dow": grid_data["dow"].to_list(),
            "hour_utc": grid_data["hour_utc"].to_list(),
            "mean": [_safe_float(v) for v in grid_data["mean"].to_list()],
            "median": [_safe_float(v) for v in grid_data["median"].to_list()],
        }

    report["diurnal_weekly_heatmap"] = heatmap_report

    # ==================================================================
    # H. 2D Lag Structure Sample
    # ==================================================================
    # (r_t, r_{t+1})
    r_t_lead = r_t[:-1]
    r_t_next = r_t[1:]
    report["lag_scatter_r_t"] = _deterministic_sample(r_t_lead, r_t_next, n_samples=15_000, seed=42)

    # (|r_t|, |r_{t+1}|)
    abs_r_t_lead = abs_r_t[:-1]
    abs_r_t_next = abs_r_t[1:]
    report["lag_scatter_abs_r_t"] = _deterministic_sample(abs_r_t_lead, abs_r_t_next, n_samples=15_000, seed=42)

    # ==================================================================
    # Sanitize entire report for JSON safety
    # ==================================================================
    report = _sanitize(report)

    return report
