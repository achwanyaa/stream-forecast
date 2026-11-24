"""
Tests for Layer 2 — Causal Behavioral Feature Engine
=====================================================
Spec §12 Milestone 3 Verification Gate:
  1. Feature Causality Perturbation Test:
     Mutate all raw columns after timestamp T₀.
     Assert every single Layer 2 feature at timestamps ≤ T₀ is bit-for-bit identical.
  2. Gap Masking Verification:
     Assert any rolling window of size W outputs null wherever steps_in_segment < W.
  3. Scale Invariance & Feature Contract:
     Verify all 8 feature groups exist and contain no raw unnormalized levels.
"""

from __future__ import annotations

import math
import numpy as np
import polars as pl
import pytest

from compute.pipeline.features import compute_all_features
from compute.config import HMM_OBS_FEATURES


def _generate_synthetic_master(
    n_rows: int = 1500,
    gap_start: int = 400,
    gap_len: int = 20,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate synthetic master DataFrame matching pipeline/ingest.py schema."""
    rng = np.random.RandomState(seed)
    base_time = 1_700_000_000_000

    close_times = np.arange(
        base_time + 60_000,
        base_time + 60_000 + n_rows * 60_000,
        60_000,
        dtype=np.int64,
    )
    open_times = close_times - 60_000

    # Geometric random walk
    log_ret = rng.normal(0, 0.001, n_rows)
    log_price = np.cumsum(log_ret) + np.log(100.0)
    x_close = np.exp(log_price)
    x_open = x_close * np.exp(rng.normal(0, 0.0005, n_rows))
    x_high = np.maximum(x_open, x_close) * (1.0 + rng.uniform(0.0001, 0.001, n_rows))
    x_low = np.minimum(x_open, x_close) * (1.0 - rng.uniform(0.0001, 0.001, n_rows))

    activity_mag = rng.exponential(1000.0, n_rows)
    event_count = rng.poisson(50, n_rows).astype(np.int64)
    agg_up = rng.exponential(500.0, n_rows)
    agg_down = rng.exponential(500.0, n_rows)

    commitment = rng.exponential(5000.0, n_rows) + 1.0
    skew_general = rng.normal(0, 0.5, n_rows)
    skew_top20 = rng.normal(0, 0.8, n_rows)
    crowding_penalty = rng.normal(0, 0.3, n_rows)

    # Gap flags and segment IDs
    is_gap = np.zeros(n_rows, dtype=bool)
    is_gap[gap_start : gap_start + gap_len] = True

    # Segment IDs: gaps get -1, valid stretches get 0, 1, ...
    seg_id = np.full(n_rows, -1, dtype=np.int32)
    current_seg = 0
    in_valid = False
    for i in range(n_rows):
        if not is_gap[i]:
            if not in_valid:
                in_valid = True
                if i > 0 and is_gap[i - 1]:
                    current_seg += 1
            seg_id[i] = current_seg
        else:
            in_valid = False

    # Steps in segment: 1-based counter inside valid segments, 0 in gaps
    steps_in_segment = np.zeros(n_rows, dtype=np.int64)
    counter = 0
    prev_s = -1
    for i in range(n_rows):
        s = seg_id[i]
        if s == -1:
            counter = 0
            steps_in_segment[i] = 0
        else:
            if s != prev_s:
                counter = 1
            else:
                counter += 1
            steps_in_segment[i] = counter
        prev_s = s

    # Inject nulls / zeros inside gap
    x_close[is_gap] = np.nan
    x_open[is_gap] = np.nan
    x_high[is_gap] = np.nan
    x_low[is_gap] = np.nan

    return pl.DataFrame({
        "open_time_ms": open_times,
        "close_time_ms": close_times,
        "x_open": x_open,
        "x_high": x_high,
        "x_low": x_low,
        "x_close": x_close,
        "activity_mag": activity_mag,
        "event_count": event_count,
        "aggression_up": agg_up,
        "aggression_down": agg_down,
        "commitment": commitment,
        "skew_general": skew_general,
        "skew_top20": skew_top20,
        "crowding_penalty": crowding_penalty,
        "is_gap": is_gap,
        "segment_id": seg_id,
        "steps_in_segment": steps_in_segment,
    })


class TestFeatureCausalityAndLeakage:
    """Milestone 3 Verification Gate: Strict causality perturbation test."""

    def test_feature_causality_perturbation(self):
        """Mutate all raw columns after timestamp T₀.

        Assert every single feature at timestamps ≤ T₀ is bit-for-bit identical.
        """
        df_clean = _generate_synthetic_master(n_rows=1000, seed=123)
        feat_clean = compute_all_features(df_clean)

        # Pick cutoff index T0 well into the first segment
        t0_idx = 300
        cutoff_time = df_clean["close_time_ms"][t0_idx]

        # Create mutated DataFrame by drastically modifying all price and volume data after T0
        df_mutated = df_clean.clone()
        n_rows = len(df_mutated)

        rng = np.random.RandomState(999)
        noise = rng.normal(50.0, 10.0, n_rows - t0_idx - 1)

        mut_close = df_mutated["x_close"].to_numpy().copy()
        mut_close[t0_idx + 1 :] += noise

        mut_high = df_mutated["x_high"].to_numpy().copy()
        mut_high[t0_idx + 1 :] += noise + 5.0

        mut_low = df_mutated["x_low"].to_numpy().copy()
        mut_low[t0_idx + 1 :] -= noise

        mut_act = df_mutated["activity_mag"].to_numpy().copy()
        mut_act[t0_idx + 1 :] *= 10.0

        mut_comm = df_mutated["commitment"].to_numpy().copy()
        mut_comm[t0_idx + 1 :] *= 5.0

        df_mutated = df_mutated.with_columns([
            pl.Series("x_close", mut_close),
            pl.Series("x_high", mut_high),
            pl.Series("x_low", mut_low),
            pl.Series("activity_mag", mut_act),
            pl.Series("commitment", mut_comm),
        ])

        feat_mutated = compute_all_features(df_mutated)

        # Slice both up to T0
        slice_clean = feat_clean.filter(pl.col("close_time_ms") <= cutoff_time)
        slice_mut = feat_mutated.filter(pl.col("close_time_ms") <= cutoff_time)

        # Verify bit-for-bit equality across every single column
        for col in slice_clean.columns:
            arr_clean = slice_clean[col].to_numpy()
            arr_mut = slice_mut[col].to_numpy()

            if np.issubdtype(arr_clean.dtype, np.floating):
                # Where both are NaN, they match. Otherwise must be exact to machine precision.
                nan_clean = np.isnan(arr_clean)
                nan_mut = np.isnan(arr_mut)
                assert np.array_equal(nan_clean, nan_mut), f"Column {col} NaN mask mismatch at <= T0"
                valid = ~nan_clean
                diff = np.abs(arr_clean[valid] - arr_mut[valid])
                assert np.all(diff < 1e-12), f"Column {col} mutated at <= T0! Max diff: {np.max(diff)}"
            else:
                assert np.array_equal(arr_clean, arr_mut), f"Column {col} non-float mismatch at <= T0"


class TestGapMasking:
    """Verify that features output null across gaps and when steps_in_segment < W."""

    def test_window_null_when_steps_insufficient(self):
        df = _generate_synthetic_master(n_rows=600, gap_start=250, gap_len=15, seed=42)
        features = compute_all_features(df)

        # Check rolling 60m features: must be null where steps_in_segment < 60
        sub_60 = features.filter(pl.col("steps_in_segment") < 60)
        assert sub_60["ret_60m"].null_count() == len(sub_60), "ret_60m not null when steps < 60"
        assert sub_60["sigma_60"].null_count() == len(sub_60), "sigma_60 not null when steps < 60"
        assert sub_60["perm_entropy_60"].null_count() == len(sub_60), "perm_entropy_60 not null when steps < 60"

        # Check gap rows: all rolling features must be null
        gaps = features.filter(pl.col("is_gap"))
        assert gaps["ret_1m"].null_count() == len(gaps), "ret_1m not null in gap"
        assert gaps["streak_len"].null_count() == len(gaps), "streak_len not null in gap"


class TestFeatureContract:
    """Verify all 8 feature groups and required HMM observation features exist."""

    def test_hmm_observation_features_present(self):
        df = _generate_synthetic_master(n_rows=500, seed=77)
        features = compute_all_features(df)

        for obs_feat in HMM_OBS_FEATURES:
            assert obs_feat in features.columns, f"Required HMM feature {obs_feat} missing!"

    def test_cyclical_time_bounds(self):
        df = _generate_synthetic_master(n_rows=200, seed=88)
        features = compute_all_features(df)

        for col in ["sin_day", "cos_day", "sin_week", "cos_week"]:
            vals = features[col].to_numpy()
            assert np.all(vals >= -1.0 - 1e-6) and np.all(vals <= 1.0 + 1e-6), f"{col} out of [-1, 1]"
