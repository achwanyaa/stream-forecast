"""
tests/test_leakage.py — Multi-Rate Perturbation Test (Spec §12 Gate).

Proves zero future leakage: mutating Tier 2 data after timestamp T₀
must NOT change any aligned row at or before T₀ (bit-for-bit identical).
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal
import pytest

# Ensure both compute/ and repository root are on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

try:
    from compute.pipeline.ingest import ingest_and_align
except ModuleNotFoundError:
    from pipeline.ingest import ingest_and_align

SEED = 42


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def _generate_synthetic_tier1(n_minutes: int = 2000, seed: int = SEED) -> pl.DataFrame:
    """Generate synthetic Tier 1 (1-minute) data."""
    rng = np.random.RandomState(seed)
    base_time = 1_700_000_000_000  # arbitrary start (ms)

    close_times = np.arange(
        base_time + 60_000,
        base_time + 60_000 + n_minutes * 60_000,
        60_000,
        dtype=np.int64,
    )
    open_times = close_times - 60_000

    # Random walk for prices
    log_returns = rng.normal(0, 0.001, n_minutes)
    log_prices = np.cumsum(log_returns) + np.log(100.0)
    closes = np.exp(log_prices)
    opens = closes * np.exp(rng.normal(0, 0.0005, n_minutes))
    highs = np.maximum(opens, closes) * (1 + rng.uniform(0, 0.002, n_minutes))
    lows = np.minimum(opens, closes) * (1 - rng.uniform(0, 0.002, n_minutes))

    return pl.DataFrame({
        "open_time_ms": open_times,
        "close_time_ms": close_times,
        "x_open": opens,
        "x_high": highs,
        "x_low": lows,
        "x_close": closes,
        "activity_mag": rng.exponential(1000, n_minutes),
        "event_count": rng.poisson(50, n_minutes).astype(np.int64),
        "aggression_up": rng.exponential(500, n_minutes),
        "aggression_down": rng.exponential(500, n_minutes),
    })


def _generate_synthetic_tier2_5m(tier1: pl.DataFrame, seed: int = SEED) -> pl.DataFrame:
    """Generate synthetic Tier 2 5-minute data aligned to Tier 1 timeline."""
    rng = np.random.RandomState(seed + 1)
    ct_min = int(tier1["close_time_ms"].min())
    ct_max = int(tier1["close_time_ms"].max())

    close_times = np.arange(ct_min, ct_max + 1, 300_000, dtype=np.int64)
    open_times = close_times - 300_000
    n = len(close_times)

    return pl.DataFrame({
        "open_time_ms": open_times,
        "close_time_ms": close_times,
        "commitment": rng.exponential(5000, n) + 1.0,
        "skew_general": rng.normal(0, 0.5, n),
        "skew_top20": rng.normal(0, 0.8, n),
    })


def _generate_synthetic_tier2_8h(tier1: pl.DataFrame, seed: int = SEED) -> pl.DataFrame:
    """Generate synthetic Tier 2 8-hour data aligned to Tier 1 timeline."""
    rng = np.random.RandomState(seed + 2)
    ct_min = int(tier1["close_time_ms"].min())
    ct_max = int(tier1["close_time_ms"].max())

    close_times = np.arange(ct_min, ct_max + 1, 28_800_000, dtype=np.int64)
    open_times = close_times - 28_800_000
    n = len(close_times)

    return pl.DataFrame({
        "open_time_ms": open_times,
        "close_time_ms": close_times,
        "crowding_penalty": rng.normal(0, 0.3, n),
    })


# ---------------------------------------------------------------------------
# Test: Multi-Rate Perturbation — Zero Future Leakage
# ---------------------------------------------------------------------------

class TestMultiRatePerturbation:
    """
    §12 Verification Gate: Mutate Tier 2 for all close_time_ms > T₀.
    Assert aligned output for close_time_ms ≤ T₀ is bit-for-bit identical.
    """

    def test_tier2_5m_mutation_does_not_affect_past(self):
        """Mutating Tier 2 (5m) data after T₀ must not change any row ≤ T₀."""
        tier1 = _generate_synthetic_tier1(n_minutes=2000)
        tier2_5m = _generate_synthetic_tier2_5m(tier1)
        tier2_8h = _generate_synthetic_tier2_8h(tier1)

        # Choose T₀ as the midpoint of the Tier 1 timeline
        ct_min = int(tier1["close_time_ms"].min())
        ct_max = int(tier1["close_time_ms"].max())
        t0 = ct_min + ((ct_max - ct_min) // 2)
        # Snap to nearest minute boundary
        t0 = (t0 // 60_000) * 60_000

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save original data
            t1_path = os.path.join(tmpdir, "tier1.parquet")
            t2_5m_path = os.path.join(tmpdir, "tier2_5m.parquet")
            t2_8h_path = os.path.join(tmpdir, "tier2_8h.parquet")
            out_original = os.path.join(tmpdir, "master_original.parquet")
            out_mutated = os.path.join(tmpdir, "master_mutated.parquet")

            tier1.write_parquet(t1_path)
            tier2_5m.write_parquet(t2_5m_path)
            tier2_8h.write_parquet(t2_8h_path)

            # Run original alignment
            ingest_and_align(t1_path, t2_5m_path, t2_8h_path, out_original)

            # Mutate tier2_5m: multiply all values after T₀ by 999
            tier2_5m_mutated = tier2_5m.with_columns([
                pl.when(pl.col("close_time_ms") > t0)
                .then(pl.col("commitment") * 999.0)
                .otherwise(pl.col("commitment"))
                .alias("commitment"),
                pl.when(pl.col("close_time_ms") > t0)
                .then(pl.col("skew_general") * 999.0)
                .otherwise(pl.col("skew_general"))
                .alias("skew_general"),
                pl.when(pl.col("close_time_ms") > t0)
                .then(pl.col("skew_top20") * 999.0)
                .otherwise(pl.col("skew_top20"))
                .alias("skew_top20"),
            ])

            t2_5m_mut_path = os.path.join(tmpdir, "tier2_5m_mutated.parquet")
            tier2_5m_mutated.write_parquet(t2_5m_mut_path)

            # Run mutated alignment
            ingest_and_align(t1_path, t2_5m_mut_path, t2_8h_path, out_mutated)

            # Compare: rows at close_time_ms ≤ T₀ must be identical
            original = pl.read_parquet(out_original).filter(pl.col("close_time_ms") <= t0)
            mutated = pl.read_parquet(out_mutated).filter(pl.col("close_time_ms") <= t0)

            assert original.shape == mutated.shape, (
                f"Shape mismatch: {original.shape} vs {mutated.shape}"
            )

            # Bit-for-bit comparison across all columns
            from polars.testing import assert_frame_equal
            assert_frame_equal(original, mutated, check_exact=True)

    def test_tier2_8h_mutation_does_not_affect_past(self):
        """Mutating Tier 2 (8h) data after T₀ must not change any row ≤ T₀."""
        tier1 = _generate_synthetic_tier1(n_minutes=2000)
        tier2_5m = _generate_synthetic_tier2_5m(tier1)
        tier2_8h = _generate_synthetic_tier2_8h(tier1)

        ct_min = int(tier1["close_time_ms"].min())
        ct_max = int(tier1["close_time_ms"].max())
        t0 = ct_min + ((ct_max - ct_min) // 2)
        t0 = (t0 // 60_000) * 60_000

        with tempfile.TemporaryDirectory() as tmpdir:
            t1_path = os.path.join(tmpdir, "tier1.parquet")
            t2_5m_path = os.path.join(tmpdir, "tier2_5m.parquet")
            t2_8h_path = os.path.join(tmpdir, "tier2_8h.parquet")
            out_original = os.path.join(tmpdir, "master_original.parquet")
            out_mutated = os.path.join(tmpdir, "master_mutated.parquet")

            tier1.write_parquet(t1_path)
            tier2_5m.write_parquet(t2_5m_path)
            tier2_8h.write_parquet(t2_8h_path)

            ingest_and_align(t1_path, t2_5m_path, t2_8h_path, out_original)

            # Mutate tier2_8h: multiply crowding_penalty after T₀ by 999
            tier2_8h_mutated = tier2_8h.with_columns(
                pl.when(pl.col("close_time_ms") > t0)
                .then(pl.col("crowding_penalty") * 999.0)
                .otherwise(pl.col("crowding_penalty"))
                .alias("crowding_penalty")
            )

            t2_8h_mut_path = os.path.join(tmpdir, "tier2_8h_mutated.parquet")
            tier2_8h_mutated.write_parquet(t2_8h_mut_path)

            ingest_and_align(t1_path, t2_5m_path, t2_8h_mut_path, out_mutated)

            original = pl.read_parquet(out_original).filter(pl.col("close_time_ms") <= t0)
            mutated = pl.read_parquet(out_mutated).filter(pl.col("close_time_ms") <= t0)

            assert original.shape == mutated.shape
            from polars.testing import assert_frame_equal
            assert_frame_equal(original, mutated, check_exact=True)

    def test_both_tiers_mutated_simultaneously(self):
        """Mutating both Tier 2 sources after T₀ must not change any row ≤ T₀."""
        tier1 = _generate_synthetic_tier1(n_minutes=2000)
        tier2_5m = _generate_synthetic_tier2_5m(tier1)
        tier2_8h = _generate_synthetic_tier2_8h(tier1)

        ct_min = int(tier1["close_time_ms"].min())
        ct_max = int(tier1["close_time_ms"].max())
        t0 = ct_min + ((ct_max - ct_min) // 2)
        t0 = (t0 // 60_000) * 60_000

        with tempfile.TemporaryDirectory() as tmpdir:
            t1_path = os.path.join(tmpdir, "tier1.parquet")
            t2_5m_path = os.path.join(tmpdir, "tier2_5m.parquet")
            t2_8h_path = os.path.join(tmpdir, "tier2_8h.parquet")
            out_original = os.path.join(tmpdir, "master_original.parquet")
            out_mutated = os.path.join(tmpdir, "master_mutated.parquet")

            tier1.write_parquet(t1_path)
            tier2_5m.write_parquet(t2_5m_path)
            tier2_8h.write_parquet(t2_8h_path)

            ingest_and_align(t1_path, t2_5m_path, t2_8h_path, out_original)

            # Mutate both tier2 sources
            tier2_5m_mutated = tier2_5m.with_columns([
                pl.when(pl.col("close_time_ms") > t0)
                .then(pl.col("commitment") * -1.0)
                .otherwise(pl.col("commitment"))
                .alias("commitment"),
                pl.when(pl.col("close_time_ms") > t0)
                .then(pl.col("skew_general") + 100.0)
                .otherwise(pl.col("skew_general"))
                .alias("skew_general"),
                pl.when(pl.col("close_time_ms") > t0)
                .then(pl.col("skew_top20") + 100.0)
                .otherwise(pl.col("skew_top20"))
                .alias("skew_top20"),
            ])

            tier2_8h_mutated = tier2_8h.with_columns(
                pl.when(pl.col("close_time_ms") > t0)
                .then(pl.col("crowding_penalty") + 100.0)
                .otherwise(pl.col("crowding_penalty"))
                .alias("crowding_penalty")
            )

            t2_5m_mut_path = os.path.join(tmpdir, "tier2_5m_m.parquet")
            t2_8h_mut_path = os.path.join(tmpdir, "tier2_8h_m.parquet")
            tier2_5m_mutated.write_parquet(t2_5m_mut_path)
            tier2_8h_mutated.write_parquet(t2_8h_mut_path)

            ingest_and_align(t1_path, t2_5m_mut_path, t2_8h_mut_path, out_mutated)

            original = pl.read_parquet(out_original).filter(pl.col("close_time_ms") <= t0)
            mutated = pl.read_parquet(out_mutated).filter(pl.col("close_time_ms") <= t0)

            assert original.shape == mutated.shape
            assert_frame_equal(original, mutated, check_exact=True)

    def test_run_audit_end_to_end(self):
        """Verify Phase 1 audit runs on aligned master.parquet and serializes to JSON."""
        import json
        try:
            from compute.pipeline.audit import run_audit
        except ModuleNotFoundError:
            from pipeline.audit import run_audit

        tier1 = _generate_synthetic_tier1(n_minutes=2000)
        tier2_5m = _generate_synthetic_tier2_5m(tier1)
        tier2_8h = _generate_synthetic_tier2_8h(tier1)

        with tempfile.TemporaryDirectory() as tmpdir:
            t1_path = os.path.join(tmpdir, "tier1.parquet")
            t2_5m_path = os.path.join(tmpdir, "tier2_5m.parquet")
            t2_8h_path = os.path.join(tmpdir, "tier2_8h.parquet")
            out_path = os.path.join(tmpdir, "master.parquet")

            tier1.write_parquet(t1_path)
            tier2_5m.write_parquet(t2_5m_path)
            tier2_8h.write_parquet(t2_8h_path)

            ingest_and_align(t1_path, t2_5m_path, t2_8h_path, out_path)

            report = run_audit(out_path)

            # Assert key sections exist
            assert "gap_summary" in report
            assert "autocorrelation" in report
            assert "welch_psd" in report
            assert "tail_profile" in report
            assert "permutation_entropy" in report
            assert "diurnal_weekly_heatmap" in report
            assert "lag_scatter_r_t" in report
            assert "lag_scatter_abs_r_t" in report

            # Assert serializable to JSON
            dumped = json.dumps(report)
            assert len(dumped) > 0
            loaded = json.loads(dumped)
            assert loaded["gap_summary"]["total_rows"] == 2000
