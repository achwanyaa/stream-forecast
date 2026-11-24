"""
tests/test_gaps.py — Gap Masking Tests (Spec §12 Gate).

Proves:
  1. is_gap == True inside injected gaps.
  2. segment_id increments after a gap.
  3. steps_in_segment resets to 1 after a gap.
  4. Rolling windows / targets never bridge across gaps (output null).
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import polars as pl
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
# Synthetic data with injected gap
# ---------------------------------------------------------------------------

def _generate_tier1_with_gap(
    n_minutes: int = 500,
    gap_start_offset: int = 200,
    gap_length: int = 12,
    seed: int = SEED,
) -> pl.DataFrame:
    """
    Generate synthetic Tier 1 data with a 12-minute gap injected.

    The gap is created by removing rows in the middle of the series,
    so when the grid is constructed, those timestamps will be missing
    from the left-join → x_close will be null → is_gap == True.
    """
    rng = np.random.RandomState(seed)
    base_time = 1_700_000_000_000

    close_times = np.arange(
        base_time + 60_000,
        base_time + 60_000 + n_minutes * 60_000,
        60_000,
        dtype=np.int64,
    )
    open_times = close_times - 60_000

    # Random walk prices
    log_returns = rng.normal(0, 0.001, n_minutes)
    log_prices = np.cumsum(log_returns) + np.log(100.0)
    closes = np.exp(log_prices)
    opens = closes * np.exp(rng.normal(0, 0.0005, n_minutes))
    highs = np.maximum(opens, closes) * (1 + rng.uniform(0, 0.002, n_minutes))
    lows = np.minimum(opens, closes) * (1 - rng.uniform(0, 0.002, n_minutes))

    df = pl.DataFrame({
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

    # Remove rows in the gap range to simulate a data outage
    gap_start_idx = gap_start_offset
    gap_end_idx = gap_start_offset + gap_length

    # Get the close_time_ms values that will be missing
    gap_close_times = close_times[gap_start_idx:gap_end_idx]

    # Remove the gap rows from tier1
    df = df.filter(~pl.col("close_time_ms").is_in(gap_close_times.tolist()))

    return df, gap_close_times


def _generate_tier2_5m_simple(tier1: pl.DataFrame, seed: int = SEED) -> pl.DataFrame:
    rng = np.random.RandomState(seed + 10)
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


def _generate_tier2_8h_simple(tier1: pl.DataFrame, seed: int = SEED) -> pl.DataFrame:
    rng = np.random.RandomState(seed + 20)
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
# Tests
# ---------------------------------------------------------------------------

class TestGapMasking:
    """Verify gap masking per §2 and §12."""

    @pytest.fixture(autouse=True)
    def setup_data(self, tmp_path):
        """Create aligned master.parquet with a 12-minute gap injected."""
        self.gap_length = 12
        self.gap_start_offset = 200

        tier1, self.gap_close_times = _generate_tier1_with_gap(
            n_minutes=500,
            gap_start_offset=self.gap_start_offset,
            gap_length=self.gap_length,
        )

        tier2_5m = _generate_tier2_5m_simple(tier1)
        tier2_8h = _generate_tier2_8h_simple(tier1)

        t1_path = str(tmp_path / "tier1.parquet")
        t2_5m_path = str(tmp_path / "tier2_5m.parquet")
        t2_8h_path = str(tmp_path / "tier2_8h.parquet")
        self.out_path = str(tmp_path / "master.parquet")

        tier1.write_parquet(t1_path)
        tier2_5m.write_parquet(t2_5m_path)
        tier2_8h.write_parquet(t2_8h_path)

        self.master = ingest_and_align(t1_path, t2_5m_path, t2_8h_path, self.out_path)

    def test_is_gap_true_in_gap(self):
        """is_gap must be True for all timestamps inside the injected gap."""
        gap_rows = self.master.filter(
            pl.col("close_time_ms").is_in(self.gap_close_times.tolist())
        )

        assert gap_rows.height == self.gap_length, (
            f"Expected {self.gap_length} gap rows, found {gap_rows.height}"
        )
        assert gap_rows["is_gap"].all(), (
            "Not all gap rows have is_gap == True"
        )

    def test_is_gap_false_outside_gap(self):
        """is_gap must be False for rows outside the gap (with valid prices)."""
        non_gap_rows = self.master.filter(
            ~pl.col("close_time_ms").is_in(self.gap_close_times.tolist())
            & pl.col("x_close").is_not_null()
            & (pl.col("x_close") > 0)
        )
        assert (~non_gap_rows["is_gap"]).all(), (
            "Some non-gap rows with valid x_close have is_gap == True"
        )

    def test_segment_id_negative_in_gap(self):
        """Gap rows must have segment_id == -1."""
        gap_rows = self.master.filter(
            pl.col("close_time_ms").is_in(self.gap_close_times.tolist())
        )
        assert (gap_rows["segment_id"] == -1).all(), (
            "Not all gap rows have segment_id == -1"
        )

    def test_segment_id_increments_after_gap(self):
        """
        The segment_id must increment after a gap: the segment after the gap
        must have a higher segment_id than the segment before the gap.
        """
        gap_start_ct = int(self.gap_close_times[0])
        gap_end_ct = int(self.gap_close_times[-1])

        # Segment before gap
        before_gap = self.master.filter(
            (pl.col("close_time_ms") < gap_start_ct)
            & (~pl.col("is_gap"))
        )
        seg_before = before_gap["segment_id"][-1]

        # Segment after gap
        after_gap = self.master.filter(
            (pl.col("close_time_ms") > gap_end_ct)
            & (~pl.col("is_gap"))
        )
        seg_after = after_gap["segment_id"][0]

        assert seg_after > seg_before, (
            f"segment_id did not increment after gap: before={seg_before}, after={seg_after}"
        )

    def test_steps_in_segment_zero_in_gap(self):
        """steps_in_segment must be 0 inside gaps."""
        gap_rows = self.master.filter(
            pl.col("close_time_ms").is_in(self.gap_close_times.tolist())
        )
        assert (gap_rows["steps_in_segment"] == 0).all(), (
            "steps_in_segment is not 0 for all gap rows"
        )

    def test_steps_in_segment_resets_after_gap(self):
        """steps_in_segment must restart from 1 at the first valid row after the gap."""
        gap_end_ct = int(self.gap_close_times[-1])

        after_gap = self.master.filter(
            (pl.col("close_time_ms") > gap_end_ct)
            & (~pl.col("is_gap"))
        ).sort("close_time_ms")

        first_after = after_gap["steps_in_segment"][0]
        assert first_after == 1, (
            f"steps_in_segment after gap should start at 1, got {first_after}"
        )

    def test_rolling_window_nulls_across_gap(self):
        """
        A rolling window of length W must output null wherever steps_in_segment < W.
        This ensures rolling computations never bridge across gaps.

        We simulate a rolling mean with W=5 and verify that rows near/inside the gap
        produce null values.
        """
        W = 5

        # Compute a rolling mean on x_close (W=5) respecting the gap rule
        df = self.master.with_columns(
            pl.when(pl.col("steps_in_segment") >= W)
            .then(
                pl.col("x_close")
                .rolling_mean(window_size=W)
            )
            .otherwise(pl.lit(None))
            .alias("rolling_mean_5")
        )

        # All rows where steps_in_segment < W must have null rolling_mean_5
        insufficient = df.filter(pl.col("steps_in_segment") < W)
        assert insufficient["rolling_mean_5"].is_null().all(), (
            "Rolling window output is not null for rows with steps_in_segment < W"
        )

        # Gap rows specifically must have null
        gap_rows = df.filter(
            pl.col("close_time_ms").is_in(self.gap_close_times.tolist())
        )
        assert gap_rows["rolling_mean_5"].is_null().all(), (
            "Rolling window output is not null inside gap"
        )

    def test_target_does_not_bridge_gap(self):
        """
        A forward target (e.g., log-delta H=5) must be null if the target timestamp
        falls in a different segment than the current timestamp.
        This prevents training on targets that cross maintenance gaps.
        """
        df = self.master.sort("close_time_ms")

        # Simulate: y_{t,5} = ln(x_close_{t+5} / x_close_t)
        # Must be null if segment_id[t+5] != segment_id[t]
        H = 5
        df = df.with_columns([
            pl.col("x_close").shift(-H).alias("x_close_future"),
            pl.col("segment_id").shift(-H).alias("seg_future"),
        ])

        df = df.with_columns(
            pl.when(
                pl.col("segment_id").eq(pl.col("seg_future"))
                & pl.col("x_close").is_not_null()
                & pl.col("x_close_future").is_not_null()
                & (pl.col("x_close") > 0)
                & (pl.col("x_close_future") > 0)
            )
            .then((pl.col("x_close_future") / pl.col("x_close")).log())
            .otherwise(pl.lit(None))
            .alias("y_t_5")
        )

        # Rows just before the gap: their target should be null because
        # the target reaches into or past the gap
        gap_start_ct = int(self.gap_close_times[0])

        # Rows within H steps before the gap start
        near_gap = df.filter(
            (pl.col("close_time_ms") >= gap_start_ct - H * 60_000)
            & (pl.col("close_time_ms") < gap_start_ct)
        )

        # At least some of these should have null targets
        # (because their forward window crosses into the gap)
        null_targets = near_gap.filter(pl.col("y_t_5").is_null())
        assert null_targets.height > 0, (
            "No null targets found near gap boundary — targets may bridge the gap"
        )

        # Gap rows themselves must have null targets
        gap_targets = df.filter(
            pl.col("close_time_ms").is_in(self.gap_close_times.tolist())
        )
        assert gap_targets["y_t_5"].is_null().all(), (
            "Target is not null inside gap rows"
        )
