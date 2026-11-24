"""
pipeline/ingest.py — Spec §2: Data Ingestion, Zero-Leakage Alignment & Gap Masking.

Implements the exact alignment sequence from SYSTEM_SPEC §2 using Polars only.
All joins are strictly backward (strategy="backward") to enforce zero lookahead.
"""

from __future__ import annotations

import polars as pl


def ingest_and_align(
    tier1_path: str,
    tier2_5m_path: str,
    tier2_8h_path: str,
    output_path: str,
) -> pl.DataFrame:
    """
    Build a gap-masked, multi-rate aligned master.parquet from raw tier files.

    Parameters
    ----------
    tier1_path : path to tier1_1m.parquet  (1-minute OHLCV bars)
    tier2_5m_path : path to tier2_5m.parquet (5-minute crowd metrics)
    tier2_8h_path : path to tier2_8h.parquet (8-hour crowd metrics)
    output_path : destination for master.parquet (ZSTD compressed)

    Returns
    -------
    The aligned Polars DataFrame (also persisted to *output_path*).
    """

    # ------------------------------------------------------------------
    # 0. Load raw parquet files
    # ------------------------------------------------------------------
    tier1 = pl.read_parquet(tier1_path)
    tier2_5m = pl.read_parquet(tier2_5m_path)
    tier2_8h = pl.read_parquet(tier2_8h_path)

    # ------------------------------------------------------------------
    # 1. Construct Uniform 1-Minute Grid
    #    Contiguous close_time_ms range: min → max with step 60_000
    # ------------------------------------------------------------------
    ct_min = tier1["close_time_ms"].min()
    ct_max = tier1["close_time_ms"].max()

    grid = pl.DataFrame(
        {
            "close_time_ms": pl.arange(
                ct_min, ct_max + 1, step=60_000, eager=True, dtype=pl.Int64
            )
        }
    )

    # ------------------------------------------------------------------
    # 2. Left-Join Tier 1 onto the uniform grid on close_time_ms
    # ------------------------------------------------------------------
    master = grid.join(tier1, on="close_time_ms", how="left")

    # ------------------------------------------------------------------
    # 3. Compute Gap Flags
    # ------------------------------------------------------------------
    #   is_gap: x_close is null or x_close <= 0
    master = master.with_columns(
        (pl.col("x_close").is_null() | (pl.col("x_close") <= 0)).alias("is_gap")
    )

    #   segment_id:
    #     - Boundary marker: is_gap != is_gap.shift(1)  (first row is a boundary)
    #     - Raw segment id = cumsum of boundary markers (0-based)
    #     - Gap rows → -1;  valid stretches → incrementing IDs 0, 1, 2, …
    master = master.with_columns(
        (pl.col("is_gap") != pl.col("is_gap").shift(1))
        .fill_null(True)              # first row is always a boundary
        .cum_sum()
        .sub(1)                        # make 0-based
        .alias("_raw_seg")
    )

    # Build a mapping: for valid (non-gap) raw segments, assign incrementing IDs.
    # Gap segments get -1.
    seg_map = (
        master
        .select("_raw_seg", "is_gap")
        .unique(subset=["_raw_seg"])
        .sort("_raw_seg")
    )
    # For valid segments (is_gap == False), assign dense rank 0,1,2,…
    seg_map = seg_map.with_columns(
        pl.when(~pl.col("is_gap"))
        .then(
            (~pl.col("is_gap")).cum_sum() - 1
        )
        .otherwise(pl.lit(-1, dtype=pl.Int32))
        .alias("segment_id")
    )
    master = master.join(
        seg_map.select("_raw_seg", "segment_id"),
        on="_raw_seg",
        how="left",
    ).drop("_raw_seg")

    master = master.with_columns(pl.col("segment_id").cast(pl.Int32))

    #   steps_in_segment: 1-based row counter within each valid segment_id; 0 when is_gap.
    master = master.with_columns(
        pl.when(~pl.col("is_gap"))
        .then(
            pl.lit(1)
            .cum_sum()
            .over("segment_id")
        )
        .otherwise(pl.lit(0, dtype=pl.UInt32))
        .cast(pl.Int32)
        .alias("steps_in_segment")
    )

    # ------------------------------------------------------------------
    # 4. Strict Backward Multi-Rate Join — Tier 2 (5m)
    # ------------------------------------------------------------------
    # Sort both sides ascending by close_time_ms (required by join_asof).
    master = master.sort("close_time_ms")

    tier2_5m = tier2_5m.sort("close_time_ms")
    # Rename close_time_ms in tier2 to avoid collision after join
    tier2_5m_join = tier2_5m.rename({"close_time_ms": "t2_5m_close_time_ms"})
    # Also drop open_time_ms if present to avoid clash
    if "open_time_ms" in tier2_5m_join.columns:
        tier2_5m_join = tier2_5m_join.drop("open_time_ms")

    master = master.join_asof(
        tier2_5m_join,
        left_on="close_time_ms",
        right_on="t2_5m_close_time_ms",
        strategy="backward",
    )

    # Staleness tracking for tier2_5m
    master = master.with_columns(
        ((pl.col("close_time_ms") - pl.col("t2_5m_close_time_ms")) / 60_000.0).alias(
            "t2_5m_staleness_min"
        )
    )

    # Null-out tier2_5m columns when staleness exceeds 15.0 minutes
    t2_5m_value_cols = [c for c in tier2_5m.columns if c not in ("open_time_ms", "close_time_ms")]
    for col_name in t2_5m_value_cols:
        master = master.with_columns(
            pl.when(pl.col("t2_5m_staleness_min") > 15.0)
            .then(pl.lit(None))
            .otherwise(pl.col(col_name))
            .alias(col_name)
        )

    # ------------------------------------------------------------------
    # 5. Strict Backward Multi-Rate Join — Tier 2 (8h)
    # ------------------------------------------------------------------
    tier2_8h = tier2_8h.sort("close_time_ms")
    tier2_8h_join = tier2_8h.rename({"close_time_ms": "t2_8h_close_time_ms"})
    if "open_time_ms" in tier2_8h_join.columns:
        tier2_8h_join = tier2_8h_join.drop("open_time_ms")

    master = master.join_asof(
        tier2_8h_join,
        left_on="close_time_ms",
        right_on="t2_8h_close_time_ms",
        strategy="backward",
    )

    # Staleness tracking for tier2_8h
    master = master.with_columns(
        ((pl.col("close_time_ms") - pl.col("t2_8h_close_time_ms")) / 60_000.0).alias(
            "t2_8h_staleness_min"
        )
    )

    # Null-out crowding_penalty when staleness exceeds 960.0 minutes (16 hours)
    master = master.with_columns(
        pl.when(pl.col("t2_8h_staleness_min") > 960.0)
        .then(pl.lit(None))
        .otherwise(pl.col("crowding_penalty"))
        .alias("crowding_penalty")
    )

    # ------------------------------------------------------------------
    # 6. Save Master Dataset — ZSTD compression
    # ------------------------------------------------------------------
    master.write_parquet(output_path, compression="zstd")

    return master
