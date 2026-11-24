"""
Layer 0: Walk-forward fold generator, embargo logic, and noise control.

Spec §4 — Fold Geometry & Noise Control Group.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Lazy import so the module can be parsed even before config.py is created.
# At runtime the caller must have compute/ on sys.path.
# ---------------------------------------------------------------------------
from config import (
    WARMUP_ROWS,
    TRAIN_ROWS,
    INNER_VAL_ROWS,
    EMBARGO_ROWS,
    TEST_ROWS,
    STEP_ROWS,
    SEED,
    TRAIN_WINDOW_MODES,
)


# =========================================================================
#  Fold Geometry
# =========================================================================

def generate_folds(
    n_rows: int,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return a list of fold-spec dicts describing the walk-forward split.

    Parameters
    ----------
    n_rows : int
        Total number of rows in the master dataset.
    config : dict, optional
        Override defaults.  Recognized keys:
        - ``train_window_mode`` – one of ``TRAIN_WINDOW_MODES`` keys.
        - ``train_rows``, ``inner_val_rows``, ``embargo_rows``,
          ``test_rows``, ``step_rows``, ``warmup_rows`` for direct override.

    Returns
    -------
    list[dict]
        Each dict has keys:
        ``fold_idx``, ``train_start``, ``train_end``,
        ``inner_train_end``, ``inner_val_start``, ``inner_val_end``,
        ``embargo_start``, ``embargo_end``,
        ``test_start``, ``test_end``.
    """
    cfg = config or {}

    # Resolve training window
    train_rows = cfg.get("train_rows", TRAIN_ROWS)
    if "train_window_mode" in cfg:
        train_rows = TRAIN_WINDOW_MODES[cfg["train_window_mode"]]

    inner_val_rows = cfg.get("inner_val_rows", INNER_VAL_ROWS)
    embargo_rows   = cfg.get("embargo_rows",   EMBARGO_ROWS)
    test_rows      = cfg.get("test_rows",      TEST_ROWS)
    step_rows      = cfg.get("step_rows",      STEP_ROWS)
    warmup_rows    = cfg.get("warmup_rows",     WARMUP_ROWS)

    folds: list[dict[str, Any]] = []
    fold_idx = 0

    # First train block starts after warmup
    train_start = warmup_rows

    while True:
        train_end = train_start + train_rows

        # Outer embargo & test
        embargo_start = train_end
        embargo_end   = train_end + embargo_rows
        test_start    = embargo_end
        test_end      = test_start + test_rows

        if test_end > n_rows:
            break  # not enough data for this fold

        # Inner split (carved from END of train)
        inner_val_end   = train_end
        inner_val_start = train_end - inner_val_rows
        inner_embargo_end   = inner_val_start
        inner_embargo_start = inner_val_start - embargo_rows
        inner_train_end     = inner_embargo_start  # = train_end - inner_val_rows - embargo_rows

        folds.append(
            {
                "fold_idx":        fold_idx,
                "train_start":     train_start,
                "train_end":       train_end,
                "inner_train_end": inner_train_end,
                "inner_val_start": inner_val_start,
                "inner_val_end":   inner_val_end,
                "embargo_start":   embargo_start,
                "embargo_end":     embargo_end,
                "test_start":      test_start,
                "test_end":        test_end,
            }
        )

        fold_idx += 1
        train_start += step_rows  # slide by STEP_ROWS (non-overlapping tests)

    return folds


# =========================================================================
#  Noise Control Group
# =========================================================================

def generate_noise_control(
    master_df: pl.DataFrame,
    seed: int = SEED,
) -> pl.DataFrame:
    """Create a noise-control copy of *master_df*.

    Strategy (Spec §4 Noise Control):
    1. Compute 1-min log-returns ``r_t = ln(x_close[t] / x_close[t-1])``.
    2. Within each **contiguous valid segment** (``segment_id >= 0``),
       randomly permute the log-returns (preserving the exact marginal
       fat-tailed distribution while destroying temporal autocorrelation,
       volatility clustering, and Tier-2 alignment).
    3. Reconstruct synthetic ``x_close`` from the first valid price in
       each segment and the permuted returns.

    Parameters
    ----------
    master_df : pl.DataFrame
        Must contain ``x_close`` (Float64) and ``segment_id`` (Int32).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pl.DataFrame
        A copy of *master_df* with ``x_close`` replaced by the synthetic
        series.  All other columns are left untouched.
    """
    rng = np.random.default_rng(seed)

    x_close = np.array(master_df["x_close"], dtype=np.float64)
    segment_ids = np.array(master_df["segment_id"], dtype=np.int32)

    # Identify unique valid segments (segment_id >= 0)
    unique_segments = np.unique(segment_ids)
    unique_segments = unique_segments[unique_segments >= 0]

    for seg_id in unique_segments:
        mask = segment_ids == seg_id
        indices = np.where(mask)[0]

        if len(indices) < 2:
            continue

        seg_prices = x_close[indices]

        # Compute log-returns within segment
        log_returns = np.log(seg_prices[1:] / seg_prices[:-1])

        # Permute log-returns within this segment
        rng.shuffle(log_returns)

        # Reconstruct x_close from first price + permuted returns
        synthetic = np.empty_like(seg_prices)
        synthetic[0] = seg_prices[0]
        synthetic[1:] = seg_prices[0] * np.exp(np.cumsum(log_returns))

        x_close[indices] = synthetic

    return master_df.with_columns(pl.Series("x_close", x_close))
