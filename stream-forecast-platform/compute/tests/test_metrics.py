"""
Tests for pipeline/metrics.py and splits.py — Spec §4 verification gate.

Covers:
  • Non-crossing quantile repair
  • Hand-calculated pinball loss fixtures (exact values)
  • Embargo boundary assertion
  • Zero overlap between inner_train, inner_val, test for every fold
"""

from __future__ import annotations

import sys
import os
import math

import numpy as np
import pytest

# Ensure compute/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.metrics import (
    repair_quantile_crossings,
    pinball_loss,
    pinball_loss_mean,
    compute_pinball_metrics,
    skill_score,
    compute_calibration,
)
from pipeline.splits import generate_folds
from config import EMBARGO_ROWS, TEST_ROWS


# =====================================================================
#  Non-crossing quantile repair
# =====================================================================

class TestNonCrossingRepair:
    """Verify that repair_quantile_crossings sorts per row."""

    def test_already_sorted(self):
        """No change when quantiles are already in order."""
        q10 = np.array([1.0, 2.0, 3.0])
        q50 = np.array([2.0, 3.0, 4.0])
        q90 = np.array([3.0, 4.0, 5.0])
        r10, r50, r90 = repair_quantile_crossings(q10, q50, q90)
        np.testing.assert_array_equal(r10, q10)
        np.testing.assert_array_equal(r50, q50)
        np.testing.assert_array_equal(r90, q90)

    def test_fully_reversed(self):
        """Reversed quantiles should be sorted ascending."""
        q10 = np.array([5.0])
        q50 = np.array([3.0])
        q90 = np.array([1.0])
        r10, r50, r90 = repair_quantile_crossings(q10, q50, q90)
        assert r10[0] == 1.0
        assert r50[0] == 3.0
        assert r90[0] == 5.0

    def test_partial_crossing(self):
        """q50 < q10 but q90 is fine."""
        q10 = np.array([4.0])
        q50 = np.array([2.0])
        q90 = np.array([6.0])
        r10, r50, r90 = repair_quantile_crossings(q10, q50, q90)
        assert r10[0] == 2.0
        assert r50[0] == 4.0
        assert r90[0] == 6.0

    def test_all_equal(self):
        """Equal values stay equal."""
        q = np.array([3.0, 3.0])
        r10, r50, r90 = repair_quantile_crossings(q, q.copy(), q.copy())
        np.testing.assert_array_equal(r10, q)
        np.testing.assert_array_equal(r50, q)
        np.testing.assert_array_equal(r90, q)

    def test_mixed_rows(self):
        """Multiple rows with different crossing patterns."""
        q10 = np.array([1.0, 5.0, 3.0])
        q50 = np.array([2.0, 3.0, 1.0])
        q90 = np.array([3.0, 1.0, 2.0])
        r10, r50, r90 = repair_quantile_crossings(q10, q50, q90)
        # Row 0: already sorted
        assert r10[0] == 1.0 and r50[0] == 2.0 and r90[0] == 3.0
        # Row 1: fully reversed → 1, 3, 5
        assert r10[1] == 1.0 and r50[1] == 3.0 and r90[1] == 5.0
        # Row 2: 3, 1, 2 → 1, 2, 3
        assert r10[2] == 1.0 and r50[2] == 2.0 and r90[2] == 3.0


# =====================================================================
#  Pinball loss — hand-calculated fixtures
# =====================================================================

class TestPinballLoss:
    """Verify exact pinball loss values against hand-computed results."""

    def test_exact_alpha_050_overpredict(self):
        """α=0.50, y=1.0, q̂=1.5 → L = max(0.5*(-0.5), -0.5*(-0.5))
        = max(-0.25, 0.25) = 0.25"""
        y = np.array([1.0])
        q = np.array([1.5])
        loss = pinball_loss(y, q, 0.50)
        assert math.isclose(loss[0], 0.25, abs_tol=1e-15)

    def test_exact_alpha_050_underpredict(self):
        """α=0.50, y=2.0, q̂=1.0 → L = max(0.5*1.0, -0.5*1.0)
        = max(0.5, -0.5) = 0.5"""
        y = np.array([2.0])
        q = np.array([1.0])
        loss = pinball_loss(y, q, 0.50)
        assert math.isclose(loss[0], 0.5, abs_tol=1e-15)

    def test_exact_alpha_010_overpredict(self):
        """α=0.10, y=1.0, q̂=3.0 → diff=-2.0
        L = max(0.1*(-2), -0.9*(-2)) = max(-0.2, 1.8) = 1.8"""
        y = np.array([1.0])
        q = np.array([3.0])
        loss = pinball_loss(y, q, 0.10)
        assert math.isclose(loss[0], 1.8, abs_tol=1e-15)

    def test_exact_alpha_010_underpredict(self):
        """α=0.10, y=5.0, q̂=2.0 → diff=3.0
        L = max(0.1*3, -0.9*3) = max(0.3, -2.7) = 0.3"""
        y = np.array([5.0])
        q = np.array([2.0])
        loss = pinball_loss(y, q, 0.10)
        assert math.isclose(loss[0], 0.3, abs_tol=1e-15)

    def test_exact_alpha_090_overpredict(self):
        """α=0.90, y=1.0, q̂=4.0 → diff=-3.0
        L = max(0.9*(-3), -0.1*(-3)) = max(-2.7, 0.3) = 0.3"""
        y = np.array([1.0])
        q = np.array([4.0])
        loss = pinball_loss(y, q, 0.90)
        assert math.isclose(loss[0], 0.3, abs_tol=1e-15)

    def test_exact_alpha_090_underpredict(self):
        """α=0.90, y=5.0, q̂=2.0 → diff=3.0
        L = max(0.9*3, -0.1*3) = max(2.7, -0.3) = 2.7"""
        y = np.array([5.0])
        q = np.array([2.0])
        loss = pinball_loss(y, q, 0.90)
        assert math.isclose(loss[0], 2.7, abs_tol=1e-15)

    def test_perfect_prediction(self):
        """When y == q̂, pinball loss is exactly 0 for any alpha."""
        y = np.array([3.14])
        q = np.array([3.14])
        for alpha in [0.10, 0.50, 0.90]:
            loss = pinball_loss(y, q, alpha)
            assert math.isclose(loss[0], 0.0, abs_tol=1e-15)

    def test_pinball_mean(self):
        """Verify pinball_mean = (p10 + p50 + p90) / 3."""
        y  = np.array([1.0, 2.0, 3.0, 0.5])
        q10 = np.array([0.5, 1.5, 2.0, 1.0])
        q50 = np.array([1.0, 2.0, 3.0, 0.5])
        q90 = np.array([1.5, 2.5, 4.0, 0.0])

        metrics = compute_pinball_metrics(y, q10, q50, q90)
        expected_mean = (metrics["pinball_10"] + metrics["pinball_50"] + metrics["pinball_90"]) / 3
        assert math.isclose(metrics["pinball_mean"], expected_mean, abs_tol=1e-15)

    def test_vectorized_matches_elementwise(self):
        """Multi-element array matches element-by-element computation."""
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 100)
        q = rng.normal(0, 1, 100)
        alpha = 0.10

        vec_loss = pinball_loss(y, q, alpha)
        for i in range(len(y)):
            expected = max(alpha * (y[i] - q[i]), (alpha - 1) * (y[i] - q[i]))
            assert math.isclose(vec_loss[i], expected, abs_tol=1e-14)


# =====================================================================
#  Skill score
# =====================================================================

class TestSkillScore:

    def test_perfect_model(self):
        """Perfect model (0 loss) → skill = 1.0."""
        assert math.isclose(skill_score(0.0, 1.0), 1.0)

    def test_same_as_naive(self):
        """Same loss as naive → skill = 0.0."""
        assert math.isclose(skill_score(0.5, 0.5), 0.0)

    def test_worse_than_naive(self):
        """Worse than naive → skill < 0."""
        assert skill_score(2.0, 1.0) < 0.0
        assert math.isclose(skill_score(2.0, 1.0), -1.0)

    def test_zero_naive(self):
        """Zero naive baseline → returns 0 to avoid division by zero."""
        assert math.isclose(skill_score(0.5, 0.0), 0.0)


# =====================================================================
#  Calibration
# =====================================================================

class TestCalibration:

    def test_perfect_calibration(self):
        """With perfectly calibrated quantiles, coverage should match."""
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 100_000)
        q10 = np.full(100_000, np.quantile(y, 0.10))
        q90 = np.full(100_000, np.quantile(y, 0.90))

        cal = compute_calibration(y, q10, q90)
        assert abs(cal["cov_10"] - 0.10) < 0.01
        assert abs(cal["cov_90"] - 0.90) < 0.01
        assert abs(cal["cov_80"] - 0.80) < 0.02

    def test_all_above(self):
        """All actuals above q90 → cov_90 ≈ 0."""
        y = np.array([10.0, 11.0, 12.0])
        q10 = np.array([1.0, 1.0, 1.0])
        q90 = np.array([2.0, 2.0, 2.0])
        cal = compute_calibration(y, q10, q90)
        assert cal["cov_10"] == 0.0
        assert cal["cov_90"] == 0.0


# =====================================================================
#  Embargo boundary assertion
# =====================================================================

class TestEmbargoBoundary:
    """Spec §4 / Milestone 2 verification gate:
    min(test.close_time_ms) − max(train.close_time_ms) >= EMBARGO_ROWS × 60000
    """

    def _make_close_time_ms(self, n: int, start_ms: int = 0) -> np.ndarray:
        """Create a contiguous close_time_ms array with 1-min spacing."""
        return np.arange(start_ms, start_ms + n * 60_000, 60_000, dtype=np.int64)

    def test_embargo_gap_all_folds(self):
        """Every fold must have at least EMBARGO_ROWS × 60000 ms gap
        between train_end and test_start."""
        n_rows = 600_000  # ~416 days
        close_time_ms = self._make_close_time_ms(n_rows)
        folds = generate_folds(n_rows)

        assert len(folds) > 0, "Must generate at least one fold"

        for fold in folds:
            train_end_ms = close_time_ms[fold["train_end"] - 1]  # last train row
            test_start_ms = close_time_ms[fold["test_start"]]     # first test row

            gap_ms = test_start_ms - train_end_ms
            expected_gap_ms = EMBARGO_ROWS * 60_000

            assert gap_ms >= expected_gap_ms, (
                f"Fold {fold['fold_idx']}: embargo gap {gap_ms} ms "
                f"< required {expected_gap_ms} ms"
            )

    def test_embargo_rows_exact(self):
        """The embargo region spans exactly EMBARGO_ROWS rows."""
        n_rows = 600_000
        folds = generate_folds(n_rows)
        for fold in folds:
            embargo_len = fold["embargo_end"] - fold["embargo_start"]
            assert embargo_len == EMBARGO_ROWS, (
                f"Fold {fold['fold_idx']}: embargo length {embargo_len} != {EMBARGO_ROWS}"
            )


# =====================================================================
#  Zero overlap between inner_train, inner_val, test
# =====================================================================

class TestFoldNoOverlap:
    """Every fold must have zero overlap between inner_train, inner_val,
    and test ranges."""

    def test_no_overlap_all_folds(self):
        n_rows = 600_000
        folds = generate_folds(n_rows)

        assert len(folds) > 0

        for fold in folds:
            # Inner train: [train_start, inner_train_end)
            inner_train = set(range(fold["train_start"], fold["inner_train_end"]))
            # Inner val: [inner_val_start, inner_val_end)
            inner_val = set(range(fold["inner_val_start"], fold["inner_val_end"]))
            # Test: [test_start, test_end)
            test = set(range(fold["test_start"], fold["test_end"]))

            assert inner_train.isdisjoint(inner_val), (
                f"Fold {fold['fold_idx']}: inner_train overlaps inner_val"
            )
            assert inner_train.isdisjoint(test), (
                f"Fold {fold['fold_idx']}: inner_train overlaps test"
            )
            assert inner_val.isdisjoint(test), (
                f"Fold {fold['fold_idx']}: inner_val overlaps test"
            )

    def test_inner_embargo_excluded(self):
        """The inner embargo zone must not overlap inner_train or inner_val."""
        n_rows = 600_000
        folds = generate_folds(n_rows)

        for fold in folds:
            inner_embargo_start = fold["inner_train_end"]
            inner_embargo_end = fold["inner_val_start"]
            inner_embargo = set(range(inner_embargo_start, inner_embargo_end))

            inner_train = set(range(fold["train_start"], fold["inner_train_end"]))
            inner_val = set(range(fold["inner_val_start"], fold["inner_val_end"]))

            assert inner_embargo.isdisjoint(inner_train), (
                f"Fold {fold['fold_idx']}: inner_embargo overlaps inner_train"
            )
            assert inner_embargo.isdisjoint(inner_val), (
                f"Fold {fold['fold_idx']}: inner_embargo overlaps inner_val"
            )

    def test_test_rows_non_overlapping_across_folds(self):
        """Test blocks from different folds must not overlap
        (since STEP_ROWS == TEST_ROWS)."""
        n_rows = 600_000
        folds = generate_folds(n_rows)

        for i in range(len(folds)):
            for j in range(i + 1, len(folds)):
                test_i = set(range(folds[i]["test_start"], folds[i]["test_end"]))
                test_j = set(range(folds[j]["test_start"], folds[j]["test_end"]))
                assert test_i.isdisjoint(test_j), (
                    f"Fold {folds[i]['fold_idx']} and {folds[j]['fold_idx']} "
                    f"have overlapping test blocks"
                )

    def test_fold_geometry_sizes(self):
        """Verify each sub-block has the expected number of rows."""
        n_rows = 600_000
        folds = generate_folds(n_rows)

        from config import INNER_VAL_ROWS as IV, TRAIN_ROWS as TR

        for fold in folds:
            # Inner val size
            iv_len = fold["inner_val_end"] - fold["inner_val_start"]
            assert iv_len == IV, f"Fold {fold['fold_idx']}: inner_val len {iv_len} != {IV}"

            # Test size
            test_len = fold["test_end"] - fold["test_start"]
            assert test_len == TEST_ROWS, f"Fold {fold['fold_idx']}: test len {test_len} != {TEST_ROWS}"

            # Total train (outer) size
            train_len = fold["train_end"] - fold["train_start"]
            assert train_len == TR, f"Fold {fold['fold_idx']}: train len {train_len} != {TR}"

            # Inner train + inner_embargo + inner_val == train
            inner_train_len = fold["inner_train_end"] - fold["train_start"]
            inner_embargo_len = fold["inner_val_start"] - fold["inner_train_end"]
            assert inner_train_len + inner_embargo_len + iv_len == TR, (
                f"Fold {fold['fold_idx']}: inner split sizes don't sum to train"
            )
            assert inner_embargo_len == EMBARGO_ROWS, (
                f"Fold {fold['fold_idx']}: inner embargo {inner_embargo_len} != {EMBARGO_ROWS}"
            )
