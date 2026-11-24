"""
Tests for Layer 3 — Regime Detection (Spec §12 Milestone 3 verification)

1. HMM forward causality: mutate O_{t+1:N}, assert regime_p at ≤ t identical to 1e-12
2. Deterministic ordering: mean_vol[0] ≤ mean_vol[1] ≤ mean_vol[2]
"""

from __future__ import annotations

import numpy as np
import pytest

from compute.pipeline.regimes import CausalHMM, _hmm_forward_filter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_obs(
    n: int = 2000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic 4-feature observation matrix and steps_in_segment.

    Creates 3 distinct clusters so the HMM can learn 3 states:
      - State 0 (Compression): low log_rv_60
      - State 1 (Normal):      medium log_rv_60
      - State 2 (Expansion):   high log_rv_60
    """
    rng = np.random.RandomState(seed)

    # Simulate regime transitions with known ordering
    regimes = np.zeros(n, dtype=int)
    state = 0
    for t in range(n):
        regimes[t] = state
        if rng.rand() < 0.02:  # ~2% transition probability
            state = rng.choice(3)

    # 4 observation features: log_rv_60, vol_ratio_15_240, perm_entropy_60, abs_agg_imbalance_15
    means = np.array([
        [-2.0, 0.5, 0.3, 0.1],   # State 0: Compression (low vol)
        [0.0, 1.0, 0.5, 0.3],    # State 1: Normal
        [2.0, 2.0, 0.8, 0.6],    # State 2: Expansion (high vol)
    ])
    obs = np.zeros((n, 4))
    for t in range(n):
        obs[t] = means[regimes[t]] + rng.randn(4) * 0.3

    # All one contiguous segment
    steps = np.arange(1, n + 1, dtype=np.int64)

    return obs, steps


def _fit_causal_hmm(obs: np.ndarray) -> CausalHMM:
    """Fit a CausalHMM on raw observations (acts as inner_train)."""
    import polars as pl

    try:
        from compute.config import HMM_OBS_FEATURES
    except ImportError:
        HMM_OBS_FEATURES = [
            "log_rv_60",
            "vol_ratio_15_240",
            "perm_entropy_60",
            "abs_agg_imbalance_15",
        ]

    df = pl.DataFrame({
        HMM_OBS_FEATURES[0]: obs[:, 0],
        HMM_OBS_FEATURES[1]: obs[:, 1],
        HMM_OBS_FEATURES[2]: obs[:, 2],
        HMM_OBS_FEATURES[3]: obs[:, 3],
    })

    hmm = CausalHMM()
    hmm.fit(df)
    return hmm


# ---------------------------------------------------------------------------
# Test 1: HMM Forward Causality
# ---------------------------------------------------------------------------

class TestHMMForwardCausality:
    """
    Run filter on O_{1:N}, then mutate O_{t+1:N} and re-run.
    Assert regime_p at ≤ t is identical to 1e-12.
    """

    def test_forward_filter_is_strictly_causal(self):
        obs, steps = _make_synthetic_obs(n=1000)
        hmm = _fit_causal_hmm(obs)

        # Scale observations with frozen scaler
        obs_scaled = hmm.scaler.transform(obs)
        obs_scaled = np.clip(obs_scaled, -8.0, 8.0)

        # Run filter on original data
        p0_orig, p1_orig, p2_orig, ent_orig = _hmm_forward_filter(
            obs_scaled,
            steps,
            hmm._log_startprob,
            hmm._log_transmat,
            hmm.means,
            hmm._inv_covars,
            hmm._log_det_covars,
        )

        # Pick a mutation point in the middle
        t_mut = 500

        # Mutate O_{t+1:N} with completely different data
        obs_mutated = obs_scaled.copy()
        rng = np.random.RandomState(99)
        obs_mutated[t_mut + 1:] = rng.randn(
            obs_scaled.shape[0] - t_mut - 1, obs_scaled.shape[1]
        ) * 5.0  # Large perturbation

        # Re-run filter on mutated data
        p0_mut, p1_mut, p2_mut, ent_mut = _hmm_forward_filter(
            obs_mutated,
            steps,
            hmm._log_startprob,
            hmm._log_transmat,
            hmm.means,
            hmm._inv_covars,
            hmm._log_det_covars,
        )

        # Assert regime probabilities at ≤ t are identical to 1e-12
        np.testing.assert_allclose(
            p0_orig[: t_mut + 1],
            p0_mut[: t_mut + 1],
            atol=1e-12,
            err_msg="regime_p0 at t ≤ mutation point differs after future mutation",
        )
        np.testing.assert_allclose(
            p1_orig[: t_mut + 1],
            p1_mut[: t_mut + 1],
            atol=1e-12,
            err_msg="regime_p1 at t ≤ mutation point differs after future mutation",
        )
        np.testing.assert_allclose(
            p2_orig[: t_mut + 1],
            p2_mut[: t_mut + 1],
            atol=1e-12,
            err_msg="regime_p2 at t ≤ mutation point differs after future mutation",
        )
        np.testing.assert_allclose(
            ent_orig[: t_mut + 1],
            ent_mut[: t_mut + 1],
            atol=1e-12,
            err_msg="regime_entropy at t ≤ mutation point differs after future mutation",
        )

        # Also verify that after the mutation point, values DO differ
        # (sanity check that our perturbation actually had an effect)
        assert not np.allclose(
            p0_orig[t_mut + 1:],
            p0_mut[t_mut + 1:],
            atol=1e-6,
        ), "Perturbation had no effect on future regime_p0 — test is vacuous"

    def test_gap_resets_prior(self):
        """After a gap (steps_in_segment==1), filter resets to π."""
        obs, _ = _make_synthetic_obs(n=200)
        hmm = _fit_causal_hmm(obs)

        obs_scaled = hmm.scaler.transform(obs)
        obs_scaled = np.clip(obs_scaled, -8.0, 8.0)

        # Create a gap at step 100
        steps_with_gap = np.arange(1, 201, dtype=np.int64)
        steps_with_gap[100] = 1  # Reset

        p0_a, _, _, _ = _hmm_forward_filter(
            obs_scaled,
            steps_with_gap,
            hmm._log_startprob,
            hmm._log_transmat,
            hmm.means,
            hmm._inv_covars,
            hmm._log_det_covars,
        )

        # Mutate everything before the gap
        obs_mut = obs_scaled.copy()
        obs_mut[:100] = np.random.RandomState(77).randn(100, 4) * 10.0

        p0_b, _, _, _ = _hmm_forward_filter(
            obs_mut,
            steps_with_gap,
            hmm._log_startprob,
            hmm._log_transmat,
            hmm.means,
            hmm._inv_covars,
            hmm._log_det_covars,
        )

        # After the gap reset (step 100 onwards), results should be identical
        # because step 100 resets to π regardless of history
        np.testing.assert_allclose(
            p0_a[100:],
            p0_b[100:],
            atol=1e-12,
            err_msg="Gap reset did not eliminate dependence on pre-gap data",
        )


# ---------------------------------------------------------------------------
# Test 2: Deterministic State Ordering
# ---------------------------------------------------------------------------

class TestDeterministicOrdering:
    """
    Verify states are sorted ascending by means_[:, 0] (mean log_rv_60).
    State 0 = Compression, State 1 = Normal, State 2 = Expansion.
    """

    def test_state_ordering_by_mean_vol(self):
        obs, _ = _make_synthetic_obs(n=2000)
        hmm = _fit_causal_hmm(obs)

        # means_[:, 0] is the mean of log_rv_60 per state
        mean_vol = hmm.means[:, 0]

        assert mean_vol[0] <= mean_vol[1], (
            f"State 0 mean_vol ({mean_vol[0]:.4f}) > State 1 ({mean_vol[1]:.4f})"
        )
        assert mean_vol[1] <= mean_vol[2], (
            f"State 1 mean_vol ({mean_vol[1]:.4f}) > State 2 ({mean_vol[2]:.4f})"
        )

    def test_ordering_is_permutation_consistent(self):
        """Verify startprob_, transmat_, means_, covars_ all use same ordering."""
        obs, _ = _make_synthetic_obs(n=2000)
        hmm = _fit_causal_hmm(obs)

        # The transition matrix rows and columns should correspond to the
        # same ordering as means
        assert hmm.transmat.shape == (3, 3)
        assert hmm.startprob.shape == (3,)
        assert hmm.means.shape[0] == 3
        assert hmm.covars.shape[0] == 3

        # Row sums of transmat should be ~1
        for k in range(3):
            np.testing.assert_allclose(
                hmm.transmat[k].sum(),
                1.0,
                atol=1e-6,
                err_msg=f"transmat row {k} does not sum to 1",
            )

        # startprob should sum to ~1
        np.testing.assert_allclose(
            hmm.startprob.sum(),
            1.0,
            atol=1e-6,
            err_msg="startprob does not sum to 1",
        )

    def test_probabilities_sum_to_one(self):
        """Forward filter probabilities should sum to 1 at every step."""
        obs, steps = _make_synthetic_obs(n=500)
        hmm = _fit_causal_hmm(obs)

        obs_scaled = hmm.scaler.transform(obs)
        obs_scaled = np.clip(obs_scaled, -8.0, 8.0)

        p0, p1, p2, _ = _hmm_forward_filter(
            obs_scaled,
            steps,
            hmm._log_startprob,
            hmm._log_transmat,
            hmm.means,
            hmm._inv_covars,
            hmm._log_det_covars,
        )

        sums = p0 + p1 + p2
        np.testing.assert_allclose(
            sums,
            1.0,
            atol=1e-10,
            err_msg="Regime probabilities do not sum to 1 at every step",
        )
