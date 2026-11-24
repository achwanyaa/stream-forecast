"""
Layer 3 — Strictly Causal Regime & Change-Point Estimator
(Spec §7: Forward-Only HMM, Rolling Memory, BOCPD)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl
from numba import njit
from scipy.special import gammaln
from sklearn.preprocessing import RobustScaler
from hmmlearn.hmm import GaussianHMM

try:
    from compute.config import SEED, HMM_OBS_FEATURES
except ImportError:
    SEED = 42
    HMM_OBS_FEATURES = [
        "log_rv_60",
        "vol_ratio_15_240",
        "perm_entropy_60",
        "abs_agg_imbalance_15",
    ]

# ---------------------------------------------------------------------------
# Component 3A — Forward-Only 3-State Gaussian HMM
# ---------------------------------------------------------------------------


class CausalHMM:
    """
    Fit a 3-state Gaussian HMM on inner_train, deterministically order
    states by mean log_rv_60, and expose a strictly causal Numba forward
    filter for scoring unseen data.
    """

    CLIP_BOUND = 8.0

    def __init__(self) -> None:
        self.scaler: RobustScaler | None = None
        self.hmm: GaussianHMM | None = None
        # After deterministic reorder
        self.startprob: np.ndarray | None = None   # (3,)
        self.transmat: np.ndarray | None = None     # (3,3)
        self.means: np.ndarray | None = None        # (3,4)
        self.covars: np.ndarray | None = None       # (3,4,4)
        # Pre-computed for Numba filter
        self._log_startprob: np.ndarray | None = None
        self._log_transmat: np.ndarray | None = None
        self._inv_covars: np.ndarray | None = None
        self._log_det_covars: np.ndarray | None = None

    # ---- Fit (inner_train only) -------------------------------------------

    def fit(self, obs_df: pl.DataFrame) -> "CausalHMM":
        """
        Fit scaler + HMM on *inner_train* rows only.

        Parameters
        ----------
        obs_df : pl.DataFrame
            Must contain columns ``HMM_OBS_FEATURES`` with no nulls.
        """
        X = obs_df.select(HMM_OBS_FEATURES).to_numpy().astype(np.float64)

        # Fit RobustScaler on inner_train ONLY
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = np.clip(X_scaled, -self.CLIP_BOUND, self.CLIP_BOUND)

        # Fit GaussianHMM
        self.hmm = GaussianHMM(
            n_components=3,
            covariance_type="full",
            n_iter=100,
            random_state=SEED,
        )
        self.hmm.fit(X_scaled)

        # Deterministic state ordering: ascending by means_[:, 0] (log_rv_60)
        order = np.argsort(self.hmm.means_[:, 0])  # ascending
        self.startprob = self.hmm.startprob_[order]
        self.transmat = self.hmm.transmat_[order][:, order]
        self.means = self.hmm.means_[order]
        self.covars = self.hmm.covars_[order]

        # Pre-compute log quantities for Numba forward filter
        self._log_startprob = np.log(self.startprob + 1e-300)
        self._log_transmat = np.log(self.transmat + 1e-300)

        # Inverse covariances and log-determinants for Gaussian emission
        n_comp, n_feat = self.means.shape
        self._inv_covars = np.empty_like(self.covars)
        self._log_det_covars = np.empty(n_comp, dtype=np.float64)
        for k in range(n_comp):
            self._inv_covars[k] = np.linalg.inv(self.covars[k])
            sign, logdet = np.linalg.slogdet(self.covars[k])
            self._log_det_covars[k] = logdet

        return self

    # ---- Scale (frozen) ---------------------------------------------------

    def transform(self, obs_df: pl.DataFrame) -> np.ndarray:
        """Scale + clip observations using the *frozen* scaler."""
        X = obs_df.select(HMM_OBS_FEATURES).to_numpy().astype(np.float64)
        X_scaled = self.scaler.transform(X)
        return np.clip(X_scaled, -self.CLIP_BOUND, self.CLIP_BOUND)

    # ---- Forward filter (causal, Numba) -----------------------------------

    def forward_filter(
        self,
        obs_scaled: np.ndarray,
        steps_in_segment: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the strictly causal forward filter.

        Parameters
        ----------
        obs_scaled : ndarray (T, 4)
            Scaled & clipped observations.
        steps_in_segment : ndarray (T,) int
            From ingest alignment.  1 ⇒ first valid step in segment.

        Returns
        -------
        regime_p0, regime_p1, regime_p2, regime_entropy : each ndarray (T,)
        """
        return _hmm_forward_filter(
            obs_scaled,
            steps_in_segment.astype(np.int64),
            self._log_startprob,
            self._log_transmat,
            self.means,
            self._inv_covars,
            self._log_det_covars,
        )


@njit(cache=True)
def _log_gaussian_emission(
    o: np.ndarray,          # (D,)
    mu: np.ndarray,         # (D,)
    inv_cov: np.ndarray,    # (D, D)
    log_det: float,
    D: int,
) -> float:
    """Log N(o | mu, Sigma) for a single observation and single component."""
    diff = o - mu
    # Mahalanobis: diff^T @ inv_cov @ diff
    maha = 0.0
    for i in range(D):
        s = 0.0
        for j in range(D):
            s += inv_cov[i, j] * diff[j]
        maha += diff[i] * s
    return -0.5 * (D * 1.8378770664093453 + log_det + maha)
    # 1.8378770664093453 = ln(2*pi)


@njit(cache=True)
def _logsumexp(arr: np.ndarray) -> float:
    """Numerically stable log-sum-exp for a 1-D array."""
    mx = arr[0]
    for i in range(1, arr.shape[0]):
        if arr[i] > mx:
            mx = arr[i]
    s = 0.0
    for i in range(arr.shape[0]):
        s += math.exp(arr[i] - mx)
    return mx + math.log(s)


@njit(cache=True)
def _hmm_forward_filter(
    obs: np.ndarray,             # (T, D)
    steps_in_segment: np.ndarray,  # (T,) int64
    log_pi: np.ndarray,          # (K,)
    log_A: np.ndarray,           # (K, K)
    means: np.ndarray,           # (K, D)
    inv_covars: np.ndarray,      # (K, D, D)
    log_det_covars: np.ndarray,  # (K,)
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    MANDATORY strictly causal forward filter.
    NEVER uses backward smoothing.
    """
    T = obs.shape[0]
    D = obs.shape[1]
    K = means.shape[0]  # 3

    regime_p0 = np.empty(T, dtype=np.float64)
    regime_p1 = np.empty(T, dtype=np.float64)
    regime_p2 = np.empty(T, dtype=np.float64)
    regime_entropy = np.empty(T, dtype=np.float64)

    log_alpha = np.empty(K, dtype=np.float64)
    log_alpha_prev = np.empty(K, dtype=np.float64)
    log_emit = np.empty(K, dtype=np.float64)
    tmp = np.empty(K, dtype=np.float64)

    for t in range(T):
        # Compute emission log-likelihoods for all components
        for k in range(K):
            log_emit[k] = _log_gaussian_emission(
                obs[t], means[k], inv_covars[k], log_det_covars[k], D
            )

        if steps_in_segment[t] == 1 or t == 0:
            # First valid step in segment or very first step: reset to pi
            for k in range(K):
                log_alpha[k] = log_pi[k] + log_emit[k]
        else:
            # Forward propagation: α̃_t(k) = ln(Σ_j α_{t-1}(j)·A_{jk} + 1e-300) + ℓ_t(k)
            for k in range(K):
                for j in range(K):
                    tmp[j] = log_alpha_prev[j] + log_A[j, k]
                log_sum = _logsumexp(tmp)
                log_alpha[k] = log_sum + log_emit[k]

        # Normalize via log-sum-exp
        lse = _logsumexp(log_alpha)
        for k in range(K):
            log_alpha[k] -= lse

        # Convert to probabilities
        p0 = math.exp(log_alpha[0])
        p1 = math.exp(log_alpha[1])
        p2 = math.exp(log_alpha[2])
        regime_p0[t] = p0
        regime_p1[t] = p1
        regime_p2[t] = p2

        # Entropy = -Σ(p · ln(p + 1e-12))
        ent = 0.0
        ent -= p0 * math.log(p0 + 1e-12)
        ent -= p1 * math.log(p1 + 1e-12)
        ent -= p2 * math.log(p2 + 1e-12)
        regime_entropy[t] = ent

        # Save for next step
        for k in range(K):
            log_alpha_prev[k] = log_alpha[k]

    return regime_p0, regime_p1, regime_p2, regime_entropy


# ---------------------------------------------------------------------------
# Component 3B — Rolling Memory / Persistence Indicator
# ---------------------------------------------------------------------------

def compute_rolling_memory(
    df: pl.DataFrame,
    r_1m_col: str = "r_1m",
    r_5m_col: str = "r_5m",
    steps_col: str = "steps_in_segment",
) -> pl.DataFrame:
    """
    Compute strictly causal rolling memory indicators.

    Returns DataFrame with columns:
      - acf1_60:  lag-1 autocorrelation of r_1m over trailing 60 steps
      - acf1_240: lag-1 autocorrelation of r_1m over trailing 240 steps
      - vr_5_60:  Lo-MacKinlay Variance Ratio = Var(r_5m) / (5 * Var(r_1m))
                  over trailing 240 steps
    """
    return df.with_columns(
        # ACF1 over 60 steps
        _rolling_acf1(pl.col(r_1m_col), window=60, steps_col=steps_col).alias("acf1_60"),
        # ACF1 over 240 steps
        _rolling_acf1(pl.col(r_1m_col), window=240, steps_col=steps_col).alias("acf1_240"),
    ).with_columns(
        # Variance ratio: Var(r_5m) / (5 * Var(r_1m)) over trailing 240 steps
        pl.when(pl.col(steps_col) >= 240)
        .then(
            pl.col(r_5m_col).rolling_var(window_size=240, min_periods=240)
            / (5.0 * pl.col(r_1m_col).rolling_var(window_size=240, min_periods=240) + 1e-30)
        )
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("vr_5_60"),
    )


def _rolling_acf1(
    col: pl.Expr,
    window: int,
    steps_col: str = "steps_in_segment",
) -> pl.Expr:
    """
    Strictly causal rolling lag-1 autocorrelation over `window` steps.
    Returns null when steps_in_segment < window.
    """
    # Using Polars rolling covariance:
    #   acf1 = cov(x_t, x_{t-1}) / var(x_t)
    # Both computed over trailing `window` steps.
    lag = col.shift(1)
    cov_expr = pl.rolling_corr(col, lag, window_size=window, min_periods=window)
    return (
        pl.when(pl.col(steps_col) >= window)
        .then(cov_expr)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


# ---------------------------------------------------------------------------
# Component 3C — Bayesian Online Change-Point Detection (BOCPD)
#   Adams & MacKay 2007, Normal-Inverse-Gamma conjugate prior
#   Runs on 5-min changes of log_rv_15
# ---------------------------------------------------------------------------

HAZARD_LAMBDA = 1.0 / 288.0   # ~1 change-point per 24h of 5m ticks
R_MAX = 500                    # Truncate run-length distribution


@njit(cache=True)
def _student_t_logpdf(x: float, mu: float, var: float, nu: float) -> float:
    """
    Log-pdf of Student-t with location mu, scale^2 = var, and df = nu.
    """
    # Using gammaln from pre-computed arrays is not possible inside njit,
    # so we use the Stirling-free formula with lgamma.
    # scipy.special.gammaln is not available in Numba; we use math.lgamma.
    half_nu = 0.5 * nu
    half_nu1 = 0.5 * (nu + 1.0)
    return (
        math.lgamma(half_nu1)
        - math.lgamma(half_nu)
        - 0.5 * math.log(nu * math.pi * var + 1e-300)
        - half_nu1 * math.log(1.0 + (x - mu) ** 2 / (nu * var + 1e-300))
    )


@njit(cache=True)
def _bocpd_filter(
    data: np.ndarray,          # (T,) — 5-min changes of log_rv_15
    is_valid: np.ndarray,      # (T,) bool — whether this 5m tick is valid
    hazard: float,
    r_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Adams & MacKay 2007 BOCPD with Normal-Inverse-Gamma conjugate prior.

    Returns
    -------
    expected_run_len : (T,)
    cp_prob_15m : (T,) — P(run_len ≤ 3)
    """
    T = data.shape[0]
    expected_run_len = np.zeros(T, dtype=np.float64)
    cp_prob_15m = np.zeros(T, dtype=np.float64)

    # NIG prior hyperparameters (weakly informative)
    mu0 = 0.0
    kappa0 = 0.01
    alpha0 = 1.0
    beta0 = 0.01

    # Run-length distribution (truncated at r_max)
    # R[r] = P(run_length = r)
    R = np.zeros(r_max + 1, dtype=np.float64)
    R[0] = 1.0  # Start with run-length 0

    # Sufficient statistics arrays for each run-length
    # For NIG: track sum_x, sum_x2, count per run-length
    ss_sum = np.zeros(r_max + 1, dtype=np.float64)
    ss_sum2 = np.zeros(r_max + 1, dtype=np.float64)
    ss_n = np.zeros(r_max + 1, dtype=np.float64)

    for t in range(T):
        if not is_valid[t]:
            # Reset run-length after gap
            R[:] = 0.0
            R[0] = 1.0
            ss_sum[:] = 0.0
            ss_sum2[:] = 0.0
            ss_n[:] = 0.0
            expected_run_len[t] = 0.0
            cp_prob_15m[t] = 1.0
            continue

        x = data[t]

        # 1. Compute predictive probabilities for each run-length
        pred_logp = np.empty(r_max + 1, dtype=np.float64)
        for r in range(r_max + 1):
            n_r = ss_n[r]
            if n_r < 1.0:
                # Prior predictive: Student-t with prior params
                kn = kappa0
                mn = mu0
                an = alpha0
                bn = beta0
            else:
                sum_x = ss_sum[r]
                sum_x2 = ss_sum2[r]
                kn = kappa0 + n_r
                mn = (kappa0 * mu0 + sum_x) / kn
                an = alpha0 + 0.5 * n_r
                bn = beta0 + 0.5 * (sum_x2 - sum_x * sum_x / n_r) + \
                     0.5 * kappa0 * n_r * (sum_x / n_r - mu0) ** 2 / kn

            nu = 2.0 * an
            pred_var = bn * (kn + 1.0) / (an * kn + 1e-300)
            pred_logp[r] = _student_t_logpdf(x, mn, pred_var, nu)

        # 2. Growth probabilities
        new_R = np.zeros(r_max + 1, dtype=np.float64)
        growth_mass = 0.0
        cp_mass = 0.0

        for r in range(min(r_max, r_max)):
            lp = pred_logp[r]
            prob = R[r] * math.exp(lp) if lp > -500.0 else 0.0
            # Growth
            if r + 1 <= r_max:
                new_R[r + 1] += prob * (1.0 - hazard)
                growth_mass += prob * (1.0 - hazard)
            # Changepoint
            cp_mass += prob * hazard

        new_R[0] += cp_mass

        # Normalize
        total = new_R[0]
        for r in range(1, r_max + 1):
            total += new_R[r]
        if total > 1e-300:
            inv_total = 1.0 / total
            for r in range(r_max + 1):
                new_R[r] *= inv_total

        # 3. Update sufficient statistics
        new_sum = np.zeros(r_max + 1, dtype=np.float64)
        new_sum2 = np.zeros(r_max + 1, dtype=np.float64)
        new_n = np.zeros(r_max + 1, dtype=np.float64)

        # run-length 0 (new segment): reset stats
        new_sum[0] = 0.0
        new_sum2[0] = 0.0
        new_n[0] = 0.0

        # Extend existing run-lengths
        for r in range(r_max):
            new_sum[r + 1] = ss_sum[r] + x
            new_sum2[r + 1] = ss_sum2[r] + x * x
            new_n[r + 1] = ss_n[r] + 1.0

        # Swap
        for r in range(r_max + 1):
            R[r] = new_R[r]
            ss_sum[r] = new_sum[r]
            ss_sum2[r] = new_sum2[r]
            ss_n[r] = new_n[r]

        # 4. Emit features
        erl = 0.0
        cp_p = 0.0
        for r in range(r_max + 1):
            erl += r * R[r]
            if r <= 3:
                cp_p += R[r]
        expected_run_len[t] = erl
        cp_prob_15m[t] = cp_p

    return expected_run_len, cp_prob_15m


def compute_bocpd(
    df: pl.DataFrame,
    log_rv_15_col: str = "log_rv_15",
    steps_col: str = "steps_in_segment",
) -> pl.DataFrame:
    """
    Run BOCPD on 5-min changes of log_rv_15 (forward-filled between ticks).

    Adds columns: bocpd_expected_run_len, bocpd_cp_prob_15m
    """
    log_rv = df[log_rv_15_col].to_numpy().astype(np.float64)
    steps = df[steps_col].to_numpy().astype(np.int64)

    # 5-min changes of log_rv_15 — updated every 5m, forward-filled between
    # We compute at every minute, but only update at 5m boundaries
    T = len(log_rv)
    changes_5m = np.zeros(T, dtype=np.float64)
    is_5m_valid = np.zeros(T, dtype=np.bool_)

    for t in range(T):
        if steps[t] <= 0:
            # Gap
            continue
        # Check if this is a 5-minute boundary (steps % 5 == 0 and enough data)
        if steps[t] >= 5 and (steps[t] % 5 == 0):
            changes_5m[t] = log_rv[t] - log_rv[t - 5] if t >= 5 and steps[t] >= 6 else 0.0
            is_5m_valid[t] = True

    # Run BOCPD only on 5m ticks
    erl_5m, cp_5m = _bocpd_filter(changes_5m, is_5m_valid, HAZARD_LAMBDA, R_MAX)

    # Forward-fill results to 1m grid
    erl_1m = np.full(T, np.nan)
    cp_1m = np.full(T, np.nan)
    last_erl = np.nan
    last_cp = np.nan

    for t in range(T):
        if steps[t] <= 0:
            last_erl = np.nan
            last_cp = np.nan
            erl_1m[t] = np.nan
            cp_1m[t] = np.nan
        elif is_5m_valid[t]:
            last_erl = erl_5m[t]
            last_cp = cp_5m[t]
            erl_1m[t] = last_erl
            cp_1m[t] = last_cp
        else:
            erl_1m[t] = last_erl
            cp_1m[t] = last_cp

    return df.with_columns(
        pl.Series("bocpd_expected_run_len", erl_1m, dtype=pl.Float64),
        pl.Series("bocpd_cp_prob_15m", cp_1m, dtype=pl.Float64),
    )


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def compute_regime_features(
    df: pl.DataFrame,
    inner_train_mask: np.ndarray,
) -> tuple[pl.DataFrame, CausalHMM]:
    """
    Master entry-point for Layer 3 features.

    Parameters
    ----------
    df : pl.DataFrame
        Must contain HMM observation features, r_1m, r_5m, log_rv_15,
        steps_in_segment columns.
    inner_train_mask : ndarray bool (N,)
        True for rows belonging to inner_train.

    Returns
    -------
    df_augmented : pl.DataFrame
        Original df + regime_p0, regime_p1, regime_p2, regime_entropy,
        acf1_60, acf1_240, vr_5_60, bocpd_expected_run_len, bocpd_cp_prob_15m.
    hmm : CausalHMM
        Fitted (frozen) HMM for later use.
    """
    # --- 3A: HMM ---
    inner_train_df = df.filter(pl.Series(inner_train_mask))
    # Drop rows with any null in obs features for fitting
    inner_train_obs = inner_train_df.drop_nulls(subset=HMM_OBS_FEATURES)

    hmm = CausalHMM()
    hmm.fit(inner_train_obs)

    # Scale full dataset with frozen scaler
    obs_full = df.select(HMM_OBS_FEATURES)
    # Handle nulls: fill with 0 for scaling (will be masked by steps_in_segment)
    obs_np = obs_full.fill_null(0.0).to_numpy().astype(np.float64)
    obs_scaled = hmm.scaler.transform(obs_np)
    obs_scaled = np.clip(obs_scaled, -CausalHMM.CLIP_BOUND, CausalHMM.CLIP_BOUND)

    steps_in_seg = df["steps_in_segment"].to_numpy().astype(np.int64)

    p0, p1, p2, entropy = hmm.forward_filter(obs_scaled, steps_in_seg)

    df = df.with_columns(
        pl.Series("regime_p0", p0, dtype=pl.Float64),
        pl.Series("regime_p1", p1, dtype=pl.Float64),
        pl.Series("regime_p2", p2, dtype=pl.Float64),
        pl.Series("regime_entropy", entropy, dtype=pl.Float64),
    )

    # --- 3B: Rolling Memory ---
    df = compute_rolling_memory(df)

    # --- 3C: BOCPD ---
    df = compute_bocpd(df)

    return df, hmm
