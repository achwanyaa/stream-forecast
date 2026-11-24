"""
Layer 1 — Baseline Bench (Spec §5)
  naive_vol_scaled  — Primary Yardstick
  ms_ar             — Markov-Switching Autoregression
  chronos_zeroshot  — GPU Foundation Model
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

try:
    from compute.config import (
        SEED,
        HORIZONS_MIN,
        QUANTILES,
        GPU_STRIDE_MIN,
    )
except ImportError:
    SEED = 42
    HORIZONS_MIN = [1, 5, 15, 60]
    QUANTILES = [0.10, 0.50, 0.90]
    GPU_STRIDE_MIN = 15


# ===========================================================================
# Baseline 1: naive_vol_scaled  (Primary Yardstick)
# ===========================================================================

class NaiveVolScaled:
    """
    Point forecast: q50 = 0
    Uncertainty cones: rolling causal 10th/90th empirical quantiles
    of ỹ_{t,H} over trailing 1440 valid steps (lagged by H so y_{t-H,H}
    has closed). Multiply by σ̂·√H.
    """

    TRAILING_WINDOW = 1440

    def predict(
        self,
        df: pl.DataFrame,
        horizon: int,
        sigma_col: str = "rv_60",
    ) -> pl.DataFrame:
        """
        Produce naive_q10, naive_q50, naive_q90 columns.

        Parameters
        ----------
        df : pl.DataFrame
            Must contain ``ytilde_{horizon}``, ``steps_in_segment``,
            and ``rv_60`` (causal realized volatility).
        horizon : int
            Forecast horizon in minutes (1, 5, 15, or 60).
        """
        ytilde_col = f"ytilde_{horizon}"
        steps_col = "steps_in_segment"
        sqrt_h = float(np.sqrt(horizon))

        # Lag the vol-normalized target by H steps so y_{t-H, H} has closed
        lagged_ytilde = pl.col(ytilde_col).shift(horizon)

        # Rolling 10th and 90th quantiles over trailing 1440 valid steps
        q10_expr = (
            pl.when(pl.col(steps_col) >= self.TRAILING_WINDOW + horizon)
            .then(
                lagged_ytilde.rolling_quantile(
                    quantile=0.10,
                    window_size=self.TRAILING_WINDOW,
                    min_periods=self.TRAILING_WINDOW,
                )
            )
            .otherwise(pl.lit(None, dtype=pl.Float64))
        )
        q90_expr = (
            pl.when(pl.col(steps_col) >= self.TRAILING_WINDOW + horizon)
            .then(
                lagged_ytilde.rolling_quantile(
                    quantile=0.90,
                    window_size=self.TRAILING_WINDOW,
                    min_periods=self.TRAILING_WINDOW,
                )
            )
            .otherwise(pl.lit(None, dtype=pl.Float64))
        )

        # Scale back: multiply by σ̂ · √H
        scale = pl.col(sigma_col) * sqrt_h

        return df.with_columns(
            (q10_expr * scale).alias(f"naive_q10_h{horizon}"),
            pl.lit(0.0, dtype=pl.Float64).alias(f"naive_q50_h{horizon}"),
            (q90_expr * scale).alias(f"naive_q90_h{horizon}"),
        )


# ===========================================================================
# Baseline 2: ms_ar  (Markov-Switching Autoregression)
# ===========================================================================

class MarkovSwitchingAR:
    """
    Fit statsmodels MarkovAutoregression(k_regimes=3, order=1,
    switching_ar=True, switching_variance=True) on trailing 20k steps
    of inner_train.

    Test: filtered_marginal_probabilities only → conditional P10/P50/P90.
    """

    MAX_TRAIN_STEPS = 20_000

    def __init__(self) -> None:
        self.model = None
        self.results = None

    def fit(
        self,
        inner_train_ytilde: np.ndarray,
    ) -> "MarkovSwitchingAR":
        """
        Fit MS-AR on trailing 20k steps of inner_train.

        Parameters
        ----------
        inner_train_ytilde : ndarray (N,)
            Vol-normalized targets ỹ_{t,H} from inner_train, non-null.
        """
        from statsmodels.tsa.regime_switching.markov_autoregression import (
            MarkovAutoregression,
        )

        # Use trailing 20k steps for runtime stability
        data = inner_train_ytilde[-self.MAX_TRAIN_STEPS:]

        self.model = MarkovAutoregression(
            data,
            k_regimes=3,
            order=1,
            switching_ar=True,
            switching_variance=True,
        )
        self.results = self.model.fit(search_reps=20, random_state=SEED)
        return self

    def predict(
        self,
        test_ytilde: np.ndarray,
        sigma: np.ndarray,
        horizon: int,
    ) -> dict[str, np.ndarray]:
        """
        Produce conditional P10/P50/P90 on test data using
        filtered_marginal_probabilities ONLY.

        Parameters
        ----------
        test_ytilde : ndarray (T,)
            Vol-normalized targets in test fold.
        sigma : ndarray (T,)
            Causal realized vol (rv_60) at each test step.
        horizon : int
            Forecast horizon.

        Returns
        -------
        dict with keys ms_ar_q10, ms_ar_q50, ms_ar_q90
        """
        from scipy.stats import norm

        res = self.results

        # Get regime parameters
        k_regimes = 3
        # Intercepts (const), AR coefficients, and variances per regime
        params = res.params
        n_params = len(params)

        # Extract regime-specific parameters from the fitted model
        # regime intercepts, AR(1) coefficients, variances
        regime_means = np.zeros(k_regimes)
        regime_vars = np.zeros(k_regimes)
        regime_ar = np.zeros(k_regimes)

        for k in range(k_regimes):
            regime_means[k] = res.params[f"const[{k}]"] if hasattr(res.params, '__getitem__') else 0.0
            regime_vars[k] = res.params[f"sigma2[{k}]"] if hasattr(res.params, '__getitem__') else 1.0
            regime_ar[k] = res.params[f"ar.L1[{k}]"] if hasattr(res.params, '__getitem__') else 0.0

        # Use filtered probabilities (strictly causal)
        # Re-run filter on test data appended to training context
        from statsmodels.tsa.regime_switching.markov_autoregression import (
            MarkovAutoregression,
        )

        # Concatenate training tail + test for filtering
        train_tail = self.model.endog[-500:]  # Keep context
        full_seq = np.concatenate([train_tail, test_ytilde])

        # Apply filter using fitted parameters
        mod_full = MarkovAutoregression(
            full_seq,
            k_regimes=3,
            order=1,
            switching_ar=True,
            switching_variance=True,
        )
        filtered = mod_full.filter(res.params)
        # filtered_marginal_probabilities: (T_full, k_regimes)
        filt_probs = filtered.filtered_marginal_probabilities

        # Extract test portion
        test_start = len(train_tail)
        test_probs = filt_probs[test_start:]  # (T_test, 3)

        T_test = len(test_ytilde)
        q10 = np.zeros(T_test)
        q50 = np.zeros(T_test)
        q90 = np.zeros(T_test)

        sqrt_h = np.sqrt(horizon)

        for t in range(T_test):
            # Conditional mean and variance (mixture of Gaussians)
            prev_y = test_ytilde[t - 1] if t > 0 else 0.0
            # Weighted sum across regimes
            for k in range(k_regimes):
                mu_k = regime_means[k] + regime_ar[k] * prev_y
                std_k = np.sqrt(regime_vars[k])
                p_k = test_probs[t, k] if t < len(test_probs) else 1.0 / k_regimes
                q10[t] += p_k * (mu_k + std_k * norm.ppf(0.10))
                q50[t] += p_k * mu_k
                q90[t] += p_k * (mu_k + std_k * norm.ppf(0.90))

            # Scale back to raw space
            q10[t] *= sigma[t] * sqrt_h
            q50[t] *= sigma[t] * sqrt_h
            q90[t] *= sigma[t] * sqrt_h

        return {
            f"ms_ar_q10_h{horizon}": q10,
            f"ms_ar_q50_h{horizon}": q50,
            f"ms_ar_q90_h{horizon}": q90,
        }


# ===========================================================================
# Baseline 3: chronos_zeroshot  (GPU Foundation Model)
# ===========================================================================

class ChronosZeroShot:
    """
    Load amazon/chronos-bolt-base on GPU bfloat16.
    512-step context of vol-normalized log-returns.
    Batch 256. Strided timestamps (minute%15==0).
    Forecast H∈{1,5,15,60} at quantiles [0.10, 0.50, 0.90].
    """

    CONTEXT_LENGTH = 512
    BATCH_SIZE = 256
    MODEL_ID = "amazon/chronos-bolt-base"

    def __init__(self) -> None:
        self.pipeline = None

    def load_model(self) -> "ChronosZeroShot":
        """Load Chronos pipeline on GPU with bfloat16."""
        import torch
        from chronos import ChronosPipeline

        self.pipeline = ChronosPipeline.from_pretrained(
            self.MODEL_ID,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )
        return self

    def predict(
        self,
        test_df: pl.DataFrame,
        full_rtilde: np.ndarray,
        test_start_idx: int,
        horizons: list[int] | None = None,
        quantiles: list[float] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Run zero-shot inference at strided timestamps (minute%15==0).

        Parameters
        ----------
        test_df : pl.DataFrame
            Test fold with close_time_ms column.
        full_rtilde : ndarray
            Full vol-normalized 1-min log-returns (entire dataset).
        test_start_idx : int
            Index into full_rtilde where test fold starts.
        horizons : list[int]
            Forecast horizons (default: HORIZONS_MIN).
        quantiles : list[float]
            Quantile levels (default: QUANTILES).

        Returns
        -------
        dict with keys like chronos_q10_h1, chronos_q50_h1, etc.
        """
        import torch

        if horizons is None:
            horizons = HORIZONS_MIN
        if quantiles is None:
            quantiles = QUANTILES

        close_times = test_df["close_time_ms"].to_numpy()
        T_test = len(close_times)

        # Identify strided timestamps: minute % 15 == 0
        minutes = (close_times // 60_000) % (24 * 60)
        stride_mask = (minutes % GPU_STRIDE_MIN) == 0
        stride_indices = np.where(stride_mask)[0]

        max_horizon = max(horizons)

        # Prepare batched contexts
        results: dict[str, np.ndarray] = {}
        for h in horizons:
            for q_val in quantiles:
                q_label = f"{int(q_val * 100):02d}"
                results[f"chronos_q{q_label}_h{h}"] = np.full(T_test, np.nan)

        # Process in batches
        for batch_start in range(0, len(stride_indices), self.BATCH_SIZE):
            batch_idx = stride_indices[batch_start: batch_start + self.BATCH_SIZE]
            contexts = []

            for idx in batch_idx:
                abs_idx = test_start_idx + idx
                ctx_start = max(0, abs_idx - self.CONTEXT_LENGTH)
                ctx = full_rtilde[ctx_start:abs_idx]
                # Pad with zeros if not enough context
                if len(ctx) < self.CONTEXT_LENGTH:
                    ctx = np.concatenate([
                        np.zeros(self.CONTEXT_LENGTH - len(ctx)),
                        ctx,
                    ])
                contexts.append(torch.tensor(ctx, dtype=torch.float32))

            context_batch = torch.stack(contexts)

            # Forecast
            forecast = self.pipeline.predict_quantiles(
                context_batch,
                prediction_length=max_horizon,
                quantile_levels=quantiles,
            )
            # forecast shape: (batch, num_quantiles, prediction_length)

            for i, idx in enumerate(batch_idx):
                for q_i, q_val in enumerate(quantiles):
                    q_label = f"{int(q_val * 100):02d}"
                    for h in horizons:
                        # Cumulative horizon: sum first h steps
                        cum_val = float(forecast[i, q_i, h - 1])
                        results[f"chronos_q{q_label}_h{h}"][idx] = cum_val

        return results


# ===========================================================================
# Convenience dispatcher
# ===========================================================================

def run_baseline(
    name: str,
    inner_train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    horizon: int,
    **kwargs: Any,
) -> dict[str, np.ndarray]:
    """
    Run a named baseline on a single fold/horizon.

    Parameters
    ----------
    name : str
        One of 'naive_vol_scaled', 'ms_ar', 'chronos_zeroshot'.
    """
    if name == "naive_vol_scaled":
        model = NaiveVolScaled()
        result_df = model.predict(test_df, horizon=horizon)
        return {
            c: result_df[c].to_numpy()
            for c in result_df.columns
            if c.startswith("naive_")
        }

    elif name == "ms_ar":
        ytilde_col = f"ytilde_{horizon}"
        inner_ytilde = (
            inner_train_df[ytilde_col]
            .drop_nulls()
            .to_numpy()
            .astype(np.float64)
        )
        model = MarkovSwitchingAR()
        model.fit(inner_ytilde)

        test_ytilde = test_df[ytilde_col].fill_null(0.0).to_numpy().astype(np.float64)
        test_sigma = test_df["rv_60"].fill_null(1e-6).to_numpy().astype(np.float64)
        return model.predict(test_ytilde, test_sigma, horizon)

    elif name == "chronos_zeroshot":
        model = ChronosZeroShot()
        model.load_model()
        return model.predict(
            test_df,
            full_rtilde=kwargs["full_rtilde"],
            test_start_idx=kwargs["test_start_idx"],
            horizons=kwargs.get("horizons", HORIZONS_MIN),
            quantiles=kwargs.get("quantiles", QUANTILES),
        )

    else:
        raise ValueError(f"Unknown baseline: {name}")
