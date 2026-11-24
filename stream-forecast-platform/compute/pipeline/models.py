"""
Layer 4 — Forecasting Core (Spec §8)
  Model 4A: lgbm_quantile   — Global Quantile LightGBM
  Model 4B: lgbm_regime_moe — Regime Mixture-of-Experts
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import lightgbm as lgb

try:
    from compute.config import SEED, QUANTILES, LGBM_PARAMS
except ImportError:
    SEED = 42
    QUANTILES = [0.10, 0.50, 0.90]
    LGBM_PARAMS = {
        "objective": "quantile",
        "boosting_type": "gbdt",
        "n_estimators": 1500,
        "learning_rate": 0.025,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 500,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_alpha": 2.0,
        "reg_lambda": 10.0,
        "n_jobs": 4,
        "random_state": SEED,
        "verbose": -1,
    }


# ===========================================================================
# Helpers
# ===========================================================================

def _build_lgbm_params(alpha: float | None = None, objective: str = "quantile") -> dict:
    """Build LightGBM param dict for a specific quantile or Huber."""
    params = dict(LGBM_PARAMS)
    params["objective"] = objective
    if objective == "quantile" and alpha is not None:
        params["alpha"] = alpha
    return params


def _time_decay_weights(
    timestamps_ms: np.ndarray,
    half_life_days: float | None = None,
) -> np.ndarray | None:
    """
    Optional exponential time-decay sample weights.
    w_i = 0.5^((t_end - t_i) / half_life)
    """
    if half_life_days is None:
        return None
    half_life_ms = half_life_days * 24 * 60 * 60_000
    t_end = timestamps_ms.max()
    decay = np.power(0.5, (t_end - timestamps_ms) / half_life_ms)
    return decay.astype(np.float64)


def _non_crossing_sort(q10: np.ndarray, q50: np.ndarray, q90: np.ndarray):
    """
    Enforce non-crossing quantile repair: sort (q10, q50, q90) ascending per row.
    Modifies arrays in-place.
    """
    stacked = np.column_stack([q10, q50, q90])
    stacked.sort(axis=1)
    q10[:] = stacked[:, 0]
    q50[:] = stacked[:, 1]
    q90[:] = stacked[:, 2]


# ===========================================================================
# Model 4A: Global Quantile LightGBM (lgbm_quantile)
# ===========================================================================

class LGBMQuantile:
    """
    Three quantile regressors (α=0.10, 0.50, 0.90) + one Huber regressor
    for m_{t,H}. Early stopping on inner_val (50 rounds).
    """

    EARLY_STOPPING_ROUNDS = 50

    def __init__(self, horizon: int, decay_half_life_days: float | None = None):
        self.horizon = horizon
        self.decay_half_life_days = decay_half_life_days
        self.models: dict[str, lgb.LGBMRegressor] = {}

    def fit(
        self,
        X_train: np.ndarray,
        y_train_qtile: np.ndarray,    # ỹ_{t,H}
        y_train_mag: np.ndarray,      # m_{t,H}
        X_val: np.ndarray,
        y_val_qtile: np.ndarray,
        y_val_mag: np.ndarray,
        train_timestamps_ms: np.ndarray | None = None,
    ) -> "LGBMQuantile":
        """
        Fit all 4 regressors (3 quantile + 1 Huber) with early stopping.
        """
        sample_weight = None
        if train_timestamps_ms is not None and self.decay_half_life_days:
            sample_weight = _time_decay_weights(
                train_timestamps_ms, self.decay_half_life_days
            )

        callbacks = [lgb.early_stopping(self.EARLY_STOPPING_ROUNDS, verbose=False)]

        # --- Quantile regressors ---
        for alpha in QUANTILES:
            label = f"q{int(alpha * 100):02d}"
            params = _build_lgbm_params(alpha=alpha, objective="quantile")
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train,
                y_train_qtile,
                eval_set=[(X_val, y_val_qtile)],
                callbacks=callbacks,
                sample_weight=sample_weight,
            )
            self.models[label] = model

        # --- Huber regressor for magnitude m_{t,H} ---
        params_huber = _build_lgbm_params(objective="huber")
        model_huber = lgb.LGBMRegressor(**params_huber)
        model_huber.fit(
            X_train,
            y_train_mag,
            eval_set=[(X_val, y_val_mag)],
            callbacks=callbacks,
            sample_weight=sample_weight,
        )
        self.models["magnitude"] = model_huber

        return self

    def predict(
        self,
        X_test: np.ndarray,
        sigma_test: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Predict quantiles and magnitude on test data.

        Returns raw-scale predictions:
            q̂_{t,H}^(q) = q̃̂_{t,H}^(q) · σ̂_{t,60} · √H
        """
        sqrt_h = np.sqrt(self.horizon)

        # Normalized-scale predictions
        qtilde_10 = self.models["q10"].predict(X_test)
        qtilde_50 = self.models["q50"].predict(X_test)
        qtilde_90 = self.models["q90"].predict(X_test)

        # Non-crossing sort on normalized scale
        _non_crossing_sort(qtilde_10, qtilde_50, qtilde_90)

        # Reconstruct raw-scale
        scale = sigma_test * sqrt_h
        q10 = qtilde_10 * scale
        q50 = qtilde_50 * scale
        q90 = qtilde_90 * scale

        # Magnitude prediction
        mag = self.models["magnitude"].predict(X_test)

        h = self.horizon
        return {
            f"lgbm_q10_h{h}": q10,
            f"lgbm_q50_h{h}": q50,
            f"lgbm_q90_h{h}": q90,
            f"lgbm_mag_h{h}": mag,
        }

    def feature_importance(self) -> dict[str, np.ndarray]:
        """Return mean gain importances across the 3 quantile models."""
        importances = []
        for label in ["q10", "q50", "q90"]:
            importances.append(
                self.models[label].feature_importances_
            )
        return {
            "mean_gain": np.mean(importances, axis=0),
            "std_gain": np.std(importances, axis=0),
        }


# ===========================================================================
# Model 4B: Regime Mixture-of-Experts (lgbm_regime_moe)
# ===========================================================================

class LGBMRegimeMoE:
    """
    Per quantile per regime: specialized LightGBM weighted by regime_p{k}
    (clipped below 0.01 → 0).

    Inference: blend by forward-filtered probs:
        q̂ = Σ_k P(S_t = k | O_{1:t}) · q̂_k
    Non-crossing sort applied after blending.
    """

    EARLY_STOPPING_ROUNDS = 50
    REGIME_WEIGHT_FLOOR = 0.01  # Below this → weight = 0

    def __init__(self, horizon: int, decay_half_life_days: float | None = None):
        self.horizon = horizon
        self.decay_half_life_days = decay_half_life_days
        # experts[alpha_label][regime_k] = LGBMRegressor
        self.experts: dict[str, dict[int, lgb.LGBMRegressor]] = {}

    def fit(
        self,
        X_train: np.ndarray,
        y_train_qtile: np.ndarray,
        regime_probs_train: np.ndarray,  # (N, 3)
        X_val: np.ndarray,
        y_val_qtile: np.ndarray,
        regime_probs_val: np.ndarray,
        train_timestamps_ms: np.ndarray | None = None,
    ) -> "LGBMRegimeMoE":
        """
        Fit per-regime experts for each quantile.
        """
        base_weight = None
        if train_timestamps_ms is not None and self.decay_half_life_days:
            base_weight = _time_decay_weights(
                train_timestamps_ms, self.decay_half_life_days
            )

        callbacks = [lgb.early_stopping(self.EARLY_STOPPING_ROUNDS, verbose=False)]

        for alpha in QUANTILES:
            alpha_label = f"q{int(alpha * 100):02d}"
            self.experts[alpha_label] = {}

            for k in range(3):
                params = _build_lgbm_params(alpha=alpha, objective="quantile")
                model = lgb.LGBMRegressor(**params)

                # Sample weights = regime_p{k}, clipped below 0.01 → 0
                rw_train = regime_probs_train[:, k].copy()
                rw_train[rw_train < self.REGIME_WEIGHT_FLOOR] = 0.0

                rw_val = regime_probs_val[:, k].copy()
                rw_val[rw_val < self.REGIME_WEIGHT_FLOOR] = 0.0

                # Combine with optional time-decay
                sw_train = rw_train
                if base_weight is not None:
                    sw_train = rw_train * base_weight

                # Only fit if we have enough weighted samples
                if sw_train.sum() < 10.0:
                    # Not enough data for this regime; use uniform fallback
                    model.fit(
                        X_train,
                        y_train_qtile,
                        eval_set=[(X_val, y_val_qtile)],
                        callbacks=callbacks,
                    )
                else:
                    model.fit(
                        X_train,
                        y_train_qtile,
                        eval_set=[(X_val, y_val_qtile)],
                        callbacks=callbacks,
                        sample_weight=sw_train,
                    )

                self.experts[alpha_label][k] = model

        return self

    def predict(
        self,
        X_test: np.ndarray,
        regime_probs_test: np.ndarray,  # (T, 3) — forward-filtered probs
        sigma_test: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Blend per-regime expert predictions using forward-filtered probs:
            q̂ = Σ_k P(S_t = k | O_{1:t}) · q̂_k
        """
        sqrt_h = np.sqrt(self.horizon)
        T = X_test.shape[0]

        qtilde = {}
        for alpha_label in self.experts:
            blended = np.zeros(T, dtype=np.float64)
            for k in range(3):
                pred_k = self.experts[alpha_label][k].predict(X_test)
                blended += regime_probs_test[:, k] * pred_k
            qtilde[alpha_label] = blended

        # Non-crossing sort on normalized scale
        q10_tilde = qtilde["q10"].copy()
        q50_tilde = qtilde["q50"].copy()
        q90_tilde = qtilde["q90"].copy()
        _non_crossing_sort(q10_tilde, q50_tilde, q90_tilde)

        # Raw-scale reconstruction
        scale = sigma_test * sqrt_h
        h = self.horizon
        return {
            f"moe_q10_h{h}": q10_tilde * scale,
            f"moe_q50_h{h}": q50_tilde * scale,
            f"moe_q90_h{h}": q90_tilde * scale,
        }


# ===========================================================================
# Convenience dispatcher
# ===========================================================================

def run_model(
    name: str,
    horizon: int,
    X_inner_train: np.ndarray,
    y_inner_train_qtile: np.ndarray,
    y_inner_train_mag: np.ndarray,
    regime_probs_train: np.ndarray,
    X_inner_val: np.ndarray,
    y_inner_val_qtile: np.ndarray,
    y_inner_val_mag: np.ndarray,
    regime_probs_val: np.ndarray,
    X_test: np.ndarray,
    regime_probs_test: np.ndarray,
    sigma_test: np.ndarray,
    train_timestamps_ms: np.ndarray | None = None,
    decay_half_life_days: float | None = None,
) -> dict[str, np.ndarray]:
    """
    Run a named Layer 4 model for one fold & horizon.

    Parameters
    ----------
    name : str
        'lgbm_quantile' or 'lgbm_regime_moe'.

    Returns
    -------
    dict mapping prediction column names → arrays.
    """
    if name == "lgbm_quantile":
        model = LGBMQuantile(
            horizon=horizon,
            decay_half_life_days=decay_half_life_days,
        )
        model.fit(
            X_inner_train,
            y_inner_train_qtile,
            y_inner_train_mag,
            X_inner_val,
            y_inner_val_qtile,
            y_inner_val_mag,
            train_timestamps_ms=train_timestamps_ms,
        )
        preds = model.predict(X_test, sigma_test)
        preds["_feature_importance"] = model.feature_importance()
        return preds

    elif name == "lgbm_regime_moe":
        model = LGBMRegimeMoE(
            horizon=horizon,
            decay_half_life_days=decay_half_life_days,
        )
        model.fit(
            X_inner_train,
            y_inner_train_qtile,
            regime_probs_train,
            X_inner_val,
            y_inner_val_qtile,
            regime_probs_val,
            train_timestamps_ms=train_timestamps_ms,
        )
        return model.predict(X_test, regime_probs_test, sigma_test)

    else:
        raise ValueError(f"Unknown model: {name}")
