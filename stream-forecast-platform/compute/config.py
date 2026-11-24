"""Global constants, horizons, windows, fold geometry, and model hyperparameters.

All values sourced from SYSTEM_SPEC.md §4 and §8.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Deterministic Execution (§0 / §1)
# ---------------------------------------------------------------------------
SEED: int = 42

# ---------------------------------------------------------------------------
# Fold Geometry — §4
# ---------------------------------------------------------------------------
WARMUP_ROWS: int = 10_080          # 7 days (required for weekly rolling features)
TRAIN_ROWS: int = 129_600          # 90 days rolling training window (default)
INNER_VAL_ROWS: int = 14_400       # 10 days carved from END of train for LightGBM early stopping
EMBARGO_ROWS: int = 1_440          # 24 hours (seals diurnal & volatility-cluster bleed)
TEST_ROWS: int = 21_600            # 15 days per unseen out-of-sample fold
STEP_ROWS: int = 21_600            # Non-overlapping test folds

# ---------------------------------------------------------------------------
# Horizons & Quantiles — §4
# ---------------------------------------------------------------------------
HORIZONS_MIN: list[int] = [1, 5, 15, 60]
QUANTILES: list[float] = [0.10, 0.50, 0.90]

# ---------------------------------------------------------------------------
# GPU Stride — §4 / §5
# ---------------------------------------------------------------------------
GPU_STRIDE_MIN: int = 15           # Strided evaluation grid for Chronos GPU

# ---------------------------------------------------------------------------
# Train Window Ablation Modes — §4
# ---------------------------------------------------------------------------
TRAIN_WINDOW_MODES: dict[str, int] = {
    "short_30d": 43_200,
    "default_90d": 129_600,
    "long_180d": 259_200,
}

# ---------------------------------------------------------------------------
# HMM Observation Feature Names — §7 (Component 3A)
# ---------------------------------------------------------------------------
HMM_OBS_FEATURES: list[str] = [
    "log_rv_60",
    "vol_ratio_15_240",
    "perm_entropy_60",
    "abs_agg_imbalance_15",
]

# ---------------------------------------------------------------------------
# HMM Observation Clip Bounds — §7
# ---------------------------------------------------------------------------
HMM_CLIP_LOWER: float = -8.0
HMM_CLIP_UPPER: float = 8.0

# ---------------------------------------------------------------------------
# Staleness Thresholds (minutes) — §2
# ---------------------------------------------------------------------------
STALENESS_5M: float = 15.0         # tier2_5m staleness ceiling (minutes)
STALENESS_8H: float = 960.0        # tier2_8h staleness ceiling (minutes, = 16 hours)

# ---------------------------------------------------------------------------
# Target Clip Bounds — §4 (vol-normalized training targets)
# ---------------------------------------------------------------------------
TARGET_CLIP_LOWER: float = -15.0
TARGET_CLIP_UPPER: float = 15.0

# ---------------------------------------------------------------------------
# Floor Constants — §4 / §6
# ---------------------------------------------------------------------------
VOLATILITY_FLOOR: float = 1e-6     # min realized vol σ̂_{t,60}
EPSILON: float = 1e-12             # general numerical floor
LOG_MAG_FLOOR: float = 1e-8        # inside cascade magnitude target ln(sqrt(…) + floor)

# ---------------------------------------------------------------------------
# Cascade Threshold Multiplier — §4
# ---------------------------------------------------------------------------
CASCADE_THRESHOLD_MULTIPLIER: float = 4.0

# ---------------------------------------------------------------------------
# BOCPD Parameters — §7 (Component 3C)
# ---------------------------------------------------------------------------
BOCPD_HAZARD_RATE: float = 1.0 / 288   # ~1 regime break per 24 hours
BOCPD_MAX_RUN_LENGTH: int = 500

# ---------------------------------------------------------------------------
# Naive Baseline Trailing Window — §5
# ---------------------------------------------------------------------------
NAIVE_TRAILING_STEPS: int = 1_440  # rolling empirical quantile window

# ---------------------------------------------------------------------------
# Markov-Switching AR Training Rows — §5
# ---------------------------------------------------------------------------
MS_AR_TRAILING_STEPS: int = 20_000

# ---------------------------------------------------------------------------
# Chronos Context Length — §5
# ---------------------------------------------------------------------------
CHRONOS_CONTEXT_LENGTH: int = 512
CHRONOS_BATCH_SIZE: int = 256

# ---------------------------------------------------------------------------
# LightGBM Locked Hyperparameters — §8
# ---------------------------------------------------------------------------
LGBM_PARAMS: dict = {
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

# ---------------------------------------------------------------------------
# LightGBM Early Stopping — §8
# ---------------------------------------------------------------------------
LGBM_EARLY_STOPPING_ROUNDS: int = 50

# ---------------------------------------------------------------------------
# Noise Control — §4
# ---------------------------------------------------------------------------
NOISE_CONTROL_SEEDS: list[int] = [100, 200, 300, 400, 500]
NOISE_CONTROL_SIGMA_THRESHOLD: float = 2.0  # must beat max control by >2 std

# ---------------------------------------------------------------------------
# Regime MoE Clip Floor — §8 (Model 4B)
# ---------------------------------------------------------------------------
REGIME_MOE_WEIGHT_FLOOR: float = 0.01

# ---------------------------------------------------------------------------
# HMM Fitting — §7
# ---------------------------------------------------------------------------
HMM_N_COMPONENTS: int = 3
HMM_COVARIANCE_TYPE: str = "full"
HMM_N_ITER: int = 100
