"""
Data Export & Downsampling (Spec §10 — Chart Payload Downsampling)

overview_1h.json  — 1-hour OHLC overview (~8760-17520 points)
fold_N_1m.json    — Per-fold columnar JSON, gzipped (~350 KB each)

Supabase Storage upload + Postgres insert functions.
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


# ===========================================================================
# Downsampling helpers
# ===========================================================================

def _round_sig(x: float, sig: int = 6) -> float:
    """Round a float to `sig` significant digits."""
    if x == 0.0 or math.isnan(x) or math.isinf(x):
        return x
    return round(x, sig - 1 - int(math.floor(math.log10(abs(x)))))


def _series_to_sigdig_list(arr: np.ndarray, sig: int = 6) -> list:
    """Convert numpy array to list with values rounded to sig digits."""
    out = []
    for v in arr:
        if np.isnan(v):
            out.append(None)
        else:
            out.append(_round_sig(float(v), sig))
    return out


# ===========================================================================
# overview_1h.json
# ===========================================================================

def build_overview_1h(
    master_df: pl.DataFrame,
    preds_df: pl.DataFrame | None = None,
    cascade_timestamps: list[int] | None = None,
) -> dict:
    """
    Build 1-hour OHLC overview payload.

    Contents:
      - 1-hour OHLC of x_close
      - Mean regime probs (regime_p0, regime_p1, regime_p2) per hour
      - Hourly mean Pinball loss (if predictions available)
      - Cascade event timestamps

    Returns ~8760–17520 data points for 12–24 months.
    """
    # Add hour bucket column
    df = master_df.with_columns(
        (pl.col("close_time_ms") // 3_600_000 * 3_600_000).alias("hour_ms")
    )

    # 1-hour OHLC of x_close
    ohlc = (
        df.filter(~pl.col("is_gap"))
        .group_by("hour_ms")
        .agg(
            pl.col("x_close").first().alias("open"),
            pl.col("x_close").max().alias("high"),
            pl.col("x_close").min().alias("low"),
            pl.col("x_close").last().alias("close"),
            # Mean regime probs (if available)
            *(
                [
                    pl.col("regime_p0").mean().alias("r0"),
                    pl.col("regime_p1").mean().alias("r1"),
                    pl.col("regime_p2").mean().alias("r2"),
                ]
                if "regime_p0" in df.columns
                else []
            ),
        )
        .sort("hour_ms")
    )

    result: dict[str, Any] = {
        "t": ohlc["hour_ms"].to_list(),
        "open": _series_to_sigdig_list(ohlc["open"].to_numpy()),
        "high": _series_to_sigdig_list(ohlc["high"].to_numpy()),
        "low": _series_to_sigdig_list(ohlc["low"].to_numpy()),
        "close": _series_to_sigdig_list(ohlc["close"].to_numpy()),
    }

    if "regime_p0" in df.columns:
        result["r0"] = _series_to_sigdig_list(ohlc["r0"].to_numpy())
        result["r1"] = _series_to_sigdig_list(ohlc["r1"].to_numpy())
        result["r2"] = _series_to_sigdig_list(ohlc["r2"].to_numpy())

    # Hourly mean Pinball if predictions available
    if preds_df is not None and "pinball_mean" in preds_df.columns:
        preds_hourly = (
            preds_df.with_columns(
                (pl.col("close_time_ms") // 3_600_000 * 3_600_000).alias("hour_ms")
            )
            .group_by("hour_ms")
            .agg(pl.col("pinball_mean").mean().alias("pinball_mean"))
            .sort("hour_ms")
        )
        # Left-join onto the OHLC timestamps
        ohlc_with_pinball = ohlc.join(preds_hourly, on="hour_ms", how="left")
        if "pinball_mean" in ohlc_with_pinball.columns:
            result["pinball"] = _series_to_sigdig_list(
                ohlc_with_pinball["pinball_mean"].to_numpy()
            )

    # Cascade timestamps
    result["cascade_ts"] = cascade_timestamps or []

    return result


# ===========================================================================
# folds/fold_{idx}_1m.json  (gzipped columnar JSON)
# ===========================================================================

def build_fold_payload(
    fold_df: pl.DataFrame,
    fold_idx: int,
) -> dict:
    """
    Build per-fold columnar JSON payload.

    Schema: {t, x, p10, p50, p90, r0, r1, r2, cascade}
    All values rounded to 6 significant digits.
    ~21,600 rows per 15-day fold.
    """
    result: dict[str, list] = {
        "t": fold_df["close_time_ms"].to_list(),
        "x": _series_to_sigdig_list(fold_df["x_close"].to_numpy()),
    }

    # Prediction columns (use first available horizon's predictions)
    for col_prefix, key in [
        ("q10", "p10"),
        ("q50", "p50"),
        ("q90", "p90"),
    ]:
        # Look for prediction columns with any model prefix
        pred_col = None
        for candidate in fold_df.columns:
            if candidate.endswith(col_prefix) or col_prefix in candidate:
                pred_col = candidate
                break
        if pred_col and pred_col in fold_df.columns:
            result[key] = _series_to_sigdig_list(fold_df[pred_col].to_numpy())
        else:
            result[key] = [None] * len(fold_df)

    # Regime probabilities
    for col, key in [
        ("regime_p0", "r0"),
        ("regime_p1", "r1"),
        ("regime_p2", "r2"),
    ]:
        if col in fold_df.columns:
            result[key] = _series_to_sigdig_list(fold_df[col].to_numpy())
        else:
            result[key] = [None] * len(fold_df)

    # Cascade flags
    cascade_col = None
    for c in fold_df.columns:
        if "cascade" in c.lower():
            cascade_col = c
            break
    if cascade_col:
        result["cascade"] = fold_df[cascade_col].to_list()
    else:
        result["cascade"] = [0] * len(fold_df)

    return result


def save_fold_gzipped(
    payload: dict,
    output_dir: Path,
    fold_idx: int,
) -> Path:
    """
    Save fold payload as gzipped JSON.

    Returns the path to the written file.
    """
    folds_dir = output_dir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    out_path = folds_dir / f"fold_{fold_idx}_1m.json.gz"
    json_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    with gzip.open(out_path, "wb", compresslevel=6) as f:
        f.write(json_bytes)
    return out_path


# ===========================================================================
# Supabase Storage upload
# ===========================================================================

def upload_to_supabase_storage(
    supabase_client: Any,
    bucket: str,
    remote_path: str,
    local_path: Path,
    content_type: str = "application/json",
) -> str:
    """
    Upload a file to Supabase Storage.

    Parameters
    ----------
    supabase_client : supabase.Client
    bucket : str
        Storage bucket name (e.g. 'artifacts').
    remote_path : str
        Path within the bucket.
    local_path : Path
        Local file to upload.
    content_type : str
        MIME type.

    Returns
    -------
    str : The storage path uploaded to.
    """
    with open(local_path, "rb") as f:
        data = f.read()

    # Supabase Storage upload (upsert to allow re-runs)
    supabase_client.storage.from_(bucket).upload(
        path=remote_path,
        file=data,
        file_options={
            "content-type": content_type,
            "upsert": "true",
        },
    )
    return remote_path


def upload_overview(
    supabase_client: Any,
    experiment_id: str,
    overview_payload: dict,
    output_dir: Path,
) -> str:
    """Save and upload overview_1h.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    local = output_dir / "overview_1h.json"
    with open(local, "w") as f:
        json.dump(overview_payload, f, separators=(",", ":"))

    remote = f"experiments/{experiment_id}/overview_1h.json"
    upload_to_supabase_storage(
        supabase_client, "artifacts", remote, local
    )
    return remote


def upload_fold(
    supabase_client: Any,
    experiment_id: str,
    fold_payload: dict,
    fold_idx: int,
    output_dir: Path,
) -> str:
    """Save, gzip, and upload a fold payload."""
    gz_path = save_fold_gzipped(fold_payload, output_dir, fold_idx)
    remote = f"experiments/{experiment_id}/folds/fold_{fold_idx}_1m.json"
    upload_to_supabase_storage(
        supabase_client,
        "artifacts",
        remote,
        gz_path,
        content_type="application/gzip",
    )
    return remote


# ===========================================================================
# Supabase Postgres inserts
# ===========================================================================

def insert_experiment(
    supabase_client: Any,
    experiment_id: str,
    dataset_id: str,
    name: str,
    model_family: str,
    horizon_min: int,
    is_noise_control: bool,
    config_json: dict,
    status: str = "running",
) -> dict:
    """Insert a row into the experiments table."""
    row = {
        "id": experiment_id,
        "dataset_id": dataset_id,
        "name": name,
        "model_family": model_family,
        "horizon_min": horizon_min,
        "is_noise_control": is_noise_control,
        "status": status,
        "config_json": config_json,
    }
    result = supabase_client.table("experiments").insert(row).execute()
    return result.data[0] if result.data else row


def update_experiment_completed(
    supabase_client: Any,
    experiment_id: str,
    summary_metrics: dict,
    autopsy_path: str | None = None,
    overview_path: str | None = None,
) -> None:
    """Update experiment status to 'completed' with summary metrics."""
    update = {
        "status": "completed",
        "summary_metrics_json": summary_metrics,
    }
    if autopsy_path:
        update["autopsy_storage_path"] = autopsy_path
    if overview_path:
        update["overview_storage_path"] = overview_path

    supabase_client.table("experiments").update(update).eq(
        "id", experiment_id
    ).execute()


def insert_fold_result(
    supabase_client: Any,
    experiment_id: str,
    fold_idx: int,
    metrics: dict,
    slice_storage_path: str,
) -> dict:
    """Insert a row into the fold_results table."""
    row = {
        "experiment_id": experiment_id,
        "fold_idx": fold_idx,
        "train_start_ms": metrics["train_start_ms"],
        "train_end_ms": metrics["train_end_ms"],
        "test_start_ms": metrics["test_start_ms"],
        "test_end_ms": metrics["test_end_ms"],
        "pinball_10": metrics["pinball_10"],
        "pinball_50": metrics["pinball_50"],
        "pinball_90": metrics["pinball_90"],
        "pinball_mean": metrics["pinball_mean"],
        "skill_vs_naive": metrics.get("skill_vs_naive"),
        "cov_10": metrics["cov_10"],
        "cov_80": metrics["cov_80"],
        "cov_90": metrics["cov_90"],
        "cond_dir_acc_conf10": metrics.get("cond_dir_acc_conf10"),
        "cond_dir_acc_tension10": metrics.get("cond_dir_acc_tension10"),
        "residual_acf1": metrics.get("residual_acf1"),
        "slice_storage_path": slice_storage_path,
        "metrics_json": metrics,
    }
    result = supabase_client.table("fold_results").insert(row).execute()
    return result.data[0] if result.data else row


# ===========================================================================
# Full export pipeline
# ===========================================================================

def export_experiment_results(
    supabase_client: Any,
    experiment_id: str,
    master_df: pl.DataFrame,
    fold_dfs: list[pl.DataFrame],
    fold_metrics: list[dict],
    summary_metrics: dict,
    cascade_timestamps: list[int] | None = None,
    preds_df: pl.DataFrame | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Full export pipeline: build payloads, upload to Supabase Storage,
    insert Postgres rows.

    Returns dict with storage paths.
    """
    if output_dir is None:
        output_dir = Path("/vol/exports") / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Any] = {"fold_paths": []}

    # 1. Overview
    overview = build_overview_1h(
        master_df,
        preds_df=preds_df,
        cascade_timestamps=cascade_timestamps,
    )
    overview_path = upload_overview(
        supabase_client, experiment_id, overview, output_dir
    )
    paths["overview_path"] = overview_path

    # 2. Per-fold payloads
    for idx, fold_df in enumerate(fold_dfs):
        payload = build_fold_payload(fold_df, idx)
        fold_path = upload_fold(
            supabase_client, experiment_id, payload, idx, output_dir
        )
        paths["fold_paths"].append(fold_path)

        # Insert fold result row
        insert_fold_result(
            supabase_client,
            experiment_id,
            idx,
            fold_metrics[idx],
            fold_path,
        )

    # 3. Update experiment status
    update_experiment_completed(
        supabase_client,
        experiment_id,
        summary_metrics,
        overview_path=overview_path,
    )

    return paths
