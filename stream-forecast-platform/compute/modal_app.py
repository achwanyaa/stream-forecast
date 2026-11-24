"""Modal compute orchestration: images, volume, secrets, and CPU/GPU entrypoints.

All definitions sourced from SYSTEM_SPEC.md §9.
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# App, Volume & Secret
# ---------------------------------------------------------------------------
app = modal.App("stream-forecast-platform")
vol = modal.Volume.from_name("stream-forecast-vol", create_if_missing=True)
supabase_secret = modal.Secret.from_name("supabase-creds")

# ---------------------------------------------------------------------------
# CPU Image — all non-GPU Python work
# ---------------------------------------------------------------------------
cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "polars==1.14.0",
        "pyarrow==18.0.0",
        "numpy==1.26.4",
        "scipy==1.14.1",
        "numba==0.60.0",
        "scikit-learn==1.5.2",
        "hmmlearn==0.3.3",
        "statsmodels==0.14.4",
        "lightgbm==4.5.0",
        "supabase==2.10.0",
        "pydantic==2.9.2",
    )
    .add_local_python_source("pipeline", "config")
)

# ---------------------------------------------------------------------------
# GPU Image — Chronos foundation model inference only
# ---------------------------------------------------------------------------
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "chronos-forecasting>=1.4.0",
        "polars==1.14.0",
        "pyarrow==18.0.0",
        "numpy==1.26.4",
        "supabase==2.10.0",
    )
    .add_local_python_source("pipeline", "config")
)

# ---------------------------------------------------------------------------
# CPU Functions
# ---------------------------------------------------------------------------


@app.function(
    image=cpu_image,
    cpu=4,
    memory=16384,
    timeout=1800,
    volumes={"/vol": vol},
    secrets=[supabase_secret],
)
def ingest_dataset(dataset_id: str) -> None:
    """Validate raw files, run alignment + gap masking, compute Layer 2
    features, save master.parquet and features.parquet, then commit volume."""
    # Implementation in pipeline/ingest.py & pipeline/features.py
    from pipeline.ingest import run_ingest  # noqa: F811

    run_ingest(dataset_id)
    vol.commit()


@app.function(
    image=cpu_image,
    cpu=4,
    memory=16384,
    timeout=1800,
    volumes={"/vol": vol},
    secrets=[supabase_secret],
)
def run_audit(dataset_id: str) -> None:
    """Run Phase 1 randomness & structure audit. Upload audit_report.json to
    Supabase Storage and insert row into audit_reports."""
    from pipeline.audit import run_phase1_audit

    run_phase1_audit(dataset_id)


@app.function(
    image=cpu_image,
    cpu=4,
    memory=16384,
    timeout=1800,
    volumes={"/vol": vol},
    secrets=[supabase_secret],
)
def run_cpu_fold(
    experiment_id: str,
    dataset_id: str,
    fold_spec: dict,
    model_family: str,
    horizon: int,
    config: dict,
) -> dict:
    """Execute a single CPU fold: fit Layer 3 + Layer 1/4 model on training
    slice, predict test slice, save fold predictions Parquet, return fold
    metrics dict."""
    from pipeline.models import execute_cpu_fold

    return execute_cpu_fold(
        experiment_id=experiment_id,
        dataset_id=dataset_id,
        fold_spec=fold_spec,
        model_family=model_family,
        horizon=horizon,
        config=config,
    )


import os

_gpu_type = "T4" if os.environ.get("MODAL_ENABLE_GPU") == "1" else None

@app.function(
    image=gpu_image,
    gpu=_gpu_type,
    memory=16384,
    timeout=1800,
    volumes={"/vol": vol},
    secrets=[supabase_secret],
)
def run_chronos_fold(
    experiment_id: str,
    dataset_id: str,
    fold_spec: dict,
    horizon: int,
) -> dict:
    """Run strided zero-shot Chronos inference on a single test fold and save
    predictions Parquet. Returns fold metrics dict."""
    from pipeline.baselines import execute_chronos_fold

    return execute_chronos_fold(
        experiment_id=experiment_id,
        dataset_id=dataset_id,
        fold_spec=fold_spec,
        horizon=horizon,
    )


# ---------------------------------------------------------------------------
# Local Entrypoint — orchestrates full experiment
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def launch_experiment(
    dataset_id: str,
    model_family: str,
    horizons: str = "1,5,15,60",
    config: str = "{}",
) -> None:
    """Create experiment rows in Supabase (status='running'), generate fold
    specs, fan out all folds in parallel using .starmap(), compute aggregate
    metrics + residual autopsy + chart-ready downsampled JSON slices, upload
    to Supabase Storage, and update status='completed'."""
    import json
    from pipeline.export import orchestrate_experiment

    parsed_horizons = [int(h.strip()) for h in horizons.split(",") if h.strip()] if isinstance(horizons, str) else horizons
    parsed_config = json.loads(config) if isinstance(config, str) else config

    orchestrate_experiment(
        dataset_id=dataset_id,
        model_family=model_family,
        horizons=parsed_horizons,
        config=parsed_config,
    )


# ---------------------------------------------------------------------------
# Low-Compute Smoke Test Entrypoint (<$0.01 on Modal)
# ---------------------------------------------------------------------------


@app.function(
    image=cpu_image,
    cpu=2,
    memory=4096,
    timeout=300,
    volumes={"/vol": vol},
    secrets=[supabase_secret],
)
def run_cloud_smoke_test(n_minutes: int = 1440) -> dict:
    """Ultra-low compute end-to-end smoke test on Modal.
    Estimated compute cost: ~$0.002 (<$0.01).
    1. Generates synthetic multi-rate raw streams (1m, 5m, 8h).
    2. Executes zero-leakage ingestion, gap masking & Layer 2 feature engine.
    3. Executes Phase 1 randomness & structure audit.
    4. Connects to Supabase, uploads audit_report.json to Storage, and inserts row.
    5. Returns timing, record counts, and cost breakdown.
    """
    import os
    import time
    import tempfile
    import json
    import numpy as np
    import polars as pl
    from supabase import create_client

    t_start = time.time()
    print(f"🚀 Starting cloud smoke test on Modal with {n_minutes} minutes of data...")

    # Step 1: Generate synthetic streams
    base_time = 1_700_000_000_000
    rng = np.random.RandomState(42)

    close_1m = np.arange(base_time + 60_000, base_time + 60_000 + n_minutes * 60_000, 60_000, dtype=np.int64)
    returns = rng.normal(0, 0.001, n_minutes)
    prices = np.exp(np.cumsum(returns) + np.log(100.0))

    tier1 = pl.DataFrame({
        "open_time_ms": close_1m - 60_000,
        "close_time_ms": close_1m,
        "x_open": prices * (1.0 - rng.uniform(0, 0.001, n_minutes)),
        "x_high": prices * (1.0 + rng.uniform(0, 0.002, n_minutes)),
        "x_low": prices * (1.0 - rng.uniform(0, 0.002, n_minutes)),
        "x_close": prices,
        "activity_mag": rng.exponential(100.0, n_minutes),
        "event_count": rng.poisson(50, n_minutes).astype(np.int64),
        "aggression_up": rng.exponential(50.0, n_minutes),
        "aggression_down": rng.exponential(50.0, n_minutes),
    })

    n_5m = max(1, n_minutes // 5)
    close_5m = np.arange(base_time + 300_000, base_time + 300_000 + n_5m * 300_000, 300_000, dtype=np.int64)
    tier2_5m = pl.DataFrame({
        "open_time_ms": close_5m - 300_000,
        "close_time_ms": close_5m,
        "commitment": rng.exponential(1_000_000.0, n_5m),
        "skew_general": rng.normal(0.0, 0.1, n_5m),
        "skew_top20": rng.normal(0.05, 0.15, n_5m),
    })

    n_8h = max(1, n_minutes // 480)
    close_8h = np.arange(base_time + 28_800_000, base_time + 28_800_000 + n_8h * 28_800_000, 28_800_000, dtype=np.int64)
    tier2_8h = pl.DataFrame({
        "open_time_ms": close_8h - 28_800_000,
        "close_time_ms": close_8h,
        "crowding_penalty": rng.normal(0.0001, 0.0005, n_8h),
    })

    # Step 2: Ingest & align
    from pipeline.ingest import ingest_and_align
    with tempfile.TemporaryDirectory() as tmpdir:
        t1_path = os.path.join(tmpdir, "tier1.parquet")
        t2_5m_path = os.path.join(tmpdir, "tier2_5m.parquet")
        t2_8h_path = os.path.join(tmpdir, "tier2_8h.parquet")
        master_path = os.path.join(tmpdir, "master.parquet")

        tier1.write_parquet(t1_path)
        tier2_5m.write_parquet(t2_5m_path)
        tier2_8h.write_parquet(t2_8h_path)

        master_df = ingest_and_align(t1_path, t2_5m_path, t2_8h_path, master_path)
        print(f"✓ Alignment complete: {master_df.shape[0]} rows aligned.")

        # Step 3: Compute Layer 2 features
        from pipeline.features import compute_all_features
        feat_df = compute_all_features(master_df)
        print(f"✓ Feature engine complete: {len(feat_df.columns)} columns generated.")

        # Step 4: Run Phase 1 audit
        from pipeline.audit import run_audit
        audit_res = run_audit(master_path)
        tail_prof = audit_res.get("tail_profile", {})
        kurt = float(tail_prof.get("excess_kurtosis") or 0.0)
        tail_ratio = float(tail_prof.get("abs_q01_over_q99") or 1.0)
        print(f"✓ Phase 1 audit complete. Excess Kurtosis: {kurt:.2f}")

        # Step 5: Test Supabase connection & upload
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")
        uploaded = False
        if supabase_url and supabase_key:
            client = create_client(supabase_url, supabase_key)
            audit_bytes = json.dumps(audit_res).encode("utf-8")
            storage_path = f"audits/smoke-test/audit_report_{int(time.time())}.json"
            client.storage.from_("artifacts").upload(
                path=storage_path,
                file=audit_bytes,
                file_options={"content-type": "application/json", "upsert": "true"},
            )
            print(f"✓ Uploaded audit report to Supabase Storage: {storage_path}")

            # Insert DB row
            row = {
                "dataset_id": "smoke-test",
                "row_count": master_df.shape[0],
                "start_time_ms": int(master_df["close_time_ms"].min()),
                "end_time_ms": int(master_df["close_time_ms"].max()),
                "storage_path": storage_path,
                "summary_json": {
                    "excess_kurtosis": kurt,
                    "tail_ratio": tail_ratio,
                    "total_gap_minutes": audit_res.get("gap_summary", {}).get("total_gap_minutes", 0),
                },
            }
            client.table("audit_reports").insert(row).execute()
            print("✓ Inserted record into Supabase audit_reports table.")
            uploaded = True
        else:
            print("⚠️ Supabase credentials missing from environment.")

    elapsed = time.time() - t_start
    cost_est = elapsed * 2 * (0.108 / 3600)  # Modal 2-CPU rate ~$0.108/core-hr
    return {
        "status": "PASS",
        "rows_processed": n_minutes,
        "features_computed": len(feat_df.columns),
        "excess_kurtosis": round(kurt, 2),
        "supabase_connected": uploaded,
        "duration_seconds": round(elapsed, 2),
        "estimated_compute_cost_usd": f"${cost_est:.4f}",
    }


@app.local_entrypoint()
def smoke_test(n_minutes: int = 1440) -> None:
    """Run low-compute smoke test (<$0.01) on Modal and print verification results."""
    res = run_cloud_smoke_test.remote(n_minutes=n_minutes)
    print("\n" + "=" * 60)
    print("MODAL CLOUD SMOKE TEST RESULTS:")
    print("=" * 60)
    for k, v in res.items():
        print(f"  {k}: {v}")
    print("=" * 60)
