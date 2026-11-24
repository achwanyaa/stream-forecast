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


@app.function(
    image=gpu_image,
    gpu="T4",
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
    horizons: list[int],
    config: dict,
) -> None:
    """Create experiment rows in Supabase (status='running'), generate fold
    specs, fan out all folds in parallel using .starmap(), compute aggregate
    metrics + residual autopsy + chart-ready downsampled JSON slices, upload
    to Supabase Storage, and update status='completed'."""
    from pipeline.export import orchestrate_experiment

    orchestrate_experiment(
        dataset_id=dataset_id,
        model_family=model_family,
        horizons=horizons,
        config=config,
    )
