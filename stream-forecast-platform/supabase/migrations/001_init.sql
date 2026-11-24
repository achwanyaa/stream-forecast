create extension if not exists "pgcrypto";

create table if not exists audit_reports (
  id uuid primary key default gen_random_uuid(),
  dataset_id text not null,
  row_count bigint not null,
  start_time_ms bigint not null,
  end_time_ms bigint not null,
  storage_path text not null,          -- e.g., audits/{dataset_id}/audit_report.json
  summary_json jsonb not null,         -- compact KPIs for fast listing
  created_at timestamptz not null default now()
);

create table if not exists experiments (
  id uuid primary key default gen_random_uuid(),
  dataset_id text not null,
  name text not null,
  model_family text not null,          -- naive_vol_scaled | ms_ar | chronos_zeroshot | lgbm_quantile | lgbm_regime_moe | noise_control
  horizon_min int not null,            -- 1 | 5 | 15 | 60
  is_noise_control boolean not null default false,
  status text not null default 'running', -- running | completed | failed
  config_json jsonb not null,
  summary_metrics_json jsonb,          -- aggregated pinball, skill_vs_naive, pct_folds_positive, calibration, cond_hit_rate
  autopsy_storage_path text,           -- e.g., experiments/{id}/autopsy.json
  overview_storage_path text,          -- e.g., experiments/{id}/overview_1h.json (downsampled 12-24m overview)
  created_at timestamptz not null default now()
);

create table if not exists fold_results (
  id uuid primary key default gen_random_uuid(),
  experiment_id uuid not null references experiments(id) on delete cascade,
  fold_idx int not null,
  train_start_ms bigint not null,
  train_end_ms bigint not null,
  test_start_ms bigint not null,
  test_end_ms bigint not null,
  pinball_10 double precision not null,
  pinball_50 double precision not null,
  pinball_90 double precision not null,
  pinball_mean double precision not null,
  skill_vs_naive double precision,
  cov_10 double precision not null,
  cov_80 double precision not null,
  cov_90 double precision not null,
  cond_dir_acc_conf10 double precision,
  cond_dir_acc_tension10 double precision,
  residual_acf1 double precision,
  slice_storage_path text not null,    -- e.g., experiments/{id}/folds/fold_{idx}_1m.json (gzipped)
  metrics_json jsonb not null,
  unique (experiment_id, fold_idx)
);

create index idx_experiments_dataset_horizon on experiments(dataset_id, horizon_min);
create index idx_fold_results_exp on fold_results(experiment_id, fold_idx);
