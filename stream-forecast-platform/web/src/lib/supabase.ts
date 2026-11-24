import { createClient } from "@supabase/supabase-js";

/* ------------------------------------------------------------------ */
/*  TypeScript types matching the Postgres schema (§10)               */
/* ------------------------------------------------------------------ */

export interface AuditReport {
  id: string;
  dataset_id: string;
  row_count: number;
  start_time_ms: number;
  end_time_ms: number;
  storage_path: string;
  summary_json: Record<string, unknown>;
  created_at: string;
}

export interface Experiment {
  id: string;
  dataset_id: string;
  name: string;
  model_family: string;
  horizon_min: number;
  is_noise_control: boolean;
  status: "running" | "completed" | "failed";
  config_json: Record<string, unknown>;
  summary_metrics_json: SummaryMetrics | null;
  autopsy_storage_path: string | null;
  overview_storage_path: string | null;
  created_at: string;
}

export interface SummaryMetrics {
  pinball_mean: number;
  skill_vs_naive: number;
  pct_folds_positive: number;
  cov_10: number;
  cov_80: number;
  cov_90: number;
  cond_dir_acc_conf10: number | null;
  cond_dir_acc_tension10: number | null;
}

export interface FoldResult {
  id: string;
  experiment_id: string;
  fold_idx: number;
  train_start_ms: number;
  train_end_ms: number;
  test_start_ms: number;
  test_end_ms: number;
  pinball_10: number;
  pinball_50: number;
  pinball_90: number;
  pinball_mean: number;
  skill_vs_naive: number | null;
  cov_10: number;
  cov_80: number;
  cov_90: number;
  cond_dir_acc_conf10: number | null;
  cond_dir_acc_tension10: number | null;
  residual_acf1: number | null;
  slice_storage_path: string;
  metrics_json: Record<string, unknown>;
}

/** Fold 1-minute slice payload (loaded on-demand from Supabase Storage) */
export interface FoldSlice {
  t: number[];
  x: number[];
  p10: number[];
  p50: number[];
  p90: number[];
  r0: number[];
  r1: number[];
  r2: number[];
  cascade: number[];
}

/** Overview 1-hour payload */
export interface OverviewPayload {
  t: number[];
  open: number[];
  high: number[];
  low: number[];
  close: number[];
  r0: number[];
  r1: number[];
  r2: number[];
  pinball_mean: number[];
  cascade_timestamps: number[];
}

/** Audit report full payload (from Storage JSON) */
export interface AuditPayload {
  gap_summary: {
    total_rows: number;
    date_span_days: number;
    total_gap_minutes: number;
    num_outages: number;
    longest_outage_minutes: number;
    tier2_5m_missing_pct: number;
    tier2_8h_missing_pct: number;
  };
  acf: {
    r: { lags: number[]; values: number[] };
    abs_r: { lags: number[]; values: number[] };
    r_sq: { lags: number[]; values: number[] };
    imbalance: { lags: number[]; values: number[] };
    n_obs: number;
  };
  ljung_box: Record<string, { lag: number; stat: number; p_value: number }[]>;
  welch_psd: {
    abs_r: { freqs: number[]; power: number[] };
    activity: { freqs: number[]; power: number[] };
  };
  tail_profile: {
    excess_kurtosis: number;
    skewness: number;
    prob_5sigma_empirical: number;
    prob_5sigma_gaussian: number;
    prob_10sigma_empirical: number;
    prob_10sigma_gaussian: number;
    tail_ratio: number;
  };
  heatmap: {
    metric: string;
    dow: number[];
    hour: number[];
    values: number[][];
  }[];
  lag_scatter: {
    r_t: number[];
    r_t1: number[];
    abs_r_t: number[];
    abs_r_t1: number[];
  };
}

/** Autopsy payload (from Storage JSON) */
export interface AutopsyPayload {
  residual_acf: { lags: number[]; values: number[][] }; // per-fold ACF arrays
  regime_metrics: {
    regime: number;
    pinball_mean: number;
    cov_10: number;
    cov_80: number;
    cov_90: number;
  }[];
  hourly_metrics: {
    hour: number;
    pinball_mean: number;
    cov_10: number;
    cov_80: number;
    cov_90: number;
  }[];
  feature_importance: {
    feature: string;
    mean_gain: number;
    std_gain: number;
  }[];
}

/* ------------------------------------------------------------------ */
/*  Database type map for Supabase generic client                     */
/* ------------------------------------------------------------------ */

export interface Database {
  public: {
    Tables: {
      audit_reports: { Row: AuditReport };
      experiments: { Row: Experiment };
      fold_results: { Row: FoldResult };
    };
  };
}

/* ------------------------------------------------------------------ */
/*  Read-only Supabase client                                         */
/* ------------------------------------------------------------------ */

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || "https://placeholder.supabase.co";
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || "placeholder-anon-key";

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

/* ------------------------------------------------------------------ */
/*  Helper: download JSON from Supabase Storage bucket "artifacts"    */
/* ------------------------------------------------------------------ */

export async function downloadArtifact<T>(path: string): Promise<T> {
  const { data, error } = await supabase.storage
    .from("artifacts")
    .download(path);
  if (error) throw error;
  const text = await data.text();
  return JSON.parse(text) as T;
}
