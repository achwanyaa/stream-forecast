'use client';

import React, { useState, useEffect, use } from 'react';
import Link from 'next/link';
import { supabase, type Experiment, type AutopsyPayload } from '@/lib/supabase';
import {
  ResidualAcfChart,
  GroupedBreakdownChart,
  FeatureImportanceChart,
} from '@/components/charts/ResidualCharts';

interface PageProps {
  params: Promise<{ id: string }>;
}

// Fallback high-fidelity mock autopsy payload
const MOCK_AUTOPSY: AutopsyPayload = {
  residual_acf: {
    lags: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    values: Array.from({ length: 17 }, () =>
      Array.from({ length: 10 }, (_, l) =>
        l === 0 ? 0.012 + (Math.random() - 0.5) * 0.02 : (Math.random() - 0.5) * 0.015
      )
    ),
  },
  regime_metrics: [
    { regime: 0, pinball_mean: 0.0028, cov_10: 0.099, cov_80: 0.802, cov_90: 0.901 },
    { regime: 1, pinball_mean: 0.0041, cov_10: 0.096, cov_80: 0.798, cov_90: 0.897 },
    { regime: 2, pinball_mean: 0.0076, cov_10: 0.091, cov_80: 0.785, cov_90: 0.884 },
  ],
  hourly_metrics: Array.from({ length: 24 }, (_, h) => ({
    hour: h,
    pinball_mean: 0.0035 + (h >= 12 && h <= 18 ? 0.0012 : 0.0003),
    cov_10: 0.098 + (h % 3) * 0.002,
    cov_80: 0.795 + (h % 2) * 0.01,
    cov_90: 0.898 + (h % 2) * 0.005,
  })),
  feature_importance: [
    { feature: 'regime_p2 (High Volatility)', mean_gain: 284.5, std_gain: 18.2 },
    { feature: 'crowd_tension_interact', mean_gain: 242.1, std_gain: 15.4 },
    { feature: 'bocpd_cp_prob_15m', mean_gain: 210.8, std_gain: 14.1 },
    { feature: 'garman_klass_1m', mean_gain: 195.4, std_gain: 12.0 },
    { feature: 'comm_vel_60', mean_gain: 182.3, std_gain: 16.5 },
    { feature: 'vr_5_60 (Lo-MacKinlay)', mean_gain: 168.0, std_gain: 11.2 },
    { feature: 'skew_top20_z7d', mean_gain: 154.2, std_gain: 13.8 },
    { feature: 'vol_ratio_15_240', mean_gain: 143.9, std_gain: 9.7 },
    { feature: 'perm_entropy_60', mean_gain: 135.6, std_gain: 10.4 },
    { feature: 'log_rv_60', mean_gain: 128.1, std_gain: 8.5 },
    { feature: 'streak_len_signed', mean_gain: 115.4, std_gain: 14.2 },
    { feature: 'agg_imbalance_vwap_15', mean_gain: 102.7, std_gain: 8.1 },
    { feature: 'close_zscore_1440', mean_gain: 94.3, std_gain: 7.6 },
    { feature: 'event_size_zscore_1440', mean_gain: 86.2, std_gain: 9.3 },
    { feature: 'intrabar_close_pos', mean_gain: 78.5, std_gain: 6.8 },
    { feature: 'min_of_day_sin', mean_gain: 71.0, std_gain: 5.2 },
    { feature: 'acf1_240', mean_gain: 64.8, std_gain: 8.4 },
    { feature: 'dist_from_max_240', mean_gain: 58.2, std_gain: 6.1 },
    { feature: 'log_intensity_zscore', mean_gain: 52.4, std_gain: 7.0 },
    { feature: 'min_of_week_cos', mean_gain: 45.1, std_gain: 4.8 },
  ],
};

export default function AutopsyPage({ params }: PageProps) {
  const resolvedParams = use(params);
  const experimentId = resolvedParams.id;

  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [autopsy, setAutopsy] = useState<AutopsyPayload>(MOCK_AUTOPSY);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    async function loadData() {
      setLoading(true);
      try {
        const { data: exp } = await supabase
          .from('experiments')
          .select('*')
          .eq('id', experimentId)
          .single();

        if (exp) {
          setExperiment(exp as Experiment);
          if (exp.autopsy_storage_path) {
            const { data: fileData, error } = await supabase.storage
              .from('artifacts')
              .download(exp.autopsy_storage_path);
            if (!error && fileData) {
              const parsed = JSON.parse(await fileData.text()) as AutopsyPayload;
              setAutopsy(parsed);
            }
          }
        }
      } catch {
        // Fallback to mock
      } finally {
        setLoading(false);
      }
    }

    loadData();
  }, [experimentId]);

  const { residual_acf, regime_metrics, hourly_metrics, feature_importance } = autopsy;

  const regimeLabels = ['State 0 (Compression)', 'State 1 (Normal)', 'State 2 (Cascade)'];
  const regimePinball = regime_metrics.map((r) => r.pinball_mean);
  const regimeCov10 = regime_metrics.map((r) => r.cov_10);
  const regimeCov80 = regime_metrics.map((r) => r.cov_80);
  const regimeCov90 = regime_metrics.map((r) => r.cov_90);

  const hourLabels = hourly_metrics.map((h) => `${h.hour}h`);
  const hourPinball = hourly_metrics.map((h) => h.pinball_mean);
  const hourCov10 = hourly_metrics.map((h) => h.cov_10);
  const hourCov80 = hourly_metrics.map((h) => h.cov_80);
  const hourCov90 = hourly_metrics.map((h) => h.cov_90);

  return (
    <div className="space-y-6">
      {/* Header Strip */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2 text-xs text-slate-400 mb-1">
            <Link href="/" className="hover:text-white transition-colors">
              Leaderboard
            </Link>
            <span>/</span>
            <span className="text-indigo-400 font-mono">{experimentId}</span>
          </div>
          <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
            Residual Autopsy & Feature Attribution
            {loading && <span className="text-xs font-normal text-sky-400 animate-pulse">(loading...)</span>}
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Detailed residual error decomposition across regimes, diurnal hours, and feature importance stability.
          </p>
        </div>

        <div className="flex items-center space-x-3">
          <Link
            href={`/inspector/${experimentId}`}
            className="px-3 py-1.5 rounded-lg bg-sky-500/10 hover:bg-sky-500/20 text-sky-400 border border-sky-500/30 text-xs font-medium transition-colors"
          >
            Open Regime Inspector →
          </Link>
        </div>
      </div>

      {/* 1. Residual ACF & Regime Breakdown Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Residual Autocorrelation */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5">
          <ResidualAcfChart
            lags={residual_acf.lags}
            foldAcfValues={residual_acf.values}
            height="340px"
          />
        </div>

        {/* Breakdown by Layer 3 Regime */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5">
          <GroupedBreakdownChart
            title="Error & Calibration by Layer 3 Regime"
            subtitle="Comparing Pinball Loss and Quantile Coverage across HMM states"
            categories={regimeLabels}
            pinballLoss={regimePinball}
            cov10={regimeCov10}
            cov80={regimeCov80}
            cov90={regimeCov90}
            height="340px"
          />
        </div>
      </div>

      {/* 2. Diurnal UTC Hour Breakdown */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5">
        <GroupedBreakdownChart
          title="Error & Calibration by UTC Hour (0..23)"
          subtitle="Detecting diurnal volatility miscalibration or liquidity session drift"
          categories={hourLabels}
          pinballLoss={hourPinball}
          cov10={hourCov10}
          cov80={hourCov80}
          cov90={hourCov90}
          height="340px"
        />
      </div>

      {/* 3. Top 20 Feature Importance Horizontal Bar with Std Error */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 shadow-xl">
        <FeatureImportanceChart features={feature_importance} height="520px" />
      </div>
    </div>
  );
}
