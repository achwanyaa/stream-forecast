'use client';

import React, { useState, useEffect, useMemo } from 'react';
import Link from 'next/link';
import { supabase, type Experiment, type FoldResult } from '@/lib/supabase';
import SkillStabilityChart, { type ModelSkillSeries } from '@/components/charts/SkillStabilityChart';

// Fallback mock experiments if Supabase is empty
const MOCK_EXPERIMENTS: (Experiment & { foldResults: FoldResult[] })[] = [
  {
    id: 'exp-moe-60m',
    dataset_id: 'stream-alpha-24m',
    name: 'Regime MoE (Quantile LGBM)',
    model_family: 'lgbm_regime_moe',
    horizon_min: 60,
    is_noise_control: false,
    status: 'completed',
    config_json: { num_leaves: 31, lr: 0.025 },
    summary_metrics_json: {
      pinball_mean: 0.00412,
      skill_vs_naive: 0.052,
      pct_folds_positive: 88.2,
      cov_10: 0.098,
      cov_80: 0.795,
      cov_90: 0.899,
      cond_dir_acc_conf10: 0.638,
      cond_dir_acc_tension10: 0.612,
    },
    autopsy_storage_path: 'experiments/exp-moe-60m/autopsy.json',
    overview_storage_path: 'experiments/exp-moe-60m/overview_1h.json',
    created_at: new Date().toISOString(),
    foldResults: Array.from({ length: 17 }, (_, i) => ({
      id: `f-${i}`,
      experiment_id: 'exp-moe-60m',
      fold_idx: i,
      train_start_ms: 1700000000000 + i * 21600 * 60000,
      train_end_ms: 1700000000000 + (i + 6) * 21600 * 60000,
      test_start_ms: 1700000000000 + (i + 6) * 21600 * 60000 + 1440 * 60000,
      test_end_ms: 1700000000000 + (i + 7) * 21600 * 60000,
      pinball_10: 0.0022,
      pinball_50: 0.0051,
      pinball_90: 0.0050,
      pinball_mean: 0.0041,
      skill_vs_naive: 0.04 + (Math.sin(i * 0.7) * 0.03),
      cov_10: 0.095 + (i % 3) * 0.002,
      cov_80: 0.79 + (i % 2) * 0.01,
      cov_90: 0.89 + (i % 2) * 0.01,
      cond_dir_acc_conf10: 0.62 + (i % 4) * 0.01,
      cond_dir_acc_tension10: 0.60 + (i % 3) * 0.01,
      residual_acf1: 0.012,
      slice_storage_path: `experiments/exp-moe-60m/folds/fold_${i}_1m.json`,
      metrics_json: {},
    })),
  },
  {
    id: 'exp-lgbm-global-60m',
    dataset_id: 'stream-alpha-24m',
    name: 'Global Quantile LGBM',
    model_family: 'lgbm_quantile',
    horizon_min: 60,
    is_noise_control: false,
    status: 'completed',
    config_json: { num_leaves: 31, lr: 0.025 },
    summary_metrics_json: {
      pinball_mean: 0.00421,
      skill_vs_naive: 0.038,
      pct_folds_positive: 82.4,
      cov_10: 0.092,
      cov_80: 0.805,
      cov_90: 0.902,
      cond_dir_acc_conf10: 0.605,
      cond_dir_acc_tension10: 0.584,
    },
    autopsy_storage_path: 'experiments/exp-lgbm-global-60m/autopsy.json',
    overview_storage_path: 'experiments/exp-lgbm-global-60m/overview_1h.json',
    created_at: new Date().toISOString(),
    foldResults: Array.from({ length: 17 }, (_, i) => ({
      id: `f-lgbm-${i}`,
      experiment_id: 'exp-lgbm-global-60m',
      fold_idx: i,
      train_start_ms: 1700000000000 + i * 21600 * 60000,
      train_end_ms: 1700000000000 + (i + 6) * 21600 * 60000,
      test_start_ms: 1700000000000 + (i + 6) * 21600 * 60000 + 1440 * 60000,
      test_end_ms: 1700000000000 + (i + 7) * 21600 * 60000,
      pinball_10: 0.0023,
      pinball_50: 0.0052,
      pinball_90: 0.0051,
      pinball_mean: 0.0042,
      skill_vs_naive: 0.03 + (Math.cos(i * 0.6) * 0.025),
      cov_10: 0.093,
      cov_80: 0.801,
      cov_90: 0.898,
      cond_dir_acc_conf10: 0.59,
      cond_dir_acc_tension10: 0.57,
      residual_acf1: 0.015,
      slice_storage_path: `experiments/exp-lgbm-global-60m/folds/fold_${i}_1m.json`,
      metrics_json: {},
    })),
  },
  {
    id: 'exp-chronos-60m',
    dataset_id: 'stream-alpha-24m',
    name: 'Chronos Foundation Zero-Shot',
    model_family: 'chronos_zeroshot',
    horizon_min: 60,
    is_noise_control: false,
    status: 'completed',
    config_json: { model: 'amazon/chronos-bolt-base' },
    summary_metrics_json: {
      pinball_mean: 0.00435,
      skill_vs_naive: 0.018,
      pct_folds_positive: 64.7,
      cov_10: 0.115,
      cov_80: 0.772,
      cov_90: 0.887,
      cond_dir_acc_conf10: 0.542,
      cond_dir_acc_tension10: 0.531,
    },
    autopsy_storage_path: 'experiments/exp-chronos-60m/autopsy.json',
    overview_storage_path: 'experiments/exp-chronos-60m/overview_1h.json',
    created_at: new Date().toISOString(),
    foldResults: Array.from({ length: 17 }, (_, i) => ({
      id: `f-chronos-${i}`,
      experiment_id: 'exp-chronos-60m',
      fold_idx: i,
      train_start_ms: 1700000000000 + i * 21600 * 60000,
      train_end_ms: 1700000000000 + (i + 6) * 21600 * 60000,
      test_start_ms: 1700000000000 + (i + 6) * 21600 * 60000 + 1440 * 60000,
      test_end_ms: 1700000000000 + (i + 7) * 21600 * 60000,
      pinball_10: 0.0024,
      pinball_50: 0.0054,
      pinball_90: 0.0052,
      pinball_mean: 0.0043,
      skill_vs_naive: 0.015 + (Math.sin(i * 0.9) * 0.018),
      cov_10: 0.112,
      cov_80: 0.775,
      cov_90: 0.885,
      cond_dir_acc_conf10: 0.53,
      cond_dir_acc_tension10: 0.52,
      residual_acf1: 0.025,
      slice_storage_path: `experiments/exp-chronos-60m/folds/fold_${i}_1m.json`,
      metrics_json: {},
    })),
  },
  {
    id: 'exp-noise-control-60m',
    dataset_id: 'stream-alpha-24m',
    name: 'Noise Control Group (Null Hypothesis)',
    model_family: 'noise_control',
    horizon_min: 60,
    is_noise_control: true,
    status: 'completed',
    config_json: { seeds: [1, 2, 3, 4, 5] },
    summary_metrics_json: {
      pinball_mean: 0.00445,
      skill_vs_naive: -0.005,
      pct_folds_positive: 11.8,
      cov_10: 0.101,
      cov_80: 0.798,
      cov_90: 0.899,
      cond_dir_acc_conf10: 0.498,
      cond_dir_acc_tension10: 0.502,
    },
    autopsy_storage_path: null,
    overview_storage_path: null,
    created_at: new Date().toISOString(),
    foldResults: Array.from({ length: 17 }, (_, i) => ({
      id: `f-noise-${i}`,
      experiment_id: 'exp-noise-control-60m',
      fold_idx: i,
      train_start_ms: 1700000000000 + i * 21600 * 60000,
      train_end_ms: 1700000000000 + (i + 6) * 21600 * 60000,
      test_start_ms: 1700000000000 + (i + 6) * 21600 * 60000 + 1440 * 60000,
      test_end_ms: 1700000000000 + (i + 7) * 21600 * 60000,
      pinball_10: 0.0025,
      pinball_50: 0.0055,
      pinball_90: 0.0054,
      pinball_mean: 0.0045,
      skill_vs_naive: -0.01 + (Math.sin(i * 1.5) * 0.012),
      cov_10: 0.10,
      cov_80: 0.80,
      cov_90: 0.90,
      cond_dir_acc_conf10: 0.50,
      cond_dir_acc_tension10: 0.50,
      residual_acf1: 0.001,
      slice_storage_path: '',
      metrics_json: {},
    })),
  },
];

export default function LeaderboardPage() {
  const [datasetId, setDatasetId] = useState<string>('stream-alpha-24m');
  const [horizonMin, setHorizonMin] = useState<number>(60);
  const [showNoiseControl, setShowNoiseControl] = useState<boolean>(true);
  const [experiments, setExperiments] = useState<typeof MOCK_EXPERIMENTS>(MOCK_EXPERIMENTS);
  const [sortField, setSortField] = useState<string>('skill_vs_naive');
  const [sortAsc, setSortAsc] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(false);

  // Fetch experiments from Supabase with fallback to mock data
  useEffect(() => {
    async function loadData() {
      setLoading(true);
      try {
        const { data: exps, error } = await supabase
          .from('experiments')
          .select('*')
          .eq('dataset_id', datasetId)
          .eq('horizon_min', horizonMin);

        if (!error && exps && exps.length > 0) {
          const loadedExps = await Promise.all(
            exps.map(async (exp) => {
              const { data: folds } = await supabase
                .from('fold_results')
                .select('*')
                .eq('experiment_id', exp.id)
                .order('fold_idx', { ascending: true });
              return {
                ...exp,
                foldResults: folds || [],
              };
            })
          );
          setExperiments(loadedExps as unknown as typeof MOCK_EXPERIMENTS);
        } else {
          // Fallback to mock data matching horizon
          setExperiments(
            MOCK_EXPERIMENTS.map((e) => ({
              ...e,
              horizon_min: horizonMin,
            }))
          );
        }
      } catch {
        // Fallback to mock on connection error
        setExperiments(
          MOCK_EXPERIMENTS.map((e) => ({
            ...e,
            horizon_min: horizonMin,
          }))
        );
      } finally {
        setLoading(false);
      }
    }

    loadData();
  }, [datasetId, horizonMin]);

  // Filtered and sorted experiments
  const filteredExperiments = useMemo(() => {
    return experiments
      .filter((e) => showNoiseControl || !e.is_noise_control)
      .sort((a, b) => {
        let valA = 0;
        let valB = 0;
        if (sortField === 'skill_vs_naive') {
          valA = a.summary_metrics_json?.skill_vs_naive ?? -Infinity;
          valB = b.summary_metrics_json?.skill_vs_naive ?? -Infinity;
        } else if (sortField === 'pinball_mean') {
          valA = a.summary_metrics_json?.pinball_mean ?? Infinity;
          valB = b.summary_metrics_json?.pinball_mean ?? Infinity;
        } else if (sortField === 'pct_folds_positive') {
          valA = a.summary_metrics_json?.pct_folds_positive ?? -Infinity;
          valB = b.summary_metrics_json?.pct_folds_positive ?? -Infinity;
        } else if (sortField === 'cond_dir_acc_conf10') {
          valA = a.summary_metrics_json?.cond_dir_acc_conf10 ?? -Infinity;
          valB = b.summary_metrics_json?.cond_dir_acc_conf10 ?? -Infinity;
        }
        return sortAsc ? valA - valB : valB - valA;
      });
  }, [experiments, showNoiseControl, sortField, sortAsc]);

  // Prepare chart series
  const chartFolds = useMemo(() => {
    const maxFolds = Math.max(...experiments.map((e) => e.foldResults.length), 17);
    return Array.from({ length: maxFolds }, (_, i) => i);
  }, [experiments]);

  const chartSeries = useMemo<ModelSkillSeries[]>(() => {
    return experiments.map((exp) => ({
      id: exp.id,
      name: exp.name,
      modelFamily: exp.model_family,
      isNoiseControl: exp.is_noise_control,
      skillData: exp.foldResults.map((f) => f.skill_vs_naive),
    }));
  }, [experiments]);

  const handleSort = (field: string) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header and Filter Bar */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 sm:p-5 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
            Walk-Forward Experiment Leaderboard
            {loading && <span className="text-xs font-normal text-sky-400 animate-pulse">(updating...)</span>}
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Strict out-of-sample evaluation across non-overlapping walk-forward folds. Scored against causal Naive baseline.
          </p>
        </div>

        {/* Filter Controls */}
        <div className="flex flex-wrap items-center gap-3 text-sm">
          {/* Dataset Dropdown */}
          <div className="flex items-center space-x-2">
            <label className="text-xs text-slate-400 font-medium">Dataset:</label>
            <select
              value={datasetId}
              onChange={(e) => setDatasetId(e.target.value)}
              className="bg-slate-950 border border-slate-700 text-slate-200 text-xs rounded-lg px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-sky-500"
            >
              <option value="stream-alpha-24m">stream-alpha-24m (1.05M rows)</option>
              <option value="stream-beta-12m">stream-beta-12m (525k rows)</option>
            </select>
          </div>

          {/* Horizon Tabs */}
          <div className="flex items-center bg-slate-950 p-1 rounded-lg border border-slate-800">
            {[1, 5, 15, 60].map((h) => (
              <button
                key={h}
                onClick={() => setHorizonMin(h)}
                className={`px-3 py-1 rounded text-xs font-medium transition-all ${
                  horizonMin === h
                    ? 'bg-sky-500 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {h}m
              </button>
            ))}
          </div>

          {/* Noise Control Toggle */}
          <label className="flex items-center space-x-2 cursor-pointer select-none bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800">
            <input
              type="checkbox"
              checked={showNoiseControl}
              onChange={(e) => setShowNoiseControl(e.target.checked)}
              className="rounded bg-slate-900 border-slate-700 text-sky-500 focus:ring-0 focus:ring-offset-0"
            />
            <span className="text-xs text-slate-300">Noise Control (95% Band)</span>
          </label>
        </div>
      </div>

      {/* Comparison Table */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl overflow-hidden shadow-lg">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-slate-800 bg-slate-950/60 text-slate-400 uppercase tracking-wider text-[11px]">
                <th className="py-3 px-4 font-semibold">Model Name</th>
                <th className="py-3 px-4 font-semibold">Family</th>
                <th
                  className="py-3 px-4 font-semibold cursor-pointer hover:text-white"
                  onClick={() => handleSort('pinball_mean')}
                >
                  Mean Pinball {sortField === 'pinball_mean' && (sortAsc ? '▲' : '▼')}
                </th>
                <th
                  className="py-3 px-4 font-semibold cursor-pointer hover:text-white text-right"
                  onClick={() => handleSort('skill_vs_naive')}
                >
                  Skill vs. Naive {sortField === 'skill_vs_naive' && (sortAsc ? '▲' : '▼')}
                </th>
                <th
                  className="py-3 px-4 font-semibold cursor-pointer hover:text-white text-right"
                  onClick={() => handleSort('pct_folds_positive')}
                >
                  % Folds Beating Naive {sortField === 'pct_folds_positive' && (sortAsc ? '▲' : '▼')}
                </th>
                <th className="py-3 px-4 font-semibold text-center">
                  Calibration (cov_10 / cov_80 / cov_90)
                </th>
                <th
                  className="py-3 px-4 font-semibold cursor-pointer hover:text-white text-right"
                  onClick={() => handleSort('cond_dir_acc_conf10')}
                >
                  Hit Rate (Conf10 / Tension10) {sortField === 'cond_dir_acc_conf10' && (sortAsc ? '▲' : '▼')}
                </th>
                <th className="py-3 px-4 font-semibold text-center">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono text-slate-300">
              {filteredExperiments.map((exp) => {
                const metrics = exp.summary_metrics_json;
                const isNoise = exp.is_noise_control;
                const skillPct = metrics?.skill_vs_naive !== undefined ? metrics.skill_vs_naive * 100 : null;

                return (
                  <tr
                    key={exp.id}
                    className={`hover:bg-slate-800/40 transition-colors ${
                      isNoise ? 'bg-red-950/10 text-slate-400 italic' : ''
                    }`}
                  >
                    <td className="py-3 px-4 font-sans font-medium text-slate-100 flex items-center gap-2">
                      {exp.name}
                      {isNoise && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-red-900/40 text-red-300 border border-red-800 font-mono">
                          Null Control
                        </span>
                      )}
                    </td>
                    <td className="py-3 px-4 font-mono text-slate-400 text-[11px]">{exp.model_family}</td>
                    <td className="py-3 px-4">
                      {metrics?.pinball_mean ? metrics.pinball_mean.toFixed(5) : '—'}
                    </td>
                    <td className="py-3 px-4 text-right">
                      {skillPct !== null ? (
                        <span
                          className={`font-semibold px-2 py-0.5 rounded ${
                            skillPct > 2.0
                              ? 'bg-emerald-950/60 text-emerald-400 border border-emerald-800/50'
                              : skillPct > 0
                              ? 'bg-sky-950/60 text-sky-400 border border-sky-800/50'
                              : 'bg-rose-950/60 text-rose-400 border border-rose-800/50'
                          }`}
                        >
                          {skillPct > 0 ? `+${skillPct.toFixed(2)}%` : `${skillPct.toFixed(2)}%`}
                        </span>
                      ) : (
                        '—'
                      )}
                    </td>
                    <td className="py-3 px-4 text-right">
                      {metrics?.pct_folds_positive !== undefined
                        ? `${metrics.pct_folds_positive.toFixed(1)}%`
                        : '—'}
                    </td>
                    <td className="py-3 px-4 text-center">
                      {metrics ? (
                        <span className="text-[11px]">
                          <span
                            className={Math.abs(metrics.cov_10 - 0.1) < 0.02 ? 'text-emerald-400' : 'text-amber-400'}
                          >
                            {(metrics.cov_10 * 100).toFixed(1)}%
                          </span>
                          {' / '}
                          <span
                            className={Math.abs(metrics.cov_80 - 0.8) < 0.03 ? 'text-emerald-400' : 'text-amber-400'}
                          >
                            {(metrics.cov_80 * 100).toFixed(1)}%
                          </span>
                          {' / '}
                          <span
                            className={Math.abs(metrics.cov_90 - 0.9) < 0.02 ? 'text-emerald-400' : 'text-amber-400'}
                          >
                            {(metrics.cov_90 * 100).toFixed(1)}%
                          </span>
                        </span>
                      ) : (
                        '—'
                      )}
                    </td>
                    <td className="py-3 px-4 text-right">
                      {metrics?.cond_dir_acc_conf10 !== null && metrics?.cond_dir_acc_conf10 !== undefined ? (
                        <span>
                          {(metrics.cond_dir_acc_conf10 * 100).toFixed(1)}% /{' '}
                          {metrics.cond_dir_acc_tension10 !== null && metrics.cond_dir_acc_tension10 !== undefined
                            ? `${(metrics.cond_dir_acc_tension10 * 100).toFixed(1)}%`
                            : '—'}
                        </span>
                      ) : (
                        '—'
                      )}
                    </td>
                    <td className="py-3 px-4 text-center font-sans">
                      <div className="flex items-center justify-center space-x-2">
                        <Link
                          href={`/inspector/${exp.id}`}
                          className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-sky-400 hover:text-sky-300 text-[11px] font-medium transition-colors"
                        >
                          Inspector
                        </Link>
                        <Link
                          href={`/autopsy/${exp.id}`}
                          className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-indigo-400 hover:text-indigo-300 text-[11px] font-medium transition-colors"
                        >
                          Autopsy
                        </Link>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Fold-by-Fold Skill Stability Chart */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 shadow-lg">
        <SkillStabilityChart
          folds={chartFolds}
          models={chartSeries}
          noiseCeiling95={0.012}
          showNoiseControl={showNoiseControl}
        />
      </div>
    </div>
  );
}
