'use client';

import React, { useMemo } from 'react';
import BaseChart from './BaseChart';
import type { EChartsOption } from 'echarts';

/* ------------------------------------------------------------------ */
/*  1. Residual ACF Chart (lag 1..10 across folds)                    */
/* ------------------------------------------------------------------ */

interface ResidualAcfChartProps {
  lags: number[]; // e.g. [1, 2, ..., 10]
  foldAcfValues: number[][]; // per-fold array of lag 1..10 ACFs
  height?: string;
}

export function ResidualAcfChart({
  lags,
  foldAcfValues,
  height = '320px',
}: ResidualAcfChartProps) {
  const option = useMemo<EChartsOption>(() => {
    // Compute mean and std across folds for each lag
    const numLags = lags.length;
    const meanAcf: number[] = [];
    const stdAcf: number[] = [];

    for (let l = 0; l < numLags; l++) {
      const vals = foldAcfValues.map((fold) => fold[l] ?? 0);
      const mean = vals.reduce((a, b) => a + b, 0) / (vals.length || 1);
      const variance =
        vals.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (vals.length || 1);
      meanAcf.push(Number(mean.toFixed(4)));
      stdAcf.push(Number(Math.sqrt(variance).toFixed(4)));
    }

    return {
      backgroundColor: 'transparent',
      title: {
        text: 'Residual Autocorrelation (Lags 1..10)',
        subtext: 'Mean residual ACF across out-of-sample folds with ±1 standard deviation bounds',
        left: 'left',
        textStyle: { color: '#f1f5f9', fontSize: 14, fontWeight: 600 },
        subtextStyle: { color: '#94a3b8', fontSize: 11 },
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        backgroundColor: '#0f172a',
        borderColor: '#334155',
        textStyle: { color: '#f8fafc', fontSize: 11 },
      },
      grid: {
        top: 60,
        left: '5%',
        right: '4%',
        bottom: '12%',
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        name: 'Lag (min)',
        nameLocation: 'middle',
        nameGap: 24,
        data: lags.map((l) => `Lag ${l}`),
        axisLine: { lineStyle: { color: '#475569' } },
        axisLabel: { color: '#94a3b8', fontSize: 11 },
      },
      yAxis: {
        type: 'value',
        name: 'Residual ACF',
        nameTextStyle: { color: '#94a3b8', fontSize: 11 },
        splitLine: { lineStyle: { color: '#1e293b' } },
        axisLabel: { color: '#94a3b8', fontSize: 10 },
      },
      series: [
        {
          name: 'Mean ACF',
          type: 'bar',
          data: meanAcf,
          barMaxWidth: 24,
          itemStyle: {
            color: '#38bdf8',
            borderRadius: [3, 3, 0, 0],
          },
          markLine: {
            silent: true,
            symbol: 'none',
            lineStyle: { color: '#64748b', type: 'dashed' },
            data: [{ yAxis: 0 }],
          },
        },
      ],
    };
  }, [lags, foldAcfValues]);

  return <BaseChart option={option} style={{ height }} />;
}

/* ------------------------------------------------------------------ */
/*  2. Grouped Bar: Pinball & Calibration by Regime and by Hour       */
/* ------------------------------------------------------------------ */

interface GroupedBreakdownProps {
  categories: string[]; // e.g. ['State 0 (Compress)', 'State 1 (Normal)', 'State 2 (Cascade)'] or ['0h', '1h'..]
  pinballLoss: number[];
  cov10: number[];
  cov80: number[];
  cov90: number[];
  title: string;
  subtitle?: string;
  height?: string;
}

export function GroupedBreakdownChart({
  categories,
  pinballLoss,
  cov10,
  cov80,
  cov90,
  title,
  subtitle,
  height = '340px',
}: GroupedBreakdownProps) {
  const option = useMemo<EChartsOption>(() => {
    return {
      backgroundColor: 'transparent',
      title: {
        text: title,
        subtext: subtitle,
        left: 'left',
        textStyle: { color: '#f1f5f9', fontSize: 14, fontWeight: 600 },
        subtextStyle: { color: '#94a3b8', fontSize: 11 },
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        backgroundColor: '#0f172a',
        borderColor: '#334155',
        textStyle: { color: '#f8fafc', fontSize: 11 },
      },
      legend: {
        top: 28,
        right: 10,
        textStyle: { color: '#cbd5e1', fontSize: 11 },
      },
      grid: {
        top: 70,
        left: '5%',
        right: '5%',
        bottom: '12%',
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        data: categories,
        axisLine: { lineStyle: { color: '#475569' } },
        axisLabel: { color: '#94a3b8', fontSize: 10, interval: categories.length > 10 ? 1 : 0 },
      },
      yAxis: [
        {
          type: 'value',
          name: 'Pinball Loss',
          nameTextStyle: { color: '#94a3b8', fontSize: 10 },
          splitLine: { lineStyle: { color: '#1e293b' } },
          axisLabel: { color: '#94a3b8', fontSize: 10 },
        },
        {
          type: 'value',
          name: 'Coverage (%)',
          min: 0,
          max: 100,
          nameTextStyle: { color: '#94a3b8', fontSize: 10 },
          splitLine: { show: false },
          axisLabel: { color: '#94a3b8', fontSize: 10, formatter: '{value}%' },
        },
      ],
      series: [
        {
          name: 'Pinball Loss',
          type: 'bar',
          yAxisIndex: 0,
          data: pinballLoss.map((v) => Number(v.toFixed(5))),
          itemStyle: { color: '#f59e0b' },
        },
        {
          name: 'cov_10 (Ideal: 10%)',
          type: 'line',
          yAxisIndex: 1,
          data: cov10.map((v) => Number((v * 100).toFixed(1))),
          lineStyle: { color: '#38bdf8', width: 2 },
          symbol: 'circle',
        },
        {
          name: 'cov_80 (Ideal: 80%)',
          type: 'line',
          yAxisIndex: 1,
          data: cov80.map((v) => Number((v * 100).toFixed(1))),
          lineStyle: { color: '#10b981', width: 2 },
          symbol: 'diamond',
        },
        {
          name: 'cov_90 (Ideal: 90%)',
          type: 'line',
          yAxisIndex: 1,
          data: cov90.map((v) => Number((v * 100).toFixed(1))),
          lineStyle: { color: '#ec4899', width: 2 },
          symbol: 'triangle',
        },
      ],
    };
  }, [categories, pinballLoss, cov10, cov80, cov90, title, subtitle]);

  return <BaseChart option={option} style={{ height }} />;
}

/* ------------------------------------------------------------------ */
/*  3. Feature Importance Horizontal Bar with Standard Error Bars     */
/* ------------------------------------------------------------------ */

interface FeatureImportanceProps {
  features: {
    feature: string;
    mean_gain: number;
    std_gain: number;
  }[];
  height?: string;
}

export function FeatureImportanceChart({
  features,
  height = '500px',
}: FeatureImportanceProps) {
  const option = useMemo<EChartsOption>(() => {
    // Sort ascending so highest gain is on top of horizontal bar
    const sorted = [...features].sort((a, b) => a.mean_gain - b.mean_gain).slice(-20);

    const featureNames = sorted.map((f) => f.feature);
    const meanValues = sorted.map((f) => Number(f.mean_gain.toFixed(2)));
    const errorData = sorted.map((f, i) => [
      i,
      Number(Math.max(0, f.mean_gain - f.std_gain).toFixed(2)),
      Number((f.mean_gain + f.std_gain).toFixed(2)),
    ]);

    return {
      backgroundColor: 'transparent',
      title: {
        text: 'Top 20 LightGBM Feature Importances',
        subtext: 'Mean feature gain across walk-forward folds with standard error bars',
        left: 'left',
        textStyle: { color: '#f1f5f9', fontSize: 14, fontWeight: 600 },
        subtextStyle: { color: '#94a3b8', fontSize: 11 },
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        backgroundColor: '#0f172a',
        borderColor: '#334155',
        textStyle: { color: '#f8fafc', fontSize: 11 },
        formatter: (params: unknown) => {
          if (!Array.isArray(params) || params.length === 0) return '';
          const p = params[0];
          const feat = sorted[p.dataIndex];
          if (!feat) return '';
          return `<strong>${feat.feature}</strong><br/>Mean Gain: ${feat.mean_gain.toFixed(2)}<br/>Std Error: ±${feat.std_gain.toFixed(2)}`;
        },
      },
      grid: {
        top: 60,
        left: '25%',
        right: '8%',
        bottom: '8%',
        containLabel: false,
      },
      xAxis: {
        type: 'value',
        name: 'Gain',
        nameTextStyle: { color: '#94a3b8', fontSize: 10 },
        splitLine: { lineStyle: { color: '#1e293b' } },
        axisLabel: { color: '#94a3b8', fontSize: 10 },
      },
      yAxis: {
        type: 'category',
        data: featureNames,
        axisLine: { lineStyle: { color: '#475569' } },
        axisLabel: { color: '#cbd5e1', fontSize: 11 },
      },
      series: [
        {
          name: 'Mean Gain',
          type: 'bar',
          data: meanValues,
          itemStyle: {
            color: '#38bdf8',
            borderRadius: [0, 3, 3, 0],
          },
        },
        // Error bar representation using custom series or lines
        {
          type: 'custom',
          name: 'Std Error',
          itemStyle: { borderWidth: 1.5, color: '#f59e0b' },
          renderItem: (params, api) => {
            const yValue = api.value(0);
            const highPoint = api.coord([api.value(2), yValue]);
            const lowPoint = api.coord([api.value(1), yValue]);
            const halfHeight = 4;

            return {
              type: 'group',
              children: [
                {
                  type: 'line',
                  shape: {
                    x1: lowPoint[0],
                    y1: lowPoint[1],
                    x2: highPoint[0],
                    y2: highPoint[1],
                  },
                  style: api.style(),
                },
                {
                  type: 'line',
                  shape: {
                    x1: lowPoint[0],
                    y1: lowPoint[1] - halfHeight,
                    x2: lowPoint[0],
                    y2: lowPoint[1] + halfHeight,
                  },
                  style: api.style(),
                },
                {
                  type: 'line',
                  shape: {
                    x1: highPoint[0],
                    y1: highPoint[1] - halfHeight,
                    x2: highPoint[0],
                    y2: highPoint[1] + halfHeight,
                  },
                  style: api.style(),
                },
              ],
            };
          },
          data: errorData,
          z: 10,
        },
      ],
    };
  }, [features]);

  return <BaseChart option={option} style={{ height }} />;
}
