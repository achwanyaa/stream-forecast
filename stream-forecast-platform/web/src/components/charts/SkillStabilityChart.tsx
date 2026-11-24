'use client';

import React, { useMemo } from 'react';
import BaseChart from './BaseChart';
import type { EChartsOption } from 'echarts';

export interface ModelSkillSeries {
  id: string;
  name: string;
  modelFamily: string;
  isNoiseControl?: boolean;
  skillData: (number | null)[]; // fold-by-fold skill vs naive (e.g. 0.04 for +4%)
}

interface SkillStabilityChartProps {
  folds: number[]; // e.g. [0, 1, 2, ..., N-1]
  models: ModelSkillSeries[];
  noiseCeiling95?: number; // 95% ceiling threshold (e.g. 0.015)
  showNoiseControl?: boolean;
}

export default function SkillStabilityChart({
  folds,
  models,
  noiseCeiling95 = 0.01,
  showNoiseControl = true,
}: SkillStabilityChartProps) {
  const option = useMemo<EChartsOption>(() => {
    // Generate series per model
    const seriesList: NonNullable<EChartsOption['series']> = models
      .filter((m) => showNoiseControl || !m.isNoiseControl)
      .map((m) => {
        const isNoise = m.isNoiseControl;
        return {
          name: m.name,
          type: 'line',
          data: m.skillData.map((v) => (v !== null ? Number((v * 100).toFixed(2)) : null)),
          smooth: true,
          showSymbol: true,
          symbolSize: 6,
          lineStyle: {
            width: isNoise ? 1.5 : 2.5,
            type: isNoise ? ('dashed' as const) : ('solid' as const),
            opacity: isNoise ? 0.6 : 1,
          },
        };
      });

    const opt: EChartsOption = {
      backgroundColor: 'transparent',
      title: {
        text: 'Fold-by-Fold Skill vs. Naive Stability',
        subtext: 'Out-of-sample skill score per 15-day non-overlapping fold with Noise Control threshold',
        left: 'left',
        textStyle: { color: '#f1f5f9', fontSize: 14, fontWeight: 600 },
        subtextStyle: { color: '#94a3b8', fontSize: 11 },
      },
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#0f172a',
        borderColor: '#334155',
        textStyle: { color: '#f8fafc', fontSize: 12 },
        valueFormatter: (val) => (val !== undefined && val !== null ? `${val}%` : 'N/A'),
      },
      legend: {
        type: 'scroll',
        top: 28,
        right: 10,
        textStyle: { color: '#cbd5e1', fontSize: 11 },
      },
      grid: {
        top: 75,
        left: '4%',
        right: '4%',
        bottom: '10%',
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        name: 'Fold Index (Chronological)',
        nameLocation: 'middle',
        nameGap: 28,
        data: folds.map((f) => `Fold ${f}`),
        axisLine: { lineStyle: { color: '#475569' } },
        axisLabel: { color: '#94a3b8', fontSize: 11 },
      },
      yAxis: {
        type: 'value',
        name: 'Skill vs Naive (%)',
        nameTextStyle: { color: '#94a3b8', fontSize: 11 },
        splitLine: { lineStyle: { color: '#1e293b' } },
        axisLabel: {
          color: '#94a3b8',
          fontSize: 11,
          formatter: '{value}%',
        },
      },
      series: [
        ...seriesList,
        // Noise Control 95% Ceiling Band
        {
          name: 'Noise Control 95% Band',
          type: 'line',
          data: [],
          markLine: {
            silent: true,
            symbol: 'none',
            lineStyle: {
              color: '#ef4444',
              type: 'dashed',
              width: 2,
            },
            data: [
              {
                yAxis: Number((noiseCeiling95 * 100).toFixed(2)),
                label: {
                  formatter: `Noise 95% Ceiling (+${(noiseCeiling95 * 100).toFixed(1)}%)`,
                  position: 'insideEndTop',
                  color: '#ef4444',
                  fontSize: 10,
                },
              },
              {
                yAxis: 0,
                lineStyle: { color: '#64748b', type: 'solid', width: 1 },
                label: {
                  formatter: 'Naive Baseline (0%)',
                  position: 'insideEndBottom',
                  color: '#64748b',
                  fontSize: 10,
                },
              },
            ],
          },
          markArea: {
            silent: true,
            itemStyle: {
              color: 'rgba(239, 68, 68, 0.07)',
            },
            data: [
              [
                { yAxis: 0 },
                { yAxis: Number((noiseCeiling95 * 100).toFixed(2)) },
              ],
            ],
          },
        },
      ],
    };

    return opt;
  }, [folds, models, noiseCeiling95, showNoiseControl]);

  return <BaseChart option={option} style={{ height: '380px' }} />;
}
