'use client';

import React, { useMemo } from 'react';
import BaseChart from './BaseChart';
import type { EChartsOption } from 'echarts';

interface AcfBarChartProps {
  title: string;
  subtitle?: string;
  lags: number[];
  values: number[];
  nObs: number;
  barColor?: string;
  height?: string;
}

export default function AcfBarChart({
  title,
  subtitle,
  lags,
  values,
  nObs,
  barColor = '#38bdf8',
  height = '300px',
}: AcfBarChartProps) {
  const option = useMemo<EChartsOption>(() => {
    // 95% Bartlett confidence bound: +/- 1.96 / sqrt(N)
    const band = nObs > 0 ? 1.96 / Math.sqrt(nObs) : 0.05;

    return {
      backgroundColor: 'transparent',
      title: {
        text: title,
        subtext: subtitle,
        left: 'left',
        textStyle: { color: '#f1f5f9', fontSize: 13, fontWeight: 600 },
        subtextStyle: { color: '#94a3b8', fontSize: 10 },
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        backgroundColor: '#0f172a',
        borderColor: '#334155',
        textStyle: { color: '#f8fafc', fontSize: 11 },
        formatter: (params: unknown) => {
          const p = Array.isArray(params) ? params[0] : params;
          if (!p) return '';
          const lag = p.name;
          const val = Number(p.value).toFixed(4);
          const sig = Math.abs(Number(p.value)) > band ? ' (Significant)' : ' (Noise)';
          return `<strong>Lag ${lag}m</strong>: ${val}${sig}<br/><span style="color:#ef4444">95% Band: ±${band.toFixed(4)}</span>`;
        },
      },
      grid: {
        top: 50,
        left: '6%',
        right: '4%',
        bottom: '12%',
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        name: 'Lag (min)',
        nameLocation: 'middle',
        nameGap: 24,
        data: lags.map((l) => `${l}`),
        axisLine: { lineStyle: { color: '#475569' } },
        axisLabel: { color: '#94a3b8', fontSize: 10, interval: 4 },
      },
      yAxis: {
        type: 'value',
        name: 'ACF',
        nameTextStyle: { color: '#94a3b8', fontSize: 10 },
        splitLine: { lineStyle: { color: '#1e293b' } },
        axisLabel: { color: '#94a3b8', fontSize: 10 },
      },
      series: [
        {
          name: 'Autocorrelation',
          type: 'bar',
          data: values.map((v) => Number(v.toFixed(4))),
          barMaxWidth: 10,
          itemStyle: {
            color: (params) => {
              const val = Number(params.value);
              if (Math.abs(val) > band) {
                return barColor;
              }
              return '#475569';
            },
            borderRadius: [2, 2, 0, 0],
          },
          markLine: {
            silent: true,
            symbol: 'none',
            lineStyle: { color: '#ef4444', type: 'dashed', width: 1.5 },
            data: [
              {
                yAxis: Number(band.toFixed(4)),
                label: {
                  formatter: `+1.96/√N (+${band.toFixed(3)})`,
                  position: 'insideEndTop',
                  color: '#ef4444',
                  fontSize: 9,
                },
              },
              {
                yAxis: -Number(band.toFixed(4)),
                label: {
                  formatter: `-1.96/√N (-${band.toFixed(3)})`,
                  position: 'insideEndBottom',
                  color: '#ef4444',
                  fontSize: 9,
                },
              },
              {
                yAxis: 0,
                lineStyle: { color: '#64748b', type: 'solid', width: 1 },
              },
            ],
          },
        },
      ],
    };
  }, [title, subtitle, lags, values, nObs, barColor]);

  return <BaseChart option={option} style={{ height }} />;
}
