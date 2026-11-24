'use client';

import React, { useMemo } from 'react';
import BaseChart from './BaseChart';
import type { EChartsOption } from 'echarts';

interface WelchPsdChartProps {
  freqs: number[]; // frequency in cycles / minute
  absRPower: number[];
  activityPower?: number[];
  height?: string;
}

export default function WelchPsdChart({
  freqs,
  absRPower,
  activityPower,
  height = '340px',
}: WelchPsdChartProps) {
  const option = useMemo<EChartsOption>(() => {
    // 24h diurnal = 1 / 1440 min^-1 ≈ 0.0006944
    // 7d weekly = 1 / 10080 min^-1 ≈ 0.0000992
    const diurnalFreq = 1 / 1440;
    const weeklyFreq = 1 / 10080;

    // Filter to positive frequencies
    const validPoints = freqs
      .map((f, i) => ({
        freq: f,
        periodMin: f > 0 ? 1 / f : 0,
        absR: absRPower[i] ?? 0,
        act: activityPower ? activityPower[i] ?? 0 : null,
      }))
      .filter((p) => p.freq > 0);

    const series: NonNullable<EChartsOption['series']> = [
      {
        name: '|r_t| Volatility Power',
        type: 'line',
        data: validPoints.map((p) => [p.freq, p.absR]),
        showSymbol: false,
        smooth: true,
        lineStyle: { color: '#f59e0b', width: 2 },
        markLine: {
          symbol: 'none',
          lineStyle: { width: 1.5, type: 'dashed' },
          data: [
            {
              xAxis: diurnalFreq,
              lineStyle: { color: '#38bdf8' },
              label: {
                formatter: '24h Diurnal\n(1440m)',
                position: 'insideEndTop',
                color: '#38bdf8',
                fontSize: 10,
              },
            },
            {
              xAxis: weeklyFreq,
              lineStyle: { color: '#a855f7' },
              label: {
                formatter: '7d Weekly\n(10080m)',
                position: 'insideEndTop',
                color: '#a855f7',
                fontSize: 10,
              },
            },
          ],
        },
      },
    ];

    if (activityPower && activityPower.length > 0) {
      series.push({
        name: 'Activity Magnitude Power',
        type: 'line',
        data: validPoints.map((p) => [p.freq, p.act]),
        showSymbol: false,
        smooth: true,
        lineStyle: { color: '#10b981', width: 1.5, type: 'dotted' },
      });
    }

    return {
      backgroundColor: 'transparent',
      title: {
        text: 'Welch Power Spectral Density (PSD)',
        subtext: 'Frequency domain power with 24-hour and 7-day cyclical periodicity markers',
        left: 'left',
        textStyle: { color: '#f1f5f9', fontSize: 14, fontWeight: 600 },
        subtextStyle: { color: '#94a3b8', fontSize: 11 },
      },
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#0f172a',
        borderColor: '#334155',
        textStyle: { color: '#f8fafc', fontSize: 11 },
        formatter: (params: unknown) => {
          if (!Array.isArray(params) || params.length === 0) return '';
          const f = Number(params[0].value[0]);
          const periodMin = f > 0 ? (1 / f).toFixed(1) : 'inf';
          const periodHours = f > 0 ? (1 / (f * 60)).toFixed(2) : 'inf';
          let out = `<strong>Frequency: ${f.toFixed(6)} min⁻¹</strong> (${periodMin} min / ${periodHours} hrs)<br/>`;
          params.forEach((item) => {
            out += `<span style="color:${item.color}">●</span> ${item.seriesName}: ${Number(item.value[1]).toExponential(3)}<br/>`;
          });
          return out;
        },
      },
      legend: {
        top: 28,
        right: 10,
        textStyle: { color: '#cbd5e1', fontSize: 11 },
      },
      grid: {
        top: 70,
        left: '5%',
        right: '4%',
        bottom: '12%',
        containLabel: true,
      },
      xAxis: {
        type: 'value',
        name: 'Frequency (cycles/min)',
        nameLocation: 'middle',
        nameGap: 24,
        axisLine: { lineStyle: { color: '#475569' } },
        axisLabel: {
          color: '#94a3b8',
          fontSize: 10,
          formatter: (v: number) => v.toFixed(4),
        },
        splitLine: { lineStyle: { color: '#1e293b' } },
      },
      yAxis: {
        type: 'log',
        name: 'Log Power Spectral Density',
        nameTextStyle: { color: '#94a3b8', fontSize: 11 },
        splitLine: { lineStyle: { color: '#1e293b' } },
        axisLabel: {
          color: '#94a3b8',
          fontSize: 10,
          formatter: (v: number) => v.toExponential(0),
        },
      },
      series,
    };
  }, [freqs, absRPower, activityPower]);

  return <BaseChart option={option} style={{ height }} />;
}
