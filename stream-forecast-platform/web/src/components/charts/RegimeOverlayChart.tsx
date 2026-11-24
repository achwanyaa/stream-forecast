'use client';

import React, { useMemo } from 'react';
import BaseChart from './BaseChart';
import type { FoldSlice } from '@/lib/supabase';
import type { EChartsOption } from 'echarts';

interface RegimeOverlayChartProps {
  slice: FoldSlice;
  foldIdx: number;
  height?: string;
}

export default function RegimeOverlayChart({
  slice,
  foldIdx,
  height = '580px',
}: RegimeOverlayChartProps) {
  const option = useMemo<EChartsOption>(() => {
    const { t, x, p10, p50, p90, r0, r1, r2, cascade } = slice;

    // Convert timestamps to date strings
    const timeLabels = t.map((ts) => {
      const d = new Date(ts);
      return `${d.getUTCMonth() + 1}/${d.getUTCDate()} ${d.getUTCHours().toString().padStart(2, '0')}:${d.getUTCMinutes().toString().padStart(2, '0')}`;
    });

    // P90-P10 cone band: P10 is base (transparent), delta = P90 - P10 (colored band)
    const coneDelta = p90.map((high, i) => Math.max(0, high - p10[i]));

    // Cascade point markers
    const cascadePoints: [string, number][] = [];
    cascade.forEach((flag, i) => {
      if (flag === 1) {
        cascadePoints.push([timeLabels[i], x[i]]);
      }
    });

    return {
      backgroundColor: 'transparent',
      title: {
        text: `Regime & Forecast Viewport — Fold ${foldIdx}`,
        subtext: 'Actual price stream x_t with P10–P90 uncertainty cone + synchronized 3-state HMM probabilities & BOCPD cascade events',
        left: 'left',
        textStyle: { color: '#f1f5f9', fontSize: 15, fontWeight: 600 },
        subtextStyle: { color: '#94a3b8', fontSize: 11 },
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross',
          link: [{ xAxisIndex: 'all' }],
        },
        backgroundColor: '#0f172a',
        borderColor: '#334155',
        textStyle: { color: '#f8fafc', fontSize: 11 },
      },
      axisPointer: {
        link: [{ xAxisIndex: 'all' }],
      },
      legend: {
        top: 32,
        right: 10,
        textStyle: { color: '#cbd5e1', fontSize: 11 },
      },
      grid: [
        // Upper Viewport: Price + Forecast Cone + Cascade markers
        {
          top: 75,
          left: '4%',
          right: '4%',
          height: '50%',
          containLabel: true,
        },
        // Lower Viewport: Stacked Regime Probabilities (0: Slate, 1: Amber, 2: Crimson)
        {
          top: '67%',
          left: '4%',
          right: '4%',
          height: '20%',
          containLabel: true,
        },
      ],
      dataZoom: [
        {
          type: 'slider',
          xAxisIndex: [0, 1],
          bottom: '2%',
          height: 24,
          borderColor: '#334155',
          fillerColor: 'rgba(56, 189, 248, 0.15)',
          textStyle: { color: '#94a3b8', fontSize: 10 },
          handleStyle: { color: '#38bdf8' },
        },
        {
          type: 'inside',
          xAxisIndex: [0, 1],
        },
      ],
      xAxis: [
        {
          gridIndex: 0,
          type: 'category',
          data: timeLabels,
          axisLine: { lineStyle: { color: '#475569' } },
          axisLabel: { show: false },
        },
        {
          gridIndex: 1,
          type: 'category',
          data: timeLabels,
          axisLine: { lineStyle: { color: '#475569' } },
          axisLabel: { color: '#94a3b8', fontSize: 10, interval: Math.floor(t.length / 8) },
        },
      ],
      yAxis: [
        {
          gridIndex: 0,
          type: 'value',
          scale: true,
          name: 'x_close',
          nameTextStyle: { color: '#94a3b8', fontSize: 11 },
          splitLine: { lineStyle: { color: '#1e293b' } },
          axisLabel: { color: '#94a3b8', fontSize: 10 },
        },
        {
          gridIndex: 1,
          type: 'value',
          min: 0,
          max: 1,
          name: 'P(Regime)',
          nameTextStyle: { color: '#94a3b8', fontSize: 10 },
          splitLine: { lineStyle: { color: '#1e293b' } },
          axisLabel: { color: '#94a3b8', fontSize: 9 },
        },
      ],
      series: [
        // --- Upper Grid Series ---
        // P10 Base (transparent)
        {
          name: 'P10 Lower',
          type: 'line',
          xAxisIndex: 0,
          yAxisIndex: 0,
          data: p10,
          lineStyle: { opacity: 0 },
          stack: 'uncertainty-cone',
          symbol: 'none',
        },
        // P90 - P10 Shaded Cone
        {
          name: 'P10-P90 Forecast Cone',
          type: 'line',
          xAxisIndex: 0,
          yAxisIndex: 0,
          data: coneDelta,
          lineStyle: { opacity: 0 },
          stack: 'uncertainty-cone',
          symbol: 'none',
          areaStyle: {
            color: 'rgba(56, 189, 248, 0.22)',
          },
        },
        // P50 Median Forecast Line
        {
          name: 'P50 Forecast',
          type: 'line',
          xAxisIndex: 0,
          yAxisIndex: 0,
          data: p50,
          showSymbol: false,
          lineStyle: { color: '#38bdf8', width: 2, type: 'dashed' },
        },
        // Actual Price x_t
        {
          name: 'x_t (Actual)',
          type: 'line',
          xAxisIndex: 0,
          yAxisIndex: 0,
          data: x,
          showSymbol: false,
          lineStyle: { color: '#f8fafc', width: 2 },
        },
        // Cascade Event Points
        {
          name: 'Cascade Event (BOCPD)',
          type: 'scatter',
          xAxisIndex: 0,
          yAxisIndex: 0,
          data: cascadePoints,
          symbol: 'triangle',
          symbolSize: 12,
          itemStyle: { color: '#ef4444' },
        },

        // --- Lower Grid Series (Stacked Regime Probabilities) ---
        // State 0 = Slate (Compression)
        {
          name: 'State 0 (Compression)',
          type: 'line',
          xAxisIndex: 1,
          yAxisIndex: 1,
          stack: 'regimes',
          data: r0,
          showSymbol: false,
          lineStyle: { width: 0.5, color: '#64748b' },
          areaStyle: { color: 'rgba(100, 116, 139, 0.7)' },
        },
        // State 1 = Amber (Normal / Transition)
        {
          name: 'State 1 (Normal)',
          type: 'line',
          xAxisIndex: 1,
          yAxisIndex: 1,
          stack: 'regimes',
          data: r1,
          showSymbol: false,
          lineStyle: { width: 0.5, color: '#f59e0b' },
          areaStyle: { color: 'rgba(245, 158, 11, 0.7)' },
        },
        // State 2 = Crimson (Expansion / Cascade)
        {
          name: 'State 2 (Cascade)',
          type: 'line',
          xAxisIndex: 1,
          yAxisIndex: 1,
          stack: 'regimes',
          data: r2,
          showSymbol: false,
          lineStyle: { width: 0.5, color: '#ef4444' },
          areaStyle: { color: 'rgba(239, 68, 68, 0.7)' },
        },
      ],
    };
  }, [slice, foldIdx]);

  return <BaseChart option={option} style={{ height }} />;
}
