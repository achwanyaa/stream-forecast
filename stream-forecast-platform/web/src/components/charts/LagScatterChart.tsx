'use client';

import React, { useMemo } from 'react';
import BaseChart from './BaseChart';
import type { EChartsOption } from 'echarts';

interface LagScatterChartProps {
  title: string;
  subtitle?: string;
  xData: number[]; // e.g. r_t or |r_t|
  yData: number[]; // e.g. r_{t+1} or |r_{t+1}|
  xLabel?: string;
  yLabel?: string;
  color?: string;
  height?: string;
}

export default function LagScatterChart({
  title,
  subtitle,
  xData,
  yData,
  xLabel = 'r_t',
  yLabel = 'r_{t+1}',
  color = '#38bdf8',
  height = '340px',
}: LagScatterChartProps) {
  const option = useMemo<EChartsOption>(() => {
    const points: [number, number][] = [];
    const len = Math.min(xData.length, yData.length);
    for (let i = 0; i < len; i++) {
      points.push([Number(xData[i].toFixed(5)), Number(yData[i].toFixed(5))]);
    }

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
        trigger: 'item',
        backgroundColor: '#0f172a',
        borderColor: '#334155',
        textStyle: { color: '#f8fafc', fontSize: 11 },
        formatter: (params: unknown) => {
          const p = params as { value: [number, number] };
          if (!p || !p.value) return '';
          return `${xLabel}: ${p.value[0]}<br/>${yLabel}: ${p.value[1]}`;
        },
      },
      grid: {
        top: 60,
        left: '6%',
        right: '4%',
        bottom: '12%',
        containLabel: true,
      },
      xAxis: {
        type: 'value',
        name: xLabel,
        nameLocation: 'middle',
        nameGap: 24,
        axisLine: { lineStyle: { color: '#475569' } },
        axisLabel: { color: '#94a3b8', fontSize: 10 },
        splitLine: { lineStyle: { color: '#1e293b' } },
      },
      yAxis: {
        type: 'value',
        name: yLabel,
        nameTextStyle: { color: '#94a3b8', fontSize: 10 },
        axisLine: { lineStyle: { color: '#475569' } },
        axisLabel: { color: '#94a3b8', fontSize: 10 },
        splitLine: { lineStyle: { color: '#1e293b' } },
      },
      series: [
        {
          name: `${xLabel} vs ${yLabel}`,
          type: 'scatter',
          data: points,
          symbolSize: 3,
          large: true, // Use ECharts large mode for high performance
          largeThreshold: 2000,
          itemStyle: {
            color: color,
            opacity: 0.45,
          },
        },
      ],
    };
  }, [title, subtitle, xData, yData, xLabel, yLabel, color]);

  return <BaseChart option={option} style={{ height }} />;
}
