'use client';

import React, { useMemo } from 'react';
import BaseChart from './BaseChart';
import type { EChartsOption } from 'echarts';

interface HeatmapChartProps {
  title: string;
  subtitle?: string;
  matrix: number[][]; // 7 rows (DoW 0..6: Mon..Sun) x 24 cols (Hour 0..23)
  valueLabel?: string;
  height?: string;
}

const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
const HOURS = Array.from({ length: 24 }, (_, i) => `${i}h`);

export default function HeatmapChart({
  title,
  subtitle,
  matrix,
  valueLabel = 'Intensity',
  height = '340px',
}: HeatmapChartProps) {
  const option = useMemo<EChartsOption>(() => {
    // Flatten 7x24 to [hour, dow, value]
    const data: [number, number, number][] = [];
    let minVal = Infinity;
    let maxVal = -Infinity;

    for (let d = 0; d < 7; d++) {
      for (let h = 0; h < 24; h++) {
        const val = matrix[d]?.[h] ?? 0;
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
        data.push([h, d, Number(val.toFixed(4))]);
      }
    }

    if (minVal === Infinity) {
      minVal = 0;
      maxVal = 1;
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
        position: 'top',
        backgroundColor: '#0f172a',
        borderColor: '#334155',
        textStyle: { color: '#f8fafc', fontSize: 11 },
        formatter: (params: unknown) => {
          const p = params as { value: [number, number, number] };
          if (!p || !p.value) return '';
          const [h, d, val] = p.value;
          return `<strong>${DAYS[d]} @ ${h}:00 UTC</strong><br/>${valueLabel}: <strong>${val}</strong>`;
        },
      },
      grid: {
        top: 60,
        left: '4%',
        right: '8%',
        bottom: '12%',
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        data: HOURS,
        splitArea: { show: true, areaStyle: { color: ['rgba(30,41,59,0.2)', 'rgba(15,23,42,0.2)'] } },
        axisLine: { lineStyle: { color: '#475569' } },
        axisLabel: { color: '#94a3b8', fontSize: 10 },
      },
      yAxis: {
        type: 'category',
        data: DAYS,
        splitArea: { show: true },
        axisLine: { lineStyle: { color: '#475569' } },
        axisLabel: { color: '#94a3b8', fontSize: 11 },
      },
      visualMap: {
        min: minVal,
        max: maxVal,
        calculable: true,
        orient: 'vertical',
        right: '1%',
        top: 'center',
        inRange: {
          color: ['#0f172a', '#1e293b', '#0284c7', '#38bdf8', '#f59e0b', '#ef4444'],
        },
        textStyle: { color: '#94a3b8', fontSize: 10 },
      },
      series: [
        {
          name: valueLabel,
          type: 'heatmap',
          data,
          label: { show: false },
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowColor: 'rgba(0, 0, 0, 0.5)',
            },
          },
        },
      ],
    };
  }, [title, subtitle, matrix, valueLabel]);

  return <BaseChart option={option} style={{ height }} />;
}
