'use client';

import React, { useMemo } from 'react';
import dynamic from 'next/dynamic';
import type { EChartsOption } from 'echarts';

// Lazy-loaded ReactECharts with SSR disabled and fallback loader
const ReactECharts = dynamic(() => import('echarts-for-react'), {
  ssr: false,
  loading: () => (
    <div className="h-full w-full flex items-center justify-center text-slate-400 text-sm animate-pulse bg-slate-900/40 rounded-lg">
      Loading chart canvas...
    </div>
  ),
});

export interface BaseChartProps {
  option: EChartsOption;
  style?: React.CSSProperties;
  className?: string;
  theme?: string | object;
  onEvents?: Record<string, (params: unknown) => void>;
}

export default function BaseChart({
  option,
  style = { height: '360px', width: '100%' },
  className = '',
  theme = 'dark',
  onEvents,
}: BaseChartProps) {
  // Enforce canvas renderer (never SVG for high-frequency time series)
  const chartOpts = useMemo(
    () => ({
      renderer: 'canvas' as const,
    }),
    []
  );

  return (
    <div className={`relative ${className}`} style={{ width: '100%', ...style }}>
      <ReactECharts
        option={option}
        opts={chartOpts}
        theme={theme}
        style={{ height: '100%', width: '100%' }}
        notMerge={true}
        lazyUpdate={true}
        onEvents={onEvents}
      />
    </div>
  );
}
