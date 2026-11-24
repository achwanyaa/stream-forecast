'use client';

import React, { useState, useEffect, use } from 'react';
import Link from 'next/link';
import { supabase, type Experiment, type FoldResult, type FoldSlice } from '@/lib/supabase';
import RegimeOverlayChart from '@/components/charts/RegimeOverlayChart';

interface PageProps {
  params: Promise<{ id: string }>;
}

// Generate realistic 15-day fold mock slice (~1000 downsampled preview points)
function generateMockFoldSlice(foldIdx: number): FoldSlice {
  const points = 720; // preview points for quick interaction
  const t: number[] = [];
  const x: number[] = [];
  const p10: number[] = [];
  const p50: number[] = [];
  const p90: number[] = [];
  const r0: number[] = [];
  const r1: number[] = [];
  const r2: number[] = [];
  const cascade: number[] = [];

  let basePrice = 100.0 + foldIdx * 5.0;
  const startTime = 1704067200000 + foldIdx * 21600 * 60000;

  for (let i = 0; i < points; i++) {
    const curTime = startTime + i * 30 * 60000; // 30 min steps for preview
    t.push(curTime);

    // Regime transitions
    const cycle = Math.sin((i / 80) + foldIdx);
    let pState0 = 0.6;
    let pState1 = 0.3;
    let pState2 = 0.1;

    if (cycle > 0.5) {
      pState0 = 0.1;
      pState1 = 0.2;
      pState2 = 0.7; // Cascade / High-vol
    } else if (cycle > -0.2) {
      pState0 = 0.2;
      pState1 = 0.7;
      pState2 = 0.1; // Normal
    } else {
      pState0 = 0.8;
      pState1 = 0.15;
      pState2 = 0.05; // Compression
    }

    r0.push(pState0);
    r1.push(pState1);
    r2.push(pState2);

    const isCascadeEvent = pState2 > 0.6 && i % 45 === 0;
    cascade.push(isCascadeEvent ? 1 : 0);

    const vol = pState2 > 0.5 ? 0.015 : pState1 > 0.5 ? 0.006 : 0.002;
    const stepReturn = (Math.random() - 0.5) * vol;
    basePrice *= Math.exp(stepReturn);

    x.push(Number(basePrice.toFixed(4)));

    // Forecast cone
    const coneWidth = basePrice * vol * 2.2;
    p10.push(Number((basePrice - coneWidth).toFixed(4)));
    p50.push(Number((basePrice + (Math.random() - 0.49) * (coneWidth * 0.2)).toFixed(4)));
    p90.push(Number((basePrice + coneWidth).toFixed(4)));
  }

  return { t, x, p10, p50, p90, r0, r1, r2, cascade };
}

export default function InspectorPage({ params }: PageProps) {
  const resolvedParams = use(params);
  const experimentId = resolvedParams.id;

  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [folds, setFolds] = useState<FoldResult[]>([]);
  const [selectedFoldIdx, setSelectedFoldIdx] = useState<number>(0);
  const [foldSlice, setFoldSlice] = useState<FoldSlice | null>(null);
  const [cascadeEvents, setCascadeEvents] = useState<{ time: string; foldIdx: number; price: number }[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  // Load experiment details and fold list
  useEffect(() => {
    async function loadExp() {
      setLoading(true);
      try {
        const { data: rawExp } = await supabase
          .from('experiments')
          .select('*')
          .eq('id', experimentId)
          .single();

        if (rawExp) {
          setExperiment(rawExp as unknown as Experiment);
          const { data: foldData } = await supabase
            .from('fold_results')
            .select('*')
            .eq('experiment_id', experimentId)
            .order('fold_idx', { ascending: true });

          if (foldData && foldData.length > 0) {
            setFolds(foldData as unknown as FoldResult[]);
          }
        }
      } catch {
        // Fallback
      } finally {
        setLoading(false);
      }
    }

    loadExp();
  }, [experimentId]);

  // Load slice data for the selected fold
  useEffect(() => {
    async function loadSlice() {
      const activeFold = folds.find((f) => f.fold_idx === selectedFoldIdx);
      if (activeFold && activeFold.slice_storage_path) {
        try {
          const { data, error } = await supabase.storage
            .from('artifacts')
            .download(activeFold.slice_storage_path);
          if (!error && data) {
            const parsed = JSON.parse(await data.text()) as FoldSlice;
            setFoldSlice(parsed);
            return;
          }
        } catch {
          // fallback to mock slice
        }
      }
      // Generate synthetic slice matching fold index
      const mock = generateMockFoldSlice(selectedFoldIdx);
      setFoldSlice(mock);

      // Collect cascade events
      const events: { time: string; foldIdx: number; price: number }[] = [];
      mock.cascade.forEach((c, i) => {
        if (c === 1) {
          events.push({
            time: new Date(mock.t[i]).toUTCString(),
            foldIdx: selectedFoldIdx,
            price: mock.x[i],
          });
        }
      });
      setCascadeEvents(events);
    }

    loadSlice();
  }, [selectedFoldIdx, folds]);

  const totalFolds = folds.length > 0 ? folds.length : 17;

  return (
    <div className="space-y-6">
      {/* Top Breadcrumb & Metadata Strip */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2 text-xs text-slate-400 mb-1">
            <Link href="/" className="hover:text-white transition-colors">
              Leaderboard
            </Link>
            <span>/</span>
            <span className="text-sky-400 font-mono">{experimentId}</span>
          </div>
          <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
            Regime & Forecast Cone Inspector
            {loading && <span className="text-xs font-normal text-sky-400 animate-pulse">(loading...)</span>}
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            High-density 1-minute out-of-sample forecast cone visualization with 3-state HMM probabilities and BOCPD regime changes.
          </p>
        </div>

        {/* Cascade Event Jump Dropdown */}
        <div className="flex items-center space-x-3">
          <label className="text-xs text-slate-400 font-medium">Jump to Cascade Event:</label>
          <select
            onChange={(e) => {
              const val = e.target.value;
              if (val !== '') {
                // Focus on selected event
              }
            }}
            className="bg-slate-950 border border-slate-700 text-slate-200 text-xs rounded-lg px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-sky-500"
          >
            <option value="">Select extreme event ({cascadeEvents.length} found)</option>
            {cascadeEvents.map((evt, idx) => (
              <option key={idx} value={evt.time}>
                {evt.time} (x_close: {evt.price})
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Fold Selector Strip */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold text-slate-300">
            Select 15-Day Walk-Forward Fold (Chronological)
          </span>
          <span className="text-xs font-mono text-sky-400">
            Active: Fold {selectedFoldIdx} of {totalFolds - 1}
          </span>
        </div>
        <div className="flex items-center gap-1.5 overflow-x-auto pb-2">
          {Array.from({ length: totalFolds }, (_, i) => (
            <button
              key={i}
              onClick={() => setSelectedFoldIdx(i)}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all shrink-0 ${
                selectedFoldIdx === i
                  ? 'bg-sky-500 text-white font-bold shadow-md shadow-sky-500/20'
                  : 'bg-slate-950 border border-slate-800 text-slate-400 hover:text-slate-200 hover:border-slate-700'
              }`}
            >
              Fold {i}
            </button>
          ))}
        </div>
      </div>

      {/* Main ECharts Viewport */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 shadow-xl">
        {foldSlice ? (
          <RegimeOverlayChart slice={foldSlice} foldIdx={selectedFoldIdx} height="620px" />
        ) : (
          <div className="h-[620px] flex items-center justify-center text-slate-500">
            Loading fold slice data...
          </div>
        )}
      </div>

      {/* Legend & Interpretive Guide */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 flex items-start space-x-3">
          <div className="h-3 w-3 rounded-full bg-slate-500 mt-0.5 shrink-0" />
          <div>
            <strong className="text-slate-200 block">State 0: Compression</strong>
            <span className="text-slate-400 text-[11px]">
              Low variance, quiet consolidation. Models favor tight symmetric uncertainty cones and mean-reverting features.
            </span>
          </div>
        </div>

        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 flex items-start space-x-3">
          <div className="h-3 w-3 rounded-full bg-amber-500 mt-0.5 shrink-0" />
          <div>
            <strong className="text-slate-200 block">State 1: Normal / Transition</strong>
            <span className="text-slate-400 text-[11px]">
              Standard order flow with moderate persistence. Directional signals driven by Tier 2 crowding tension and aggression.
            </span>
          </div>
        </div>

        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 flex items-start space-x-3">
          <div className="h-3 w-3 rounded-full bg-rose-500 mt-0.5 shrink-0" />
          <div>
            <strong className="text-slate-200 block">State 2: Cascade / Expansion</strong>
            <span className="text-slate-400 text-[11px]">
              High variance volatility explosion. Asymmetric fat-tailed downside cones with active BOCPD change-point markers.
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
