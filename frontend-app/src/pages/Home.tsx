import React, { useState } from 'react';
import { apiPost } from '@/lib/api';
import { RunConfig, RunResult } from '@/lib/types';
import { FALLBACK_META } from '@/lib/fallbackMeta';
import { Play, Network, CheckCircle2, AlertCircle, MousePointerClick, Activity, Binary } from 'lucide-react';

export default function Home() {
  const [config, setConfig] = useState<RunConfig>(FALLBACK_META.presets[0].config);
  const [result, setResult] = useState<RunResult | null>(null);
  const [isSimulating, setIsSimulating] = useState(false);

  const handleRun = async () => {
    setIsSimulating(true);
    try {
      const res = await apiPost<RunResult>('/run', config);
      setResult(res);
    } catch (error) {
      console.error("Failed to run simulation on backend:", error);
      alert("Backend error! Make sure the python server is running on port 8001.");
    } finally {
      setIsSimulating(false);
    }
  };

  return (
    <div className="h-[100dvh] max-h-[100dvh] w-full overflow-hidden bg-background text-foreground font-sans selection:bg-primary/20 flex flex-col items-center">
      <div className="w-full max-w-[1200px] h-full flex flex-col px-4 md:px-6 pt-[18px] pb-[20px]">
        
        {/* HEADER */}
        <header className="flex-shrink-0 flex flex-col items-center text-center mb-4">
          <h1 className="text-[32px] md:text-[34px] font-extrabold text-[#1E293B] tracking-tight leading-[1.1]">
            Memory vs. Attention
          </h1>
          <p className="text-[15px] font-medium text-slate-500 mt-1">
            An Interactive BDI Explainer
          </p>
          <p className="text-[13.5px] text-slate-600 leading-[1.4] max-w-[700px] mx-auto mt-2">
            {FALLBACK_META.claim}
          </p>
        </header>

        {/* MAIN TWO-COLUMN LAYOUT: 100% remaining height */}
        <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-[340px_1fr] gap-[20px] items-stretch overflow-hidden">
          
          {/* LEFT PANEL: PRESETS */}
          <aside className="bg-card rounded-[16px] border border-slate-200 shadow-sm p-[16px] flex flex-col h-full overflow-hidden">
            <h2 className="flex-shrink-0 text-[13px] font-semibold uppercase tracking-[0.05em] text-slate-400 mb-2 pl-1">
              Simulation presets
            </h2>
            
            <div className="flex-1 flex flex-col gap-[5px] justify-between">
              {FALLBACK_META.presets.map((p) => {
                const isSelected = config.task === p.config.task && config.seq_len === p.config.seq_len;
                return (
                  <button
                    key={p.id}
                    onClick={() => {
                      setConfig(p.config);
                      setResult(null);
                    }}
                    className={`group text-left px-3 py-1.5 rounded-[10px] transition-all border flex flex-col justify-center relative flex-1 ${
                      isSelected
                        ? 'bg-indigo-50/50 border-indigo-200 shadow-sm'
                        : 'bg-transparent border-transparent hover:bg-slate-50 hover:border-slate-200'
                    }`}
                  >
                    <div className="flex items-center justify-between w-full mb-0.5">
                      <span className={`text-[13.5px] font-semibold ${isSelected ? 'text-indigo-900' : 'text-slate-800 group-hover:text-slate-900'}`}>
                        {p.name}
                      </span>
                      {isSelected && (
                        <div className="w-1.5 h-1.5 rounded-full bg-primary flex-shrink-0" />
                      )}
                    </div>
                    <span className="text-[11.5px] text-slate-500 leading-[1.25]">
                      {p.subtitle}
                    </span>
                  </button>
                );
              })}
            </div>
          </aside>

          {/* RIGHT PANEL: SIMULATION & BUTTON */}
          <div className="flex flex-col h-full overflow-hidden">
            <main className="flex-1 min-h-0 bg-card rounded-[16px] border border-slate-200 shadow-sm p-[16px] flex flex-col overflow-hidden">
              {!result ? (
                /* EMPTY STATE */
                <div className="flex-1 flex flex-col items-center justify-center text-center animate-in fade-in duration-500 overflow-hidden">
                  <div className="w-10 h-10 rounded-full bg-slate-50 border border-slate-100 flex items-center justify-center mb-3 shadow-sm flex-shrink-0">
                    <Network className="w-[20px] h-[20px] text-indigo-400 stroke-[1.5]" />
                  </div>
                  <h3 className="text-[18px] font-semibold text-slate-800 mb-2 flex-shrink-0">Ready to simulate</h3>
                  <p className="text-[13px] text-slate-500 max-w-[380px] mb-4 leading-relaxed flex-shrink-0">
                    Choose a scenario from the presets and run the simulation to explore how memory and attention interact.
                  </p>
                  
                  {/* 3-Step Visual Hint */}
                  <div className="flex items-center gap-3 text-[11px] font-medium text-slate-400 flex-shrink-0">
                    <div className="flex flex-col items-center gap-1.5">
                      <MousePointerClick className="w-3.5 h-3.5 text-slate-300" />
                      <span>Choose a preset</span>
                    </div>
                    <div className="w-6 h-[1px] bg-slate-200" />
                    <div className="flex flex-col items-center gap-1.5">
                      <Play className="w-3.5 h-3.5 text-slate-300" />
                      <span>Run simulation</span>
                    </div>
                    <div className="w-6 h-[1px] bg-slate-200" />
                    <div className="flex flex-col items-center gap-1.5">
                      <Activity className="w-3.5 h-3.5 text-slate-300" />
                      <span>Explore results</span>
                    </div>
                  </div>
                </div>
              ) : (
                /* RESULTS STATE */
                <div className="flex-1 min-h-0 flex flex-col animate-rise overflow-hidden">
                  <header className="flex-shrink-0 border-b border-slate-100 pb-2 mb-2">
                    <h2 className="text-[16px] font-bold text-slate-800 mb-1">Simulation Overview</h2>
                    <div className="flex items-center gap-2 text-[12px] text-slate-500">
                      Target: <code className="px-1.5 py-0.5 rounded bg-slate-100 text-slate-700 font-mono">{result.query_key}</code>
                      <span className="text-slate-300">→</span>
                      Expected: <code className="px-1.5 py-0.5 rounded bg-indigo-50 text-indigo-700 font-mono font-semibold">{result.ground_truth}</code>
                    </div>
                  </header>

                  <div className="flex-1 min-h-0 flex flex-col gap-[12px] overflow-y-auto pr-2 pb-2">
                    {result.modes.map((mode) => (
                      <div
                        key={mode.mode}
                        className={`flex-none p-3.5 rounded-[12px] border relative overflow-hidden flex flex-col gap-2.5 shadow-sm transition-all ${
                          mode.correct 
                            ? 'bg-white border-slate-200 hover:border-slate-300' 
                            : 'bg-red-50/30 border-red-100 hover:border-red-200'
                        }`}
                      >
                        <div className={`absolute top-0 left-0 w-[4px] h-full ${mode.correct ? 'bg-indigo-500' : 'bg-red-500'}`} />
                        
                        <div className="flex flex-col sm:flex-row sm:justify-between sm:items-start gap-2 pl-3">
                          <div className="flex items-center gap-2">
                            {mode.correct ? (
                              <CheckCircle2 className="w-[18px] h-[18px] text-emerald-500" />
                            ) : (
                              <AlertCircle className="w-[18px] h-[18px] text-red-500" />
                            )}
                            <h4 className="font-bold text-[14px] text-slate-800 leading-none">
                              {mode.label}
                            </h4>
                          </div>
                          <span
                            className={`px-2 py-1 rounded-[6px] text-[10px] font-bold tracking-wider uppercase leading-none ${
                              mode.correct
                                ? 'bg-emerald-50 text-emerald-700 border border-emerald-200/60'
                                : 'bg-red-50 text-red-700 border border-red-200/60'
                            }`}
                          >
                            {mode.correct ? 'Success' : 'Failed'}
                          </span>
                        </div>

                        <div className="pl-3 flex-none">
                          <p className="text-[13px] text-slate-600 leading-relaxed">
                            {mode.explanation}
                          </p>
                        </div>

                        {/* DATA CARDS */}
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 pl-3 mt-1">
                          <div className="bg-slate-50/50 rounded-[6px] border border-slate-100 p-2 shadow-sm">
                            <span className="block text-slate-500 text-[10px] font-bold uppercase tracking-wider mb-1 flex items-center gap-1">
                              <Binary className="w-3 h-3" /> Output
                            </span>
                            <span className={`font-mono text-[12px] font-semibold truncate block ${!mode.correct && mode.prediction !== 'UNKNOWN' ? 'text-red-600' : 'text-slate-800'}`}>
                              {mode.prediction}
                            </span>
                          </div>
                          <div className="bg-slate-50/50 rounded-[6px] border border-slate-100 p-2 shadow-sm">
                            <span className="block text-slate-500 text-[10px] font-bold uppercase tracking-wider mb-1">Conf</span>
                            <span className="font-mono text-[12px] text-slate-800 font-semibold">
                              {(mode.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="bg-slate-50/50 rounded-[6px] border border-slate-100 p-2 shadow-sm">
                            <span className="block text-slate-500 text-[10px] font-bold uppercase tracking-wider mb-1">State Size</span>
                            <span className="font-mono text-[11px] text-slate-800 font-semibold truncate block" title={mode.state_scaling}>
                              {mode.state_scaling.split('—')[0].trim()}
                            </span>
                          </div>
                          {mode.evictions !== undefined ? (
                            <div className="bg-slate-50/50 rounded-[6px] border border-slate-100 p-2 shadow-sm">
                              <span className="block text-slate-500 text-[10px] font-bold uppercase tracking-wider mb-1">Evictions</span>
                              <span className="font-mono text-[12px] text-slate-800 font-semibold">{mode.evictions}</span>
                            </div>
                          ) : (
                            <div className="bg-transparent rounded-[6px] p-2" />
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </main>
            
            {/* RUN SIMULATION BUTTON (RIGHT-ALIGNED UNDER MAIN CONTENT) */}
            <div className="flex-shrink-0 flex justify-end mt-[14px]">
              <button
                onClick={handleRun}
                disabled={isSimulating}
                className="w-[180px] h-[44px] flex items-center justify-center gap-2 bg-primary text-white text-[14px] font-medium rounded-[10px] hover:bg-indigo-700 active:bg-indigo-800 transition-all shadow-sm focus:ring-2 focus:ring-primary focus:ring-offset-2 disabled:opacity-70"
              >
                {isSimulating ? (
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                ) : (
                  <Play className="w-[15px] h-[15px] fill-current" />
                )}
                {isSimulating ? 'Running...' : 'Run Simulation'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
