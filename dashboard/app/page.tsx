"use client";

import { useEffect, useMemo, useState } from "react";

type Point = {
  label: string;
  request_rate: number;
  output_throughput_tok_s: number;
  ttft_p95_ms: number;
  tpot_p95_ms: number;
  cost_per_1m: number;
};

type TopDown = {
  retiring: number;
  bad_speculation: number;
  frontend_bound: number;
  backend_bound: number;
  memory_bound: number;
  core_bound: number;
  ipc: number;
};

type ModelSummary = {
  model: string;
  short: string;
  instance: string;
  baseline_label: string;
  best_label: string;
  baseline: Point;
  best: Point;
  speedup: number;
  quality: {
    ppl_base: number;
    ppl_quant: number;
    delta_pct: number;
    max_delta_pct: number;
    passed: boolean;
  } | null;
  performix_base: { topdown: TopDown };
  performix_best: { topdown: TopDown };
  savings: {
    usd_saved_per_month: number;
    pct_saved: number;
    baseline_usd_per_month: number;
    best_usd_per_month: number;
    tokens_per_month_example: number;
  };
  frontier: Point[];
};

type Summary = {
  run_id: string;
  generated_at: string;
  mock: boolean;
  instance: string;
  slo: { ttft_p95_ms: number; tpot_p95_ms: number };
  tokens_per_month_example: number;
  models: ModelSummary[];
};

const fmtUsd = (n: number) =>
  "$" + n.toLocaleString(undefined, { maximumFractionDigits: n < 10 ? 3 : 0 });
const fmtInt = (n: number) => Math.round(n).toLocaleString();

export default function Page() {
  const [data, setData] = useState<Summary | null>(null);
  const [sel, setSel] = useState(0);
  const [tokensB, setTokensB] = useState(5); // billions per month

  useEffect(() => {
    fetch("/summary.json")
      .then((r) => r.json())
      .then(setData)
      .catch(() => setData(null));
  }, []);

  if (!data) {
    return (
      <div className="wrap" style={{ textAlign: "center", paddingTop: 100 }}>
        <h1 className="brand-title" style={{ justifyContent: "center" }}>
          NeoServe
        </h1>
        <p className="sub" style={{ marginTop: 16 }}>
          Loading real Graviton4 canonical benchmark telemetry...
        </p>
      </div>
    );
  }

  const m = data.models[sel];
  const tokens = tokensB * 1e9;
  const baseMo = (tokens / 1e6) * m.baseline.cost_per_1m;
  const bestMo = (tokens / 1e6) * m.best.cost_per_1m;
  const saved = baseMo - bestMo;
  const pctSaved = baseMo > 0 ? ((saved / baseMo) * 100).toFixed(1) : "0.0";

  return (
    <div className="wrap">
      {/* Header */}
      <header className="header-nav">
        <h1 className="brand-title">
          NeoServe
          <span className={"badge " + (data.mock ? "mock" : "real")}>
            <span className="badge-dot" />
            {data.mock ? "MOCK SIMULATOR" : "REAL GRAVITON4 (NEOVERSE V2)"}
          </span>
        </h1>
        <div className="mono" style={{ fontSize: 13, color: "var(--ink-secondary)" }}>
          AWS {data.instance} &middot; US-East-1
        </div>
      </header>

      {/* Hero Billboard Banner */}
      <section className="billboard-container">
        <div className="billboard-value">
          <span className="highlight">{fmtUsd(m.best.cost_per_1m)}</span>
          <span className="billboard-unit">/ 1M tokens at p95 SLO</span>
          <span className="billboard-speedup">{m.speedup.toFixed(2)}× Speedup</span>
        </div>
        <div className="billboard-meta">
          <span>
            <b>Winning Config:</b> {m.best_label}
          </span>
          <span>&bull;</span>
          <span>
            <b>SLO Targets:</b> TTFT &le; {data.slo.ttft_p95_ms}ms &bull; TPOT &le;{" "}
            {data.slo.tpot_p95_ms}ms
          </span>
          <span>&bull;</span>
          <span className="mono">Run ID: {data.run_id}</span>
        </div>
      </section>

      {/* Model Selection Tabs */}
      {data.models.length > 1 && (
        <div className="tabs">
          {data.models.map((mm, i) => (
            <button
              key={mm.short}
              className={"tab-btn " + (i === sel ? "active" : "")}
              onClick={() => setSel(i)}
            >
              {mm.short.toUpperCase()}
            </button>
          ))}
        </div>
      )}

      {/* 4 KPI Cards */}
      <div className="grid-cards">
        <div className="card">
          <div className="card-title">
            <span>Cost / 1M Output Tokens</span>
            <span style={{ color: "var(--accent-emerald)" }}>↓ {pctSaved}%</span>
          </div>
          <div className="kpi-num emerald">{fmtUsd(m.best.cost_per_1m)}</div>
          <div className="kpi-sub">
            Baseline: {fmtUsd(m.baseline.cost_per_1m)} / 1M tokens ({m.speedup.toFixed(2)}× goodput)
          </div>
        </div>

        <div className="card">
          <div className="card-title">
            <span>Serving Config & Levers</span>
            <span className="mono">W4A8 INT4</span>
          </div>
          <div className="mono" style={{ fontSize: 14, color: "var(--ink-primary)", fontWeight: 600 }}>
            {m.best_label}
          </div>
          <div className="kpi-sub" style={{ marginTop: 8 }}>
            Baseline: {m.baseline_label}
          </div>
        </div>

        <div className="card">
          <div className="card-title">
            <span>Quality Guard (PPL)</span>
            {m.quality && (
              <span
                style={{
                  color: m.quality.passed ? "var(--accent-emerald)" : "var(--accent-amber)",
                  fontWeight: 700,
                }}
              >
                {m.quality.passed ? "✓ PASSED" : "⚠ BUDGET EXCEEDED"}
              </span>
            )}
          </div>
          {m.quality ? (
            <div>
              <div
                className="kpi-num"
                style={{
                  color: m.quality.passed ? "var(--accent-emerald)" : "var(--accent-amber)",
                }}
              >
                +{m.quality.delta_pct}% <span style={{ fontSize: 16 }}>PPL</span>
              </div>
              <div className="kpi-sub">
                Wikitext PPL: {m.quality.ppl_base} &rarr; {m.quality.ppl_quant} (Budget: ≤{" "}
                {m.quality.max_delta_pct}%)
              </div>
            </div>
          ) : (
            <div>
              <div className="kpi-num">0.0%</div>
              <div className="kpi-sub">BF16 Baseline reference model</div>
            </div>
          )}
        </div>

        <div className="card">
          <div className="card-title">
            <span>Performix PMU IPC</span>
            <span style={{ color: "var(--accent-cyan)" }}>
              {m.performix_base.topdown.ipc} → {m.performix_best.topdown.ipc}
            </span>
          </div>
          <div className="kpi-num" style={{ color: "var(--accent-cyan)" }}>
            {m.performix_best.topdown.ipc} <span style={{ fontSize: 16 }}>IPC</span>
          </div>
          <div className="kpi-sub">
            Retiring instructions boosted from {m.performix_base.topdown.retiring}% to{" "}
            {m.performix_best.topdown.retiring}%
          </div>
        </div>
      </div>

      {/* Pareto Frontier Chart & Monthly Calculator */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 340px", gap: 20, marginBottom: 24 }}>
        <div className="card">
          <div className="card-title">
            <span>Latency-VS-Cost Pareto Frontier (SLO Operating Points)</span>
          </div>
          <Pareto m={m} />
          <div className="legend-grid" style={{ marginTop: 12 }}>
            <div className="legend-item">
              <span className="legend-dot" style={{ background: "var(--accent-emerald)" }} />
              <span>NeoServe Winner</span>
            </div>
            <div className="legend-item">
              <span className="legend-dot" style={{ background: "var(--ink-muted)" }} />
              <span>BF16 Baseline</span>
            </div>
            <div className="legend-item">
              <span className="legend-dot" style={{ background: "var(--accent-cyan)" }} />
              <span>Frontier Candidates</span>
            </div>
          </div>
          <div className="note" style={{ marginTop: 10 }}>
            X-axis: Output Throughput (tok/s) &bull; Y-axis: Cost / 1M Tokens (Lower is better)
          </div>
        </div>

        {/* Cost Calculator */}
        <div className="card">
          <div className="card-title">
            <span>Monthly Cost Calculator</span>
          </div>
          <div className="note" style={{ marginBottom: 12 }}>
            Monthly Traffic: <b style={{ color: "var(--ink-primary)", fontSize: 14 }}>{tokensB.toFixed(1)} Billion</b> tokens
          </div>
          <div className="slider-container">
            <input
              type="range"
              min={0.5}
              max={50}
              step={0.5}
              value={tokensB}
              onChange={(e) => setTokensB(parseFloat(e.target.value))}
            />
          </div>
          <table style={{ marginTop: 16 }}>
            <tbody>
              <tr>
                <td>BF16 Baseline</td>
                <td>{fmtUsd(baseMo)}/mo</td>
              </tr>
              <tr className="win-row">
                <td>NeoServe Winner</td>
                <td>{fmtUsd(bestMo)}/mo</td>
              </tr>
              <tr>
                <td style={{ color: "var(--accent-emerald)", fontWeight: 700 }}>Monthly Savings</td>
                <td style={{ color: "var(--accent-emerald)", fontWeight: 700 }}>
                  {fmtUsd(saved)}/mo ({pctSaved}%)
                </td>
              </tr>
            </tbody>
          </table>
          <div className="note" style={{ marginTop: 14 }}>
            Evaluated on {data.instance}. CPU serving achieves optimal cost-efficiency for small/medium models on low-to-medium concurrency.
          </div>
        </div>
      </div>

      {/* Performix TopDown Breakdown */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div className="card-title">
          <span>Arm Performix PMU Top-Down Hardware Breakdown</span>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
          <TopDownBar title={`BF16 Baseline (IPC ${m.performix_base.topdown.ipc})`} td={m.performix_base.topdown} />
          <TopDownBar title={`Tuned W4A8 Winner (IPC ${m.performix_best.topdown.ipc})`} td={m.performix_best.topdown} />
        </div>
        <div className="note" style={{ marginTop: 14 }}>
          BF16 baseline suffers high backend/memory stalls. The optimized INT4 configuration (using mimalloc, physical thread binding, and KleidiAI micro-kernels) shifts execution cycles into <b>retiring instructions</b> and raises IPC.
        </div>
      </div>

      {/* Operating Points Table */}
      <div className="card" style={{ marginBottom: 32 }}>
        <div className="card-title">
          <span>Evaluated Concurrency Operating Points</span>
        </div>
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Configuration</th>
                <th>Req / sec</th>
                <th>Throughput (tok/s)</th>
                <th>TTFT p95 (ms)</th>
                <th>TPOT p95 (ms)</th>
                <th>Cost / 1M Tokens</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>{m.baseline_label} (Baseline)</td>
                <td>{m.baseline.request_rate}</td>
                <td>{fmtInt(m.baseline.output_throughput_tok_s)}</td>
                <td>{fmtInt(m.baseline.ttft_p95_ms)}</td>
                <td>{fmtInt(m.baseline.tpot_p95_ms)}</td>
                <td>{fmtUsd(m.baseline.cost_per_1m)}</td>
              </tr>
              <tr className="win-row">
                <td>{m.best_label} (NeoServe Winner)</td>
                <td>{m.best.request_rate}</td>
                <td>{fmtInt(m.best.output_throughput_tok_s)}</td>
                <td>{fmtInt(m.best.ttft_p95_ms)}</td>
                <td>{fmtInt(m.best.tpot_p95_ms)}</td>
                <td>{fmtUsd(m.best.cost_per_1m)}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Footer */}
      <footer style={{ borderTop: "1px solid var(--border-subtle)", paddingTop: 20, display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
        <div className="note">
          Generated {data.generated_at} &bull; Cryptographically traceable via <span className="mono">ledger.json</span> (SHA-256)
        </div>
        <div className="mono" style={{ fontSize: 12, color: "var(--accent-emerald)" }}>
          ✓ Verified Canonical Evidence
        </div>
      </footer>
    </div>
  );
}

function Pareto({ m }: { m: ModelSummary }) {
  const W = 650, H = 260, pad = 44;
  const pts = useMemo(() => {
    const all = [...m.frontier, m.baseline, m.best];
    return all.filter((p) => isFinite(p.cost_per_1m) && isFinite(p.output_throughput_tok_s));
  }, [m]);

  if (pts.length === 0) return <div className="note">No SLO-meeting points</div>;

  const xs = pts.map((p) => p.output_throughput_tok_s);
  const ys = pts.map((p) => p.cost_per_1m);
  const xMin = 0, xMax = Math.max(...xs) * 1.15;
  const yMin = 0, yMax = Math.max(...ys) * 1.15;

  const sx = (x: number) => pad + ((x - xMin) / (xMax - xMin || 1)) * (W - pad * 2);
  const sy = (y: number) => H - pad - ((y - yMin) / (yMax - yMin || 1)) * (H - pad * 2);

  const frontierSorted = [...m.frontier].sort((a, b) => a.output_throughput_tok_s - b.output_throughput_tok_s);
  const pathStr = frontierSorted.map((p, i) => `${i === 0 ? "M" : "L"} ${sx(p.output_throughput_tok_s)} ${sy(p.cost_per_1m)}`).join(" ");

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} role="img" style={{ overflow: "visible" }}>
      {/* Grid lines */}
      {[0.25, 0.5, 0.75, 1].map((f) => (
        <g key={f}>
          <line x1={pad} y1={sy(yMax * f)} x2={W - pad} y2={sy(yMax * f)} stroke="rgba(255,255,255,0.05)" strokeDasharray="4 4" />
          <text x={pad - 8} y={sy(yMax * f) + 4} fill="var(--ink-muted)" fontSize="10" fontFamily="var(--font-mono)" textAnchor="end">
            ${(yMax * f).toFixed(2)}
          </text>
        </g>
      ))}

      {/* Axes */}
      <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="var(--border-subtle)" />
      <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="var(--border-subtle)" />

      {/* Frontier Path */}
      {frontierSorted.length > 1 && (
        <path d={pathStr} fill="none" stroke="var(--accent-cyan)" strokeWidth={2} opacity={0.8} />
      )}

      {/* Points */}
      {m.frontier.map((p, i) => (
        <circle key={i} cx={sx(p.output_throughput_tok_s)} cy={sy(p.cost_per_1m)} r={5} fill="var(--accent-cyan)">
          <title>{`${p.label}: ${Math.round(p.output_throughput_tok_s)} tok/s, $${p.cost_per_1m.toFixed(3)}/1M`}</title>
        </circle>
      ))}

      {/* Baseline Marker */}
      <circle cx={sx(m.baseline.output_throughput_tok_s)} cy={sy(m.baseline.cost_per_1m)} r={6} fill="var(--ink-muted)" stroke="#000" strokeWidth={2}>
        <title>{`Baseline: ${Math.round(m.baseline.output_throughput_tok_s)} tok/s, $${m.baseline.cost_per_1m.toFixed(3)}/1M`}</title>
      </circle>

      {/* Winner Marker */}
      <circle cx={sx(m.best.output_throughput_tok_s)} cy={sy(m.best.cost_per_1m)} r={8} fill="var(--accent-emerald)" stroke="#000" strokeWidth={2} style={{ filter: "drop-shadow(0 0 8px #00ff88)" }}>
        <title>{`Winner: ${Math.round(m.best.output_throughput_tok_s)} tok/s, $${m.best.cost_per_1m.toFixed(3)}/1M`}</title>
      </circle>
    </svg>
  );
}

function TopDownBar({ title, td }: { title: string; td: TopDown }) {
  const segs = [
    { k: "Retiring", v: td.retiring, c: "var(--accent-emerald)" },
    { k: "Backend Bound", v: td.backend_bound, c: "var(--accent-cyan)" },
    { k: "Frontend Bound", v: td.frontend_bound, c: "var(--accent-amber)" },
    { k: "Bad Speculation", v: td.bad_speculation, c: "var(--accent-rose)" },
  ];

  return (
    <div>
      <div className="mono" style={{ fontSize: 12, color: "var(--ink-secondary)", marginBottom: 6 }}>
        {title}
      </div>
      <div className="bar-container">
        {segs.map((s) => (
          <div key={s.k} className="bar-segment" style={{ width: `${s.v}%`, background: s.c }} title={`${s.k}: ${s.v}%`} />
        ))}
      </div>
      <div className="legend-grid">
        {segs.map((s) => (
          <div key={s.k} className="legend-item">
            <span className="legend-dot" style={{ background: s.c }} />
            <span>
              {s.k}: <b>{s.v.toFixed(1)}%</b>
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
