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
  retiring: number; bad_speculation: number; frontend_bound: number;
  backend_bound: number; memory_bound: number; core_bound: number; ipc: number;
};
type ModelSummary = {
  model: string; short: string; instance: string;
  baseline_label: string; best_label: string;
  baseline: Point; best: Point; speedup: number;
  quality: { ppl_base: number; ppl_quant: number; delta_pct: number; max_delta_pct: number; passed: boolean } | null;
  performix_base: { topdown: TopDown }; performix_best: { topdown: TopDown };
  savings: { usd_saved_per_month: number; pct_saved: number; baseline_usd_per_month: number; best_usd_per_month: number; tokens_per_month_example: number };
  frontier: Point[];
};
type Summary = {
  run_id: string; generated_at: string; mock: boolean; instance: string;
  slo: { ttft_p95_ms: number; tpot_p95_ms: number };
  tokens_per_month_example: number; models: ModelSummary[];
};

const fmtUsd = (n: number) => "$" + n.toLocaleString(undefined, { maximumFractionDigits: n < 10 ? 3 : 0 });
const fmtInt = (n: number) => Math.round(n).toLocaleString();

export default function Page() {
  const [data, setData] = useState<Summary | null>(null);
  const [sel, setSel] = useState(0);
  const [tokensB, setTokensB] = useState(5); // billions/month

  useEffect(() => {
    fetch("/summary.json").then((r) => r.json()).then(setData).catch(() => setData(null));
  }, []);

  if (!data) {
    return (
      <div className="wrap">
        <p className="h1">NeoServe</p>
        <p className="sub">
          Loading run data. Generate it with <span className="mono">python -m harness.runner --mock</span> then{" "}
          <span className="mono">npm run refresh</span>.
        </p>
      </div>
    );
  }

  const m = data.models[sel];
  const tokens = tokensB * 1e9;
  const baseMo = (tokens / 1e6) * m.baseline.cost_per_1m;
  const bestMo = (tokens / 1e6) * m.best.cost_per_1m;
  const saved = baseMo - bestMo;

  return (
    <div className="wrap">
      <p className="h1">
        NeoServe
        <span className={"badge " + (data.mock ? "mock" : "real")}>{data.mock ? "MOCK simulator" : "REAL Graviton4"}</span>
      </p>
      <p className="sub">
        Cost/SLO-aware LLM serving optimizer for AWS Graviton4 (Neoverse V2) &middot; {data.instance} &middot; SLO: TTFT p95 &le;{" "}
        {data.slo.ttft_p95_ms}ms, TPOT p95 &le; {data.slo.tpot_p95_ms}ms &middot; run {data.run_id}
      </p>

      <div className="tabs">
        {data.models.map((mm, i) => (
          <button key={mm.short} className={"tab " + (i === sel ? "active" : "")} onClick={() => setSel(i)}>
            {mm.short}
          </button>
        ))}
      </div>

      {/* KPI row */}
      <div className="row" style={{ marginBottom: 16 }}>
        <div className="card">
          <h3>Cost / 1M output tokens</h3>
          <div className="kpi good">{fmtUsd(m.best.cost_per_1m)}</div>
          <div><small>baseline {fmtUsd(m.baseline.cost_per_1m)} &middot; {m.speedup.toFixed(2)}x more SLO-meeting goodput</small></div>
        </div>
        <div className="card">
          <h3>Winning config</h3>
          <div className="mono" style={{ fontSize: 13 }}>{m.best_label}</div>
          <div className="note">baseline: {m.baseline_label}</div>
        </div>
        <div className="card">
          <h3>Quality guard</h3>
          {m.quality ? (
            <div>
              <div className="kpi" style={{ fontSize: 22, color: m.quality.passed ? "var(--accent)" : "var(--warn)" }}>
                {m.quality.passed ? "PASS" : "FAIL"} <small>{m.quality.delta_pct}% ppl</small>
              </div>
              <div className="note">{m.quality.ppl_base} &rarr; {m.quality.ppl_quant} (budget {m.quality.max_delta_pct}%)</div>
            </div>
          ) : (
            <div className="note">bf16 winner (no quantization)</div>
          )}
        </div>
      </div>

      <div className="row">
        {/* Pareto frontier */}
        <div className="card" style={{ flex: 2 }}>
          <h3>Latency-vs-cost Pareto frontier (SLO-meeting operating points)</h3>
          <Pareto m={m} />
          <div className="legend" style={{ marginTop: 8 }}>
            <span><span className="dot" style={{ background: "var(--accent)" }} />winner</span>
            <span><span className="dot" style={{ background: "var(--muted)" }} />baseline</span>
            <span><span className="dot" style={{ background: "var(--accent2)" }} />frontier configs</span>
          </div>
          <div className="note">x: SLO-meeting output throughput (tok/s) &middot; y: cost per 1M tokens (lower is better)</div>
        </div>

        {/* Cost calculator */}
        <div className="card">
          <h3>Monthly cost calculator</h3>
          <div className="note">Traffic: <b>{tokensB.toFixed(1)}B</b> output tokens / month</div>
          <input type="range" min={0.1} max={50} step={0.1} value={tokensB} onChange={(e) => setTokensB(parseFloat(e.target.value))} />
          <table style={{ marginTop: 10 }}>
            <tbody>
              <tr><td>Baseline ({m.baseline_label.split(" ")[0]})</td><td>{fmtUsd(baseMo)}/mo</td></tr>
              <tr className="win"><td>NeoServe winner</td><td>{fmtUsd(bestMo)}/mo</td></tr>
              <tr><td><b>Saved</b></td><td><b>{fmtUsd(saved)}/mo ({((saved / baseMo) * 100 || 0).toFixed(0)}%)</b></td></tr>
            </tbody>
          </table>
          <div className="note">Instance {m.instance}. CPU serving wins on tokens/$ vs x86 CPU and on spiky/low-concurrency traffic vs GPU.</div>
        </div>
      </div>

      {/* Performix top-down */}
      <div className="card" style={{ marginTop: 16 }}>
        <h3>Arm Performix top-down &mdash; why it&apos;s faster</h3>
        <div className="row">
          <TopDownBar title={`baseline (IPC ${m.performix_base.topdown.ipc})`} td={m.performix_base.topdown} />
          <TopDownBar title={`winner (IPC ${m.performix_best.topdown.ipc})`} td={m.performix_best.topdown} />
        </div>
        <div className="note">
          bf16 baseline is backend/memory-bound; the tuned INT8/INT4 config shifts cycles into <b>retiring</b> via i8mm/SMMLA
          {m.best_label.includes("w4a8") ? " + KleidiAI INT4" : ""} kernels and oneDNN weight prepacking, lifting IPC.
        </div>
      </div>

      {/* Operating-point table */}
      <div className="card" style={{ marginTop: 16 }}>
        <h3>Operating points</h3>
        <table>
          <thead>
            <tr><th>config</th><th>req/s</th><th>throughput tok/s</th><th>TTFT p95 ms</th><th>TPOT p95 ms</th><th>$/1M tok</th></tr>
          </thead>
          <tbody>
            <tr><td>{m.baseline_label} (baseline)</td><td>{m.baseline.request_rate}</td><td>{fmtInt(m.baseline.output_throughput_tok_s)}</td><td>{fmtInt(m.baseline.ttft_p95_ms)}</td><td>{fmtInt(m.baseline.tpot_p95_ms)}</td><td>{fmtUsd(m.baseline.cost_per_1m)}</td></tr>
            <tr className="win"><td>{m.best_label} (winner)</td><td>{m.best.request_rate}</td><td>{fmtInt(m.best.output_throughput_tok_s)}</td><td>{fmtInt(m.best.ttft_p95_ms)}</td><td>{fmtInt(m.best.tpot_p95_ms)}</td><td>{fmtUsd(m.best.cost_per_1m)}</td></tr>
          </tbody>
        </table>
      </div>

      <p className="note" style={{ marginTop: 24 }}>
        Generated {data.generated_at}. Numbers are traceable via the run&apos;s <span className="mono">ledger.json</span> (SHA-256).
        {data.mock && " Mock run: re-run with --real on Graviton4 before submission."}
      </p>
    </div>
  );
}

function Pareto({ m }: { m: ModelSummary }) {
  const W = 560, H = 300, pad = 44;
  const pts = useMemo(() => {
    const all = [...m.frontier, m.baseline, m.best];
    return all.filter((p) => isFinite(p.cost_per_1m) && isFinite(p.output_throughput_tok_s));
  }, [m]);
  if (pts.length === 0) return <div className="note">no SLO-meeting points</div>;
  const xs = pts.map((p) => p.output_throughput_tok_s);
  const ys = pts.map((p) => p.cost_per_1m);
  const xMin = 0, xMax = Math.max(...xs) * 1.1;
  const yMin = 0, yMax = Math.max(...ys) * 1.1;
  const sx = (x: number) => pad + ((x - xMin) / (xMax - xMin || 1)) * (W - pad * 2);
  const sy = (y: number) => H - pad - ((y - yMin) / (yMax - yMin || 1)) * (H - pad * 2);

  const frontierSorted = [...m.frontier].sort((a, b) => a.output_throughput_tok_s - b.output_throughput_tok_s);
  const path = frontierSorted.map((p, i) => `${i === 0 ? "M" : "L"} ${sx(p.output_throughput_tok_s)} ${sy(p.cost_per_1m)}`).join(" ");

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} role="img">
      {/* axes */}
      <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="var(--line)" />
      <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="var(--line)" />
      {[0.25, 0.5, 0.75, 1].map((f) => (
        <g key={f}>
          <text x={pad - 6} y={sy(yMax * f)} fill="var(--muted)" fontSize="10" textAnchor="end">${(yMax * f).toFixed(2)}</text>
          <text x={sx(xMax * f)} y={H - pad + 14} fill="var(--muted)" fontSize="10" textAnchor="middle">{Math.round(xMax * f)}</text>
        </g>
      ))}
      {/* frontier line */}
      {frontierSorted.length > 1 && <path d={path} fill="none" stroke="var(--accent2)" strokeWidth={1.5} opacity={0.8} />}
      {/* frontier points */}
      {m.frontier.map((p, i) => (
        <circle key={i} cx={sx(p.output_throughput_tok_s)} cy={sy(p.cost_per_1m)} r={4} fill="var(--accent2)" opacity={0.85}>
          <title>{p.label} @ {p.request_rate} req/s: {Math.round(p.output_throughput_tok_s)} tok/s, ${p.cost_per_1m.toFixed(3)}/1M</title>
        </circle>
      ))}
      {/* baseline + winner markers */}
      <circle cx={sx(m.baseline.output_throughput_tok_s)} cy={sy(m.baseline.cost_per_1m)} r={6} fill="var(--muted)" stroke="#000">
        <title>baseline: {Math.round(m.baseline.output_throughput_tok_s)} tok/s, ${m.baseline.cost_per_1m.toFixed(3)}/1M</title>
      </circle>
      <circle cx={sx(m.best.output_throughput_tok_s)} cy={sy(m.best.cost_per_1m)} r={7} fill="var(--accent)" stroke="#000">
        <title>winner: {Math.round(m.best.output_throughput_tok_s)} tok/s, ${m.best.cost_per_1m.toFixed(3)}/1M</title>
      </circle>
    </svg>
  );
}

function TopDownBar({ title, td }: { title: string; td: TopDown }) {
  const segs = [
    { k: "retiring", v: td.retiring, c: "var(--accent)" },
    { k: "backend", v: td.backend_bound, c: "var(--accent2)" },
    { k: "frontend", v: td.frontend_bound, c: "var(--warn)" },
    { k: "bad spec", v: td.bad_speculation, c: "#e06c75" },
  ];
  return (
    <div style={{ flex: 1, minWidth: 220 }}>
      <div className="note" style={{ marginBottom: 6 }}>{title}</div>
      <div style={{ display: "flex", borderRadius: 4, overflow: "hidden" }}>
        {segs.map((s) => (
          <div key={s.k} className="bar" style={{ width: `${s.v}%`, background: s.c }} title={`${s.k}: ${s.v}%`} />
        ))}
      </div>
      <div className="legend" style={{ marginTop: 6 }}>
        {segs.map((s) => (
          <span key={s.k}><span className="dot" style={{ background: s.c }} />{s.k} {s.v.toFixed(0)}%</span>
        ))}
      </div>
    </div>
  );
}
