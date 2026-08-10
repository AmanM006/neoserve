# NeoServe — Devpost submission (Arm Create: AI Optimization Challenge 2026)

## Elevator pitch
NeoServe finds the **cheapest SLO-meeting LLM serving config on AWS Graviton4** — not the fastest single-stream llama.cpp knob. It sweeps vLLM under real concurrency, scores **$/1M tokens at p95 TTFT/TPOT**, quality-guards quants, attaches a PMU/`perf` mechanism profile, and ships reusable cost cards + Docker recipes + MCP `recommend_config`.

## The problem everyone else skips
Track 2 is flooded with autotuners that maximize single-stream tokens/sec. Production serving asks a different question: *at my traffic and my latency SLO, what config is cheapest on Arm?*

## What we built
- Successive-halving search over allocator (`mimalloc`/`tcmalloc`), thread binding, KV space, batch size, and precision (BF16 → W4A8).
- Concurrency grid under real Poisson traffic with N-rep CIs and loadavg validity gates.
- Quality guard (`lm_eval` wikitext perplexity) to ensure quality budget isn't breached before promoting a quant.
- Mechanism profile: Arm Performix PMU top-down analysis + `perf stat` hardware counter breakdown (retiring, IPC, memory bound).
- Reusable artifacts: JSON/MD cost cards, Hugging Face model cards, tuned `Dockerfile.arm64` + `compose.yaml`, SHA-256 ledger, Next.js dashboard, and MCP `recommend_config` tool.

## Billboard (Real Graviton4 `c8g.4xlarge` Evidence)
> **$0.7451 / 1M tokens** at p95 SLO (TTFT ≤ 3000ms, TPOT ≤ 200ms) on **c8g.4xlarge** (Neoverse V2)
> **1.94× throughput speedup** vs BF16 baseline ($1.4461 / 1M tokens) — **48.5% cost savings** ($3,505/mo on 5B tokens)
> Quality delta: **+2.37% PPL** (passed ≤ 4.0% budget) — IPC **1.42 → 1.49** — `mock: false` — SHA-256 Ledger Verified

## Demo & Reproduce
1. **Ledger Verification**:
   ```bash
   python scripts/verify_ledger.py results/canonical
   ```
2. **One-Command Production Serve**:
   ```bash
   cd results/canonical/serving_recipe/qwen25-1p5b && docker compose up
   ```
3. **MCP Tool Integration**:
   ```python
   from mcp import recommend_config
   recommend_config(model="qwen25-1p5b")
   ```
4. **Offline Interactive Dashboard**:
   ```bash
   cd dashboard && npm run dev
   ```

## Built with
Python, vLLM (aarch64 CPU), llmcompressor, oneDNN / ACL / KleidiAI kernels, AWS Graviton4 (Neoverse V2), Next.js, Model Context Protocol (MCP)

## Verification & Integrity
Every emitted artifact is cryptographically hashed with SHA-256 in `results/canonical/ledger.json`. All metrics reflect real, un-mocked execution on Graviton4 hardware.
