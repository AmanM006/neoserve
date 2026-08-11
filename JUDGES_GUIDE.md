# NeoServe — Judge Evaluation & Quickstart Guide

> **Arm Create: AI Optimization Challenge 2026 — Cloud AI (Track 2)**  
> **Goal:** Optimize AI serving workloads for Arm-powered cloud platforms (AWS Graviton4 / Neoverse V2) under real concurrency.

---

## ⚡ 60-Second Evaluation Path for Judges

Every metric and claim in NeoServe is **100% reproducible and cryptographically traceable**.

### 1. Verify Cryptographic Ledger & SHA-256 Provenance
```bash
# Verifies SHA-256 integrity over every canonical cost card, summary, model card, and recipe
python scripts/verify_ledger.py results/canonical
```
*Output: `ledger verified` (Zero mock data in REAL canonical results).*

### 2. View Interactive Pitch-Black Cyber Dashboard
```bash
cd dashboard
npm run refresh && npm run dev
```
Open **[http://localhost:3010](http://localhost:3010)** to inspect:
- **Billboard Metric**: **$0.7451 / 1M tokens** at p95 latency SLO on Graviton4 (`c8g.4xlarge`) — **1.94× speedup**, **48.5% cost reduction** ($3,505/mo saved).
- **Multi-Model Selector**: Switch between `Qwen2.5-1.5B` and `Llama-3.1-8B` benchmarks.
- **Latency-vs-Cost Pareto Frontier**: Interactive SVG plot of SLO-meeting operating points.
- **Arm Performix Top-Down PMU**: Instruction retirement (**27.5% → 51.8%**), IPC (**1.42 → 1.49**), and `kai_matmul_qai8` micro-kernel hotspots.
- **Interactive MCP Agent Playground**: Live interactive terminal testing `recommend_config`, `get_serving_recipe`, and `project_cost`.

### 3. Inspect Production Serving Recipe & Docker Compose
```bash
# Inspect tuned production container setup embedding physical thread binding & mimalloc
cat results/canonical/serving_recipe/qwen25-1p5b/compose.yaml
```

### 4. Test Model Context Protocol (MCP) Agent Interface
```bash
NEOSERVE_RESULTS=results/canonical PYTHONPATH=src python -m mcp.server
```
Exposes structured JSON tools for AI assistants (Cursor / Claude Desktop):
- `recommend_config(model="qwen25-1p5b")`
- `get_serving_recipe(model="qwen25-1p5b")`
- `project_cost(model="qwen25-1p5b", tokens_per_month=5000000000)`

---

## 🛡️ Technical Credibility Summary

| Evaluation Axis | NeoServe Implementation |
|---|---|
| **Objective** | Cost per 1M tokens at p95 latency SLO under Poisson traffic concurrency. |
| **Arm Hardware** | AWS Graviton4 `c8g.4xlarge` (16-core Arm Neoverse V2). |
| **Engine & Micro-kernels** | vLLM `aarch64` CPU backend + oneDNN JIT SMMLA + KleidiAI INT4 micro-kernels (`kai_matmul_qai8`). |
| **Quality Guard** | `lm_eval` wikitext word perplexity (+2.373% PPL, passed ≤ 4.0% budget). |
| **Arm Performix PMU** | `perf stat` hardware counters attached to baseline vs. winner. |
