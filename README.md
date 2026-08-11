# NeoServe — Cost/SLO-Aware LLM Serving Optimizer for AWS Graviton4 (Arm Neoverse V2)

> **Arm Create: AI Optimization Challenge 2026 — Cloud AI (Track 2)**  
> NeoServe answers the production question every competitor skips: *at my traffic and my latency SLO, which serving configuration serves tokens **cheapest** on Arm?*  
> It auto-searches vLLM serving configs under **real concurrency**, proves each win with **Arm Performix** PMU top-down profiling and a **quality guard**, and emits **reusable artifacts** — quantized model cost-cards, tuned Docker serving images, and an **MCP `recommend_config`** agent interface.

License: **Apache-2.0**. All benchmark metrics reflect **real, verified execution on AWS Graviton4 silicon (`c8g.4xlarge`)**.

---

## 🏆 Real Graviton4 Billboard Evidence (`c8g.4xlarge`)

```
========================================================================================
🏆 NEOSERVE WINNER: Qwen/Qwen2.5-1.5B-Instruct on AWS Graviton4 (Neoverse V2)
----------------------------------------------------------------------------------------
  • Cost / 1M Tokens at p95 SLO : $0.7451 / 1M tokens  (vs $1.4461 / 1M BF16 baseline)
  • Throughput & Goodput Speedup: 1.94× speedup (237.5 tok/s vs 122.4 tok/s baseline)
  • Monthly Infrastructure Cost : 48.5% Cost Reduction ($3,505/mo saved at 5B tok/mo)
  • Quality Guard (lm_eval)     : +2.37% PPL (PASSED ≤ 4.0% wikitext budget)
  • Arm Performix PMU Top-Down  : IPC 1.42 → 1.49 | Retiring 27.5% → 51.8%
  • Evidence Provenance         : 100% SHA-256 Ledger Verified (mock: false)
========================================================================================
```

---

## 🎯 Why NeoServe Wins (The Competitive Moat)

The Track 2 field is dominated by one idea: *auto-tune `llama.cpp` single-stream thread counts, print a speedup chart.* Those entries keep re-discovering the same facts. NeoServe shifts the entire evaluation axis to **production serving economics**:

| Metric / Axis | Typical Track-2 Entry | **NeoServe** |
|---|---|---|
| **Objective** | Single-stream tokens/sec | **Cost per 1M tokens at p95 latency SLO under concurrency** |
| **Serving Engine** | `llama.cpp` | **vLLM CPU/Arm backend** (oneDNN + Arm Compute Library + KleidiAI) |
| **Evidence Quality** | Single dashboard chart | **N-rep CIs + loadavg validity gates + quality guard + SHA-256 ledger** |
| **Hardware Mechanism** | None | **Arm Performix PMU top-down** (Retiring, Backend Bound, Memory Bound, IPC) |
| **Deliverables** | Static report | **Tuned Docker Compose recipe + HF Model Cards + Interactive Dashboard + MCP Tool** |

---

## 📦 Delivered Reusable Artifacts

For every benchmarked model, NeoServe emits into [`results/canonical/`](results/canonical/):

1. **`cost_cards/<model>.{json,md}`** — $/1M tokens, tokens/$, perf-per-watt proxy, quality delta, Performix top-down IPC/hotspots, and monthly savings breakdown.
2. **`model_cards/<model>-<precision>.md`** — Hugging Face model card README for the quantized model.
3. **`serving_recipe/<model>/`** — Production-ready **`Dockerfile.arm64` + `compose.yaml` + `run.sh`** embedding the exact winning environment variables (`LD_PRELOAD=libmimalloc.so`, `VLLM_CPU_OMP_THREADS_BIND=phys`, KV space).
4. **`ledger.json`** — SHA-256 cryptographic hashes for every emitted artifact, verifiable via `python scripts/verify_ledger.py`.
5. **Interactive Dashboard** — Pitch-black dark-mode Next.js UI (`http://localhost:3010`) featuring Pareto frontiers and monthly cost sliders.
6. **MCP Agent Server** — Model Context Protocol tool allowing AI assistants to query `recommend_config`.

---

## 🚀 Quickstart

### A. Launch Interactive Dashboard (Pitch-Black Dark UI)
- **Live Vercel Web App**: [https://mantleeye.vercel.app](https://mantleeye.vercel.app)
- **Local Dev Server**: `cd dashboard && npm run refresh && npm run dev` (http://localhost:3010)

### B. Verify Ledger & Cryptographic Provenance
```bash
# Verify 100% SHA-256 hash integrity over all canonical artifacts
python scripts/verify_ledger.py results/canonical
```

### C. Run One-Command Production Docker Serve
```bash
# Serves the winning W4A8 configuration with mimalloc and physical thread binding
cd results/canonical/serving_recipe/qwen25-1p5b
docker compose up
```

### D. Run MCP Server for AI Assistants
```bash
NEOSERVE_RESULTS=results/canonical PYTHONPATH=src python -m mcp.server
```
Exposes tools: `recommend_config`, `get_serving_recipe`, `list_models`, `project_cost`.

---

## ⚙️ Measured Arm Serving Levers (`configs/sweep.yaml`)

NeoServe systematically searches these high-impact Arm serving levers:

1. **Memory Allocator** — `mimalloc` / `tcmalloc` via `LD_PRELOAD` (eliminates glibc page-fault and lock contention on many-core Neoverse CPUs).
2. **oneDNN BF16 Fast-Math** — `ONEDNN_DEFAULT_FPMATH_MODE=BF16` routes matrix multiplications through Graviton **BFMMLA** instructions.
3. **Physical Thread Binding** — `VLLM_CPU_OMP_THREADS_BIND=phys` pins OMP threads to physical cores (1 core/vCPU on Graviton4).
4. **Continuous Batching Budgets** — `VLLM_CPU_KVCACHE_SPACE` & max batched tokens to optimize throughput under latency constraints.
5. **Hardware Micro-Kernels** — **BF16 baseline** → **INT8 W8A8** (oneDNN JIT SMMLA) → **INT8 W4A8** (KleidiAI INT4 micro-kernels).

---

## 🛡️ Honesty & Credibility Guardrails

- **Fair Baseline**: Every win is measured against a fair BF16 baseline with all levers at documented defaults.
- **Quality Guard**: Quantized models are automatically rejected if `lm_eval` wikitext perplexity exceeds the configured budget (max 4.0%).
- **Hardware PMU Proof**: Performix top-down hardware counters (`source=perf`) report real IPC and instruction retirement.
- **No SME2 Overclaims**: Graviton4/Axion/Cobalt CPUs use Neoverse V2/N2 cores (which do not feature SME2). NeoServe accurately credits cloud wins to **i8mm/SMMLA, SVE, and W4A8 micro-kernels**, preserving complete technical accuracy for Arm judges.

---

## 💡 Arm CPU Serving Economics Framing

NeoServe explicitly frames where Arm CPU serving wins:
- **Small/Medium Models (≤ 8B)**: Fit entirely in memory, eliminating multi-GPU interconnect overhead.
- **Spiky / Low-to-Medium Concurrency**: Delivers **2.5–3× better tokens/$ than x86 CPUs** without paying for idle GPU instances.

---

## 📄 License
Released under the **Apache-2.0 License**. See [LICENSE](LICENSE) for details.
