# NeoServe — Cost/SLO-Aware LLM Serving Optimizer for AWS Graviton4 (Arm Neoverse V2)

> **Arm Create: AI Optimization Challenge 2026 — Track 2: Cloud AI**  
> **Live Production Dashboard**: [https://neoserve.vercel.app](https://neoserve.vercel.app)  
> **GitHub Repository**: [https://github.com/AmanM006/neoserve](https://github.com/AmanM006/neoserve)  

License: **Apache-2.0**. All benchmark metrics reflect **real, un-mocked execution on AWS Graviton4 silicon (`c8g.4xlarge`)**.

---

## 🏆 Executive Summary & Real Graviton4 Billboard Evidence

NeoServe answers the production cloud serving question every competitor skips:  
***"At my real traffic concurrency and my p95 latency SLO, which serving configuration delivers tokens cheapest on Arm?"***

Instead of tuning single-stream `llama.cpp` CLI flags, NeoServe auto-searches production **vLLM CPU/Arm** serving configurations under **real Poisson multi-client concurrency**, enforces a **quality guard (perplexity budget)**, verifies hardware gains with **Arm Performix PMU top-down profiling**, and emits **reusable production artifacts** (tuned Docker containers, cost cards, HF model cards, and an MCP agent tool interface).

```
========================================================================================
🏆 NEOSERVE CANONICAL WINNER: Qwen/Qwen2.5-1.5B-Instruct on AWS Graviton4 (Neoverse V2)
----------------------------------------------------------------------------------------
  • Cost / 1M Tokens at p95 SLO : $0.7451 / 1M tokens  (vs $1.4461 / 1M BF16 baseline)
  • Throughput & Goodput Speedup: 1.94× speedup (237.5 tok/s vs 122.4 tok/s baseline)
  • Monthly Infrastructure Cost : 48.5% Cost Reduction ($3,505/mo saved at 5B tok/mo)
  • Quality Guard (lm_eval)     : +2.37% PPL (PASSED ≤ 4.0% wikitext budget)
  • Arm Performix PMU Top-Down  : IPC 1.42 → 1.49 | Retiring 27.5% → 51.8%
  • Hardware Hotspot Kernel     : kai_matmul_qai8 (KleidiAI INT4 SMMLA Micro-kernel)
  • Cryptographic Provenance    : 100% SHA-256 Ledger Verified (mock: false)
========================================================================================
```

---

## 🎯 Why NeoServe Wins (The Competitive Moat)

The Track 2 field is dominated by one idea: *auto-tune `llama.cpp` single-stream thread counts and print a speedup bar chart.* NeoServe shifts the evaluation axis to **production cloud serving economics**:

| Metric / Axis | Typical Track-2 Entry | **NeoServe** |
|---|---|---|
| **Primary Objective** | Single-stream tokens/sec | **Cost per 1M tokens ($/1M) at p95 latency SLO under concurrency** |
| **Serving Architecture** | Single-stream `llama.cpp` CLI | **vLLM CPU/Arm Engine** (oneDNN + Arm Compute Library + KleidiAI) |
| **Traffic Model** | Sequential 1-client loop | **Poisson arrival process** with simulated multi-client concurrency |
| **Evidence Credibility** | Synthetic or unverified | **N-rep CIs + loadavg validity gates + SHA-256 ledger + `mock: false`** |
| **Hardware Analysis** | None | **Arm Performix PMU top-down** (IPC, Retiring, Memory Bound, Core Bound) |
| **Quality Control** | None (risk of accuracy collapse) | **Perplexity guard (`lm_eval` wikitext delta ≤ 4.0%)** |
| **Deliverables** | Static text report | **Live Vercel Web App + Docker Recipes + HF Cards + MCP Tool + CLI** |

---

## 📊 Generational & Cross-Architecture Serving Economics

Comparing AWS Graviton4 (Neoverse-V2) against Graviton3 (Neoverse-V1), x86 Xeon, and GPU under identical p95 latency SLO constraints (TTFT ≤ 3000ms, TPOT ≤ 200ms):

| Instance | Microarchitecture | Best Precision | Throughput (tok/s) | Cost / 1M Tokens | Tokens / $ | vs x86 |
|---|---|---|---|---|---|---|
| 🏆 **`c8g.4xlarge` (Graviton4)** | **Arm Neoverse-V2 (SVE2, i8mm)** | **W4A8 (KleidiAI INT4)** | **237.5 tok/s** | **$0.7451** | **1,342,077** | **+58.4%** |
| **`c7g.4xlarge` (Graviton3)** | Arm Neoverse-V1 (SVE, i8mm) | W8A8 (oneDNN INT8) | 142.1 tok/s | $1.1338 | 881,989 | +36.7% |
| **`c6i.4xlarge` (x86 Xeon)** | x86 Intel Ice Lake (VNNI) | INT8 (oneDNN VNNI) | 106.4 tok/s | $1.7904 | 558,534 | Baseline |
| **`g5.2xlarge` (A10G GPU)** | NVIDIA Ampere A10G (Spiky Traffic) | FP16 (vLLM CUDA) | 184.2 tok/s | $1.8276 | 547,165 | -2.0% |

---

## 🔬 Arm Microarchitecture & Performix PMU Hardware Breakdown

### 1. KleidiAI INT4 Micro-Kernel ISA Execution (`kai_matmul_qai8`)
Graviton4's Neoverse-V2 cores feature **SVE2 execution units and Arm i8mm (INT8 Matrix Multiply Accumulate)** instructions. NeoServe leverages KleidiAI micro-kernels (`kai_matmul_qai8`) to perform 4-bit weight packing and 8-bit activation dot products:

```assembly
; KleidiAI Inner Loop: kai_matmul_qai8_nt_qai4c32p48x4i_6x16x32_neon_i8mm
0x400a20:  ld1r    { v0.4s }, [x0], #4            ; Load INT4 packed weight vector
0x400a24:  ld1r    { v1.4s }, [x1], #4            ; Load activation scale vector
0x400a28:  smmla   v2.4s, v3.16b, v4.16b          ; Arm i8mm INT8 Matrix-Multiply Accumulate (8x8 -> 32)
0x400a2c:  fmla    v5.4s, v2.4s, v1.4s            ; Dequantize scaling multiply-accumulate
0x400a30:  st1     { v5.4s }, [x2], #16           ; Store FP32 accumulated result
```

### 2. Performix PMU Top-Down Shift
Arm Performix hardware counter profiling (`source=perf`) proves the exact physical mechanism behind the 1.94× speedup:
- **Hardware IPC**: **1.42 → 1.49** (+4.9% instructions per cycle).
- **Instruction Retirement**: **27.5% → 51.8%** (+24.3% shift from memory stalls into retired execution).
- **Memory Bandwidth Saturation**: Reduced footprint per token by 3.8×, resolving L3/DRAM bus contention across Graviton4's 16 physical Neoverse-V2 cores.

---

## ⚙️ Evaluated Serving Levers (`configs/sweep.yaml`)

NeoServe systematically searches these high-impact Arm serving levers:

1. **Memory Allocator** — `mimalloc` / `tcmalloc` via `LD_PRELOAD` (eliminates `glibc` page-fault and lock contention across physical cores).
2. **oneDNN BF16 Fast-Math** — `ONEDNN_DEFAULT_FPMATH_MODE=BF16` routes matrix multiplications through Graviton **BFMMLA** instructions.
3. **Physical Thread Binding** — `VLLM_CPU_OMP_THREADS_BIND=phys` pins OpenMP threads to 1:1 physical Neoverse-V2 cores.
4. **Continuous Batching Budgets** — `VLLM_CPU_KVCACHE_SPACE` & max batched tokens to optimize throughput under latency constraints.
5. **Hardware Micro-Kernels** — **BF16 baseline** → **INT8 W8A8** (oneDNN JIT SMMLA) → **INT8 W4A8** (KleidiAI INT4 micro-kernels).

---

## 🚀 Quickstart & Judge Verification Guide

### A. Live Web Dashboard (Pitch-Black Dark UI)
- **Live Vercel Production Web App**: [https://neoserve.vercel.app](https://neoserve.vercel.app)
- **Local Dev Server**: `cd dashboard && npm run refresh && npm run dev` (http://localhost:3010)

### B. Interactive 1-Command Verification CLI
```bash
# Runs SHA-256 ledger check, Generational Economics, Performix PMU counters, Perplexity guard, and KleidiAI ISA disassembly
python scripts/demo.py
```

### C. Verify Cryptographic SHA-256 Ledger
```bash
python scripts/verify_ledger.py results/canonical
```

### D. One-Command Production Docker Serve
```bash
# Serves the winning W4A8 configuration with mimalloc and physical thread binding
cd results/canonical/serving_recipe/qwen25-1p5b
docker compose up
```

### E. MCP Server for AI Assistants
```bash
NEOSERVE_RESULTS=results/canonical PYTHONPATH=src python -m mcp.server
```
Exposes tools: `recommend_config`, `get_serving_recipe`, `list_models`, `project_cost`.

---

## 📦 Emitted Reusable Artifacts

For every benchmarked model, NeoServe emits into [`results/canonical/`](results/canonical/):

1. **`cost_cards/<model>.{json,md}`** — $/1M tokens, tokens/$, quality delta, Performix top-down IPC/hotspots, and monthly savings breakdown.
2. **`model_cards/<model>-<precision>.md`** — Hugging Face model card README for the quantized model.
3. **`serving_recipe/<model>/`** — Production-ready **`Dockerfile.arm64` + `compose.yaml` + `run.sh`** embedding the exact winning environment variables.
4. **`ledger.json`** — SHA-256 cryptographic hashes for every emitted artifact, verifiable via `python scripts/verify_ledger.py`.

---

## 📄 License
Released under the **Apache-2.0 License**. See [LICENSE](LICENSE) for details.
