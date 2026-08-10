# NeoServe — Cost/SLO-aware LLM serving optimizer for AWS Graviton4 (Arm Neoverse V2)

> **Arm Create: AI Optimization Challenge 2026 — Cloud AI (Track 2).**
> NeoServe answers the question every competitor skips: *at my traffic and my latency
> SLO, which serving config serves tokens **cheapest** on Arm?* It auto-searches vLLM
> serving configs under **real concurrency**, proves each win with **Arm Performix**
> PMU profiling and a **quality guard**, and ships **reusable artifacts** — quantized
> model cost-cards, a tuned serving Docker image, and an **MCP `recommend_config`** tool.

License: **Apache-2.0**. Everything is runnable offline in `--mock` mode; real numbers
come from `--real` on a Graviton4 (`c8g`) instance.

---

## Why this is different (and why it can win the overall)

The Track 2 field is crowded with one idea: *auto-tune `llama.cpp` threads/quant/KV
**single-stream**, print a KleidiAI dashboard.* Those entries keep re-discovering the
same three facts and compete on the same axis. NeoServe changes the axis:

| | Typical Track-2 entry | **NeoServe** |
|---|---|---|
| Objective | single-stream tokens/sec | **cost/1M tokens at a p95 SLO under concurrency** (the production question) |
| Engine | `llama.cpp` | **vLLM CPU/Arm backend** (oneDNN + Arm Compute Library + KleidiAI) |
| Evidence | one dashboard | **N≥5 reps + 95% CIs + validity gates + quality guard + SHA-256 provenance** |
| "Why" | none | **Arm Performix top-down** (PMU) attached to baseline vs winner |
| Output | a report | **reusable artifacts**: quantized model cost-cards, tuned Docker image, MCP tool |

The challenge rules *explicitly* invite Performix ("Developers can use Arm Performix to
get exact benchmarks…"). Almost nobody does. NeoServe bakes it in.

---

## What it produces

Running the harness on a model emits, per model, into `results/<run>/`:

- **`cost_cards/<model>.{json,md}`** — cost/1M tokens, tokens/$, perf-per-watt proxy,
  quality delta, Performix top-down, and a monthly-savings example.
- **`model_cards/<model>-<precision>.md`** — a Hugging Face README for the published
  quantized model.
- **`serving_recipe/<model>/`** — a **tuned `Dockerfile.arm64` + `compose.yaml` +
  `run.sh`** embedding the exact winning env/flags, so anyone reproduces the fast server.
- **`report.html`**, **`summary.json`**, **`raw/cells.jsonl`**, and **`ledger.json`**
  (SHA-256 over every file — every published number is traceable).

Plus an interactive **dashboard** (Pareto frontier + `$/mo` cost calculator + Performix
bars) and an **MCP server** so an AI assistant can ask `recommend_config`.

---

## Quickstart

### A. Offline (mock) — runs anywhere in ~5 seconds
```bash
pip install -r requirements.txt
PYTHONPATH=src python -m harness.runner --mock            # Linux/macOS
# Windows PowerShell:  $env:PYTHONPATH="src"; python -m harness.runner --mock
```
This runs the full pipeline (search → concurrency grid → quality guard → Performix →
artifacts → report) using a **grounded simulator** whose per-lever multipliers come from
*measured* vLLM-on-Arm results (see references). Every mock artifact is clearly tagged.

Dashboard:
```bash
cd dashboard && npm install && npm run refresh && npm run dev   # http://localhost:3010
```

### B. Real numbers on AWS Graviton4
```bash
# 1) provision a c8g spot instance (from your laptop; needs AWS CLI)
deploy/ec2-setup.sh provision --type c8g.4xlarge --key <key> --sg <sg> --subnet <subnet>

# 2) copy the repo over, then on the instance:
deploy/ec2-setup.sh bootstrap            # installs vLLM aarch64, mimalloc, llm-compressor, lm-eval

# 3) build quantized models (W8A8 near-lossless; W4A8 uses KleidiAI INT4)
PYTHONPATH=src python -m quantize.make_w8a8 --model meta-llama/Llama-3.1-8B-Instruct --out models/llama31-8b-w8a8
PYTHONPATH=src python -m quantize.make_w4a8 --model meta-llama/Llama-3.1-8B-Instruct --out models/llama31-8b-w4a8

# 4) run the real sweep (drives vLLM + Arm Performix over the concurrency grid)
PYTHONPATH=src python -m harness.runner --real --instance c8g.4xlarge

# 5) tear down when done (cost control)
deploy/ec2-setup.sh teardown
```

---

## The optimization space (`configs/sweep.yaml`)

NeoServe searches these **measured Arm serving levers**, then scores on cost-at-SLO:

- **Allocator** — `mimalloc`/`tcmalloc` via `LD_PRELOAD` (fixes glibc page-fault/lock
  contention on many-core Neoverse; the biggest low-concurrency win).
- **oneDNN BF16 fast-math** — `ONEDNN_DEFAULT_FPMATH_MODE=BF16` runs fp32/bf16 GEMM
  through Graviton **BFMMLA** (helps bf16; ~no-op on INT8 — a nuance we get right).
- **Thread binding** — `VLLM_CPU_OMP_THREADS_BIND` pins OMP threads to physical cores
  (Graviton = 1 core/vCPU, no SMT).
- **KV-cache space** (`VLLM_CPU_KVCACHE_SPACE`) and **max batched tokens** — the
  continuous-batching knobs that trade TTFT for throughput.
- **LSE atomics** — `LDADDAL` HW atomics vs LL/SC retry in libgomp.
- **Precision** — `bf16` (fair baseline) → **INT8 W8A8** (oneDNN JIT SMMLA/i8mm) →
  **INT8 W4A8** (KleidiAI INT4 micro-kernels), each quality-guarded.

Search uses **successive halving**: a cheap saturation probe prunes the pool before the
expensive concurrency grid (N reps + CIs).

### Honesty guardrails (baked into the code)
- **Fair baseline:** every speedup is vs `bf16` with all levers at documented defaults.
- **Quality guard:** a quantized winner is rejected if perplexity worsens past
  the model's budget (`quality_max_ppl_delta_pct`); NeoServe falls back to the next config.
  REAL mode uses `lm_eval` when available, else a local transformers PPL probe — never silent mock.
- **Mechanism profile:** REAL mode prefers Arm Performix (`apx`); if absent it records a
  host `perf stat` sample tagged `source=perf`. REAL artifacts refuse `source=mock`.
- **Reps:** mock uses the full `concurrency.reps` (default 5). REAL probe/full grid uses
  up to **3 reps** for wall-clock; promote scripts expect a clear cost win before canonical.
- **Validity gates:** trials are invalidated on thermal throttle, high load, swap-in, or
  high cross-rep variance (CV).
- **No SME2 claims on cloud:** Graviton4/Axion/Cobalt are Neoverse V2/N2 with **no SME2**.
  Cloud wins come from **i8mm/SVE/DotProd + W4A8**, not SME2. NeoServe never cites the
  mobile "6× SME2" number for Track 2. (Judges are Arm engineers — this matters.)
- **Provenance:** `python scripts/verify_ledger.py results/canonical` re-hashes every file.

---

## Architecture

```
configs/ ──▶ runner.py ──▶ [quantize] ──▶ vLLM serve (Graviton4)
                 │                              ▲
                 ├── bench_serving (grid) ──────┘
                 ├── stats (reps/CI/gates)
                 ├── quality_guard (ppl delta)
                 ├── performix (apx top-down over SSH)
                 └── economics (tokens/$, Pareto) ──▶ artifacts (cost cards,
                                                       Docker recipe, model cards,
                                                       report.html, ledger.json)
                                                     ──▶ MCP recommend_config
                                                     ──▶ dashboard (Pareto + $/mo)
```

## MCP server
```bash
NEOSERVE_RESULTS=results/canonical PYTHONPATH=src python -m mcp.server
```
Tools: `list_models`, `recommend_config(model, tokens_per_month?)`,
`get_serving_recipe(model)`, `project_cost(model, tokens_per_month)`.

## Honest CPU-vs-GPU framing
NeoServe does **not** claim CPU beats a saturated GPU on cost at scale. CPU serving wins
when models are small/medium (≤~8B), concurrency is low/spiky, you already run CPU fleets
(co-located with app/data), GPUs are scarce/expensive, or you want **2–3× better
tokens/$ than x86 CPU**. The dashboard states this explicitly.

## References (measured Arm serving results NeoServe is grounded in)
- Optimizing vLLM on Arm CPUs (Neoverse V2 / Graviton4): https://blog.vllm.ai/2026/07/29/optimizing-vllm-on-arm-cpus.html
- AWS Graviton vLLM guide: https://aws.github.io/graviton/machinelearning/vllm.html
- KleidiAI: https://github.com/ARM-software/kleidiai · llama.cpp i8mm: https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/optimize-llama-cpp-with-arm-i8mm-instruction
- PyTorch on Graviton (BF16 fast-math, torch.compile): https://pytorch.org/blog/optimized-pytorch-w-graviton/
- Arm Performix: https://developer.arm.com/servers-and-cloud-computing/arm-performix · https://github.com/arm/performix
- Arm MCP Server: https://developer.arm.com/servers-and-cloud-computing/arm-mcp-server · https://github.com/arm/mcp
- Signal65 Graviton4 tokens/$ study: https://signal65.com (Arm Neoverse cost-efficiency lab insight)

See [`JOURNEY.md`](JOURNEY.md) for the honest log of what we validated, corrected, and rejected.
