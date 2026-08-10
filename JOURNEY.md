# JOURNEY — what we validated, corrected, and rejected

The strongest entries in this challenge win on *credibility*: they show their working,
including the things that didn't pan out. This is NeoServe's running honesty log.

## Corrections we made to conventional hackathon "wisdom"

1. **"KleidiAI gives ~6× on the cloud."** False for Track 2. The 6× figure is a
   **mobile** result (Armv9 **SME2** on phones / Arm C1). Current cloud Arm CPUs —
   **Graviton4, GCP Axion, Azure Cobalt 100** — are Neoverse V2/N2 and have **no SME2**.
   Cloud INT4/INT8 wins come from **i8mm/SMMLA, SVE, DotProd + W4A8**, not SME2. NeoServe
   never cites the mobile number for cloud results.

2. **"KleidiAI's big win is Q4_0."** On Arm CPUs the mainline path already accelerates
   Q4_0; KleidiAI's clearer serving win is on **Q8_0 / W8A8** (SMMLA), with an additional
   step from **W4A8** INT4 micro-kernels. We model W8A8 as the near-lossless default win
   and W4A8 as the cheapest-but-quality-guarded option.

3. **"More threads = faster."** On Graviton (1 core/vCPU, no SMT) the win is **pinning
   OMP threads to physical cores** and fixing allocator/atomics contention, not simply
   raising thread count. Modeled via the `thread_bind`, `allocator`, and `lse_atomics`
   levers.

4. **"Single-stream tokens/sec is the metric."** It isn't for serving. We optimize
   **cost/1M tokens at a p95 SLO under concurrency** (goodput), which is why the tuned
   config's advantage shows up as *sustainable SLO-meeting throughput*, not just a faster
   single stream.

## Design decisions

- **vLLM over llama.cpp** for the serving layer: continuous batching + the documented
  Neoverse V2 optimization stack (mimalloc, LSE, oneDNN prepacking, W4A8) is where the
  concurrency economics live — and it's the whitespace the rest of the field left open.
- **Quality guard is mandatory**, not optional: a W4A8 config that blows the perplexity
  budget is auto-rejected and NeoServe falls back to the next cheapest passing config.
- **Provenance ledger** (SHA-256 over every emitted file) so a judge can trust that the
  numbers in the report came from the committed raw data.

## Real Graviton4 scars (2026-08-10)

These are the failures that would have killed a demo if we papered over them:

1. **Pip vLLM assumed GPU** → `Failed to infer device type`. Fixed by building vLLM from
   source with `VLLM_TARGET_DEVICE=cpu` (`deploy/build_vllm_cpu.sh`).
2. **GPU `torchaudio` wheel** pulled `libcudart.so.13` on a CPU box and crashed import.
   Uninstalled torchaudio/torchvision for the CPU venv.
3. **`--disable-log-requests`** is gone on current vLLM CLI → exit code 2. Removed.
4. **`--device cpu` is not a backend flag** on current vLLM — it is parsed as a device
   *id* (`ValueError: Non-integer device ID 'cpu'`). CPU is selected via
   `VLLM_TARGET_DEVICE=cpu` only.
5. **Loadavg validity gate at 0.5** invalidated every cell after successive server
   restarts (residual 1-min load). Raised to 1.5 and added a cooldown between candidates;
   scoring also falls back to SLO-meeting points if all soft-gated.
6. **First bf16-only real run** (`real-20260810-054859`) produced **winner = baseline,
   0% savings**. That is an honest non-result — do not promote as a win. The follow-up
   sweep adds W4A8 + allocators.
7. **GPTQ / llmcompressor** needed: local calib (HF `wikitext` id broken with current hub),
   drop deprecated `sequential_update`, and **`ignore: [lm_head]`** so vLLM CPU can load
   the artifact (`lm_head.weight_scale` otherwise fatal).
8. **Mock Performix in REAL mode** was a credibility bug. REAL now prefers `apx`, else
   `perf stat` tagged `source=perf`, and refuses to ship `source=mock` Performix on REAL.

## Known limitations

- Perf-per-watt is a **relative TDP-share proxy** only (no RAPL on Graviton).
- REAL full-grid uses ≤3 reps for wall-clock; treat winner-confirm as a promote step.
- LSE atomics are recorded as provenance today; swapping libgomp builds is still deploy
  tooling, not a one-line env flip on every AMI.
- Cloud AI credibility still wants at least one **7B/8B** result alongside 1.5B.

## Log
- 2026-08-09 — Harness, economics, Performix integration, artifacts, MCP, dashboard, and
  deploy tooling implemented and validated end-to-end in mock mode.
- 2026-08-10 — First real c8g.4xlarge run; CPU vLLM build; W4A8 quant that loads; live
  bf16+W4A8 re-sweep; wired REAL quality/perf paths; ledger verify + promote scripts.
