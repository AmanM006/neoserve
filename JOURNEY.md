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

## Known limitations / to validate on real Graviton4 before submission

- The offline `--mock` numbers come from a **grounded simulator** (per-lever multipliers
  taken from published Neoverse V2 measurements). They exist so the pipeline, report, and
  dashboard run without spending on AWS. **All headline numbers must be regenerated with
  `--real` on a `c8g` instance before the Devpost submission.** Every mock artifact is
  tagged `MOCK`.
- **Perf-per-watt is a relative proxy only** — Graviton does not expose RAPL, so we use a
  socket-share TDP estimate and only compare configs on the *same* instance. Never
  published as an absolute wattage.
- Real-mode server lifecycle assumes vLLM aarch64 wheels install cleanly; fallback path is
  `llama.cpp` continuous-batching serving if a specific vLLM CPU feature blocks a model.

## Log
- 2026-08-09 — Harness, economics, Performix integration, artifacts, MCP, dashboard, and
  deploy tooling implemented and validated end-to-end in mock mode. Next: real Graviton4
  runs for Llama-3.1-8B and Qwen2.5-7B, then swap canonical results and re-record the demo.
