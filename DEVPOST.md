# NeoServe — Devpost draft (Arm Create: AI Optimization Challenge 2026)

## Elevator pitch
NeoServe finds the **cheapest SLO-meeting LLM serving config on AWS Graviton4** — not the fastest single-stream llama.cpp knob. It sweeps vLLM under real concurrency, scores **$/1M tokens at p95 TTFT/TPOT**, quality-guards quants, attaches a PMU/`perf` mechanism profile, and ships reusable cost cards + Docker recipes + MCP `recommend_config`.

## The problem everyone else skips
Track 2 is flooded with autotuners that maximize single-stream tokens/sec. Production serving asks a different question: *at my traffic and my latency SLO, what config is cheapest on Arm?*

## What we built
- Successive-halving search over allocator / thread bind / KV / batch / precision (bf16 → W4A8)
- Concurrency grid with reps, CIs, validity gates
- Quality guard (lm_eval or local PPL) before promoting a quant
- Mechanism profile: Arm Performix when present, else honest `perf stat` (never silent mock on REAL)
- Artifacts: cost card, HF model card, tuned Dockerfile/compose, SHA-256 ledger, Next.js dashboard, MCP tool

## Billboard (fill after promote)
> **$___ / 1M tokens** at p95 SLO on **c8g.4xlarge** — **___×** vs bf16 baseline — commit `_______` — `mock: false`

## Demo
1. Offline: `PYTHONPATH=src python -m harness.runner --mock` + dashboard
2. Real cost card from `results/canonical/`
3. `docker compose -f results/canonical/serving_recipe/<model>/compose.yaml up` → `/health`
4. MCP: `recommend_config(model=...)` returns env + projected $/mo

## Reproduce
```bash
python scripts/verify_ledger.py results/canonical
# serve winner
cd results/canonical/serving_recipe/<model> && docker compose up
```

## Built with
Python, vLLM (CPU/Arm), llmcompressor, oneDNN/ACL/KleidiAI path, AWS Graviton4 (Neoverse V2), Next.js, MCP

## What's next
W8A8 + 7B/8B narrow sweeps; full Performix `apx` when licensed on the AMI; public demo URL.
