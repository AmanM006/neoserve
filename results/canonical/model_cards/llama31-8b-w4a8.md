---
license: apache-2.0
base_model: meta-llama/Llama-3.1-8B-Instruct
tags: [arm, aarch64, graviton, neoverse, vllm, quantization, w4a8, neoserve]
---

# llama31-8b-w4a8 (NeoServe, Arm Neoverse optimized)

W4A8 quantization of `meta-llama/Llama-3.1-8B-Instruct` produced by
[NeoServe](https://github.com/AmanM006/neoserve) for cost-efficient CPU serving on
AWS Graviton4 (Neoverse V2) with vLLM (oneDNN + Arm Compute Library + KleidiAI INT4).

## Why
On Arm cloud CPUs, W4A8 matmul runs through i8mm/SMMLA/KleidiAI INT4 kernels,
delivering higher serving throughput per dollar than bf16 while holding quality.

## Measured (see cost card)
- Cost / 1M output tokens: **$0.6771** (baseline bf16 $5.4523)
- Throughput speedup vs bf16: **8.05x**
- Perplexity delta vs bf16: **1.807%** (budget 3.0%)
- Instance: c8g.4xlarge @ $0.63712/hr

## Serve
```bash
docker compose -f neoserve-recipe/compose.yaml up
# OpenAI-compatible endpoint on :8000
```

> Numbers above are from NeoServe's grounded simulator; regenerate on real Graviton4 before publishing.
