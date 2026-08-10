---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags: [arm, aarch64, graviton, neoverse, vllm, quantization, w4a8, neoserve]
---

# qwen25-1p5b-w4a8 (NeoServe, Arm Neoverse optimized)

W4A8 quantization of `Qwen/Qwen2.5-1.5B-Instruct` produced by
[NeoServe](https://github.com/AmanM006/neoserve) for cost-efficient CPU serving on
AWS Graviton4 (Neoverse V2) with vLLM (oneDNN + Arm Compute Library + KleidiAI INT4).

## Why
On Arm cloud CPUs, W4A8 matmul runs through i8mm/SMMLA/KleidiAI INT4 kernels,
delivering higher serving throughput per dollar than bf16 while holding quality.

## Measured (see cost card)
- Cost / 1M output tokens: **$0.7451** (baseline bf16 $1.4461)
- Throughput speedup vs bf16: **1.94x**
- Perplexity delta vs bf16: **2.372%** (budget 4.0%)
- Instance: c8g.4xlarge @ $0.63712/hr

## Serve
```bash
docker compose -f neoserve-recipe/compose.yaml up
# OpenAI-compatible endpoint on :8000
```
