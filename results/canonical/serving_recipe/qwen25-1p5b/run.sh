#!/usr/bin/env bash
set -euo pipefail
export VLLM_TARGET_DEVICE="cpu"
export LD_PRELOAD="/usr/local/lib/libmimalloc.so"
export VLLM_CPU_OMP_THREADS_BIND="0-15"
export VLLM_CPU_KVCACHE_SPACE="16"
export NEOSERVE_LSE_ATOMICS="off"
vllm serve models/qwen25-1p5b-w4a8 --dtype bfloat16 --max-model-len 4096 --max-num-batched-tokens 2048 --host 0.0.0.0 --port 8000
