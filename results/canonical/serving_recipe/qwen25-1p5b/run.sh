#!/usr/bin/env bash
set -euo pipefail
export LD_PRELOAD="/usr/local/lib/libmimalloc.so"
export ONEDNN_DEFAULT_FPMATH_MODE="BF16"
export VLLM_CPU_OMP_THREADS_BIND="0-15"
export VLLM_CPU_KVCACHE_SPACE="16"
export NEOSERVE_LSE_ATOMICS="on"
vllm serve neoserve/qwen25-1p5b-w4a8 --device cpu --dtype bfloat16 --max-model-len 8192 --max-num-batched-tokens 2048 --disable-log-requests --host 0.0.0.0 --port 8000
