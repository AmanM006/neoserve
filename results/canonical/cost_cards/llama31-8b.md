# NeoServe Cost Card - llama31-8b (w4a8 alloc=mimalloc bind=auto kv=16g mnbt=8192 fpmath=bf16)

> NOTE: MOCK numbers (grounded simulator). Replace with real Graviton4 runs before submission.

- **Model:** `meta-llama/Llama-3.1-8B-Instruct` (8.0B)
- **Instance:** c8g.4xlarge @ $0.63712/hr
- **Winning config:** `w4a8 alloc=mimalloc bind=auto kv=16g mnbt=8192 fpmath=bf16`
- **Baseline:** `bf16 alloc=glibc bind=auto kv=16g mnbt=4096`

## Operating point (meets SLO)
| Metric | Value |
|---|---|
| Request rate | 2.0 req/s |
| Output throughput | 261.37 tok/s |
| TTFT p95 | 2672.2 ms |
| TPOT p95 | 141.4 ms |

## Economics
| Metric | Winning | Baseline |
|---|---|---|
| Cost / 1M output tokens | $0.6771 | $5.4523 |
| Tokens / $ | 1476854.0 | - |
| Throughput speedup | 8.05x | 1.0x |
| Perf/watt proxy (tok/s/W)* | 1.867 | - |

*Relative-only proxy; Graviton does not expose RAPL.

## Quality guard
Perplexity 7.3 -> 7.4319 (1.807% delta, budget 3.0%): **PASS**

## Arm Performix top-down (winning config)
retiring 52.4% | backend 34.7% (mem 19.1%) | IPC 1.73

## Example savings
At 5,000,000,000 output tokens/month:
serving on the tuned config costs **$3386/mo** vs **$27262/mo** baseline = **$23876/mo saved (88%)**.