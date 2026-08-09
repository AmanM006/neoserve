# NeoServe Cost Card - qwen25-1p5b (w4a8 alloc=mimalloc bind=phys kv=16g mnbt=2048 fpmath=bf16 lse)

> NOTE: MOCK numbers (grounded simulator). Replace with real Graviton4 runs before submission.

- **Model:** `Qwen/Qwen2.5-1.5B-Instruct` (1.5B)
- **Instance:** c8g.4xlarge @ $0.63712/hr
- **Winning config:** `w4a8 alloc=mimalloc bind=phys kv=16g mnbt=2048 fpmath=bf16 lse`
- **Baseline:** `bf16 alloc=glibc bind=auto kv=16g mnbt=4096`

## Operating point (meets SLO)
| Metric | Value |
|---|---|
| Request rate | 4.0 req/s |
| Output throughput | 530.0 tok/s |
| TTFT p95 | 289.1 ms |
| TPOT p95 | 23.3 ms |

## Economics
| Metric | Winning | Baseline |
|---|---|---|
| Cost / 1M output tokens | $0.3339 | $0.3485 |
| Tokens / $ | 2994714.0 | - |
| Throughput speedup | 1.04x | 1.0x |
| Perf/watt proxy (tok/s/W)* | 3.786 | - |

*Relative-only proxy; Graviton does not expose RAPL.

## Quality guard
Perplexity 11.4 -> 11.6049 (1.798% delta, budget 4.0%): **PASS**

## Arm Performix top-down (winning config)
retiring 51.7% | backend 35.4% (mem 19.4%) | IPC 1.71

## Example savings
At 5,000,000,000 output tokens/month:
serving on the tuned config costs **$1670/mo** vs **$1742/mo** baseline = **$73/mo saved (4%)**.