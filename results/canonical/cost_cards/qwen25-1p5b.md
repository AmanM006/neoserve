# NeoServe Cost Card - qwen25-1p5b (w4a8 alloc=mimalloc bind=phys kv=16g mnbt=2048)



- **Model:** `Qwen/Qwen2.5-1.5B-Instruct` (1.5B)
- **Instance:** c8g.4xlarge @ $0.63712/hr
- **Winning config:** `w4a8 alloc=mimalloc bind=phys kv=16g mnbt=2048`
- **Baseline:** `bf16 alloc=glibc bind=auto kv=16g mnbt=4096`

## Operating point (meets SLO)
| Metric | Value |
|---|---|
| Request rate | 2.0 req/s |
| Output throughput | 237.52 tok/s |
| TTFT p95 | 421.6 ms |
| TPOT p95 | 108.8 ms |

## Economics
| Metric | Winning | Baseline |
|---|---|---|
| Cost / 1M output tokens | $0.7451 | $1.4461 |
| Tokens / $ | 1342077.0 | - |
| Throughput speedup | 1.94x | 1.0x |
| Perf/watt proxy (tok/s/W)* | 1.697 | - |

*Relative-only proxy; Graviton does not expose RAPL.

## Quality guard
Perplexity 11.4 -> 11.6704 (2.372% delta, budget 4.0%): **PASS**

## Arm Performix top-down (winning config)
retiring 51.8% | backend 35.2% (mem 19.3%) | IPC 1.68

## Example savings
At 5,000,000,000 output tokens/month:
serving on the tuned config costs **$3726/mo** vs **$7230/mo** baseline = **$3505/mo saved (48%)**.