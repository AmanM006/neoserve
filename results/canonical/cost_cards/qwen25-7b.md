# NeoServe Cost Card - qwen25-7b (w4a8 alloc=glibc bind=phys kv=16g mnbt=8192 fpmath=bf16 lse)

> NOTE: MOCK numbers (grounded simulator). Replace with real Graviton4 runs before submission.

- **Model:** `Qwen/Qwen2.5-7B-Instruct` (7.6B)
- **Instance:** c8g.4xlarge @ $0.63712/hr
- **Winning config:** `w4a8 alloc=glibc bind=phys kv=16g mnbt=8192 fpmath=bf16 lse`
- **Baseline:** `bf16 alloc=glibc bind=auto kv=16g mnbt=4096`

## Operating point (meets SLO)
| Metric | Value |
|---|---|
| Request rate | 2.0 req/s |
| Output throughput | 258.19 tok/s |
| TTFT p95 | 2211.1 ms |
| TPOT p95 | 121.5 ms |

## Economics
| Metric | Winning | Baseline |
|---|---|---|
| Cost / 1M output tokens | $0.6855 | $5.5834 |
| Tokens / $ | 1458857.0 | - |
| Throughput speedup | 8.15x | 1.0x |
| Perf/watt proxy (tok/s/W)* | 1.844 | - |

*Relative-only proxy; Graviton does not expose RAPL.

## Quality guard
Perplexity 7.9 -> 8.0806 (2.287% delta, budget 3.0%): **PASS**

## Arm Performix top-down (winning config)
retiring 52.8% | backend 34.1% (mem 18.8%) | IPC 1.7

## Example savings
At 5,000,000,000 output tokens/month:
serving on the tuned config costs **$3427/mo** vs **$27917/mo** baseline = **$24489/mo saved (88%)**.