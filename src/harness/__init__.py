"""NeoServe optimization harness.

Modules:
    config_space  -- models/sweep loading, candidate generation, env mapping
    stats         -- reps aggregation, confidence intervals, validity gates
    economics     -- tokens/$, cost/1M tokens, perf/watt proxy, Pareto frontier
    bench_serving -- vLLM benchmark_serving wrapper (+ local mock/simulation mode)
    quality_guard -- perplexity / task-accuracy delta guard for quantized models
    performix     -- Arm Performix (apx) runner over SSH + top-down parsing
    artifacts     -- cost cards, tuned Docker image, HF model cards, run report
    runner        -- end-to-end orchestration + CLI
"""

__version__ = "0.1.0"
