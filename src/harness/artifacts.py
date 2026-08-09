"""Emit the reusable artifacts that make NeoServe a deliverable, not just a report:

  * cost card       -- JSON + Markdown: cost/1M tokens, tokens/$, quality delta,
                       Performix top-down, savings vs baseline (per model).
  * model card      -- Hugging Face README for the published quantized model.
  * serving recipe  -- tuned Dockerfile.arm64 + compose + run.sh embedding the exact
                       winning env/flags, so anyone can reproduce the fast server.
  * HTML report     -- self-contained run report for judges.
  * provenance      -- SHA-256 ledger over every emitted file (numbers are traceable).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from jinja2 import Template

from .config_space import Candidate, InstanceSpec, ModelSpec, launch_env, vllm_serve_args
from .economics import ServingPoint


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Cost card
# --------------------------------------------------------------------------- #
def build_cost_card(model: ModelSpec, best: ServingPoint, baseline: ServingPoint,
                    quality: Optional[dict], performix: Optional[dict],
                    savings: dict, mock: bool) -> dict:
    return {
        "schema": "neoserve.costcard/v1",
        "generated_at": _now(),
        "mock": mock,
        "model": {"id": model.id, "short": model.short, "params_b": model.params_b},
        "instance": best.instance,
        "usd_per_hr": best.usd_per_hr,
        "winning_config": best.label,
        "baseline_config": baseline.label,
        "operating_point": {
            "request_rate": best.request_rate,
            "output_throughput_tok_s": round(best.output_throughput_tok_s, 2),
            "ttft_p95_ms": round(best.ttft_p95_ms, 1),
            "tpot_p95_ms": round(best.tpot_p95_ms, 1),
        },
        "economics": {
            "cost_per_1m_tokens_usd": round(best.cost_per_1m(), 4),
            "baseline_cost_per_1m_tokens_usd": round(baseline.cost_per_1m(), 4),
            "tokens_per_usd": round(best.tokens_per_usd(), 0),
            "perf_per_watt_proxy_tok_s_per_w": round(best.perf_watt(), 3),
            "throughput_speedup_x": round(savings.get("throughput_speedup_x", 0), 2),
        },
        "quality_guard": quality,
        "performix_topdown": performix,
        "savings_example": savings,
    }


_COST_CARD_MD = Template("""# NeoServe Cost Card - {{ c.model.short }} ({{ c.winning_config }})

{% if c.mock %}> NOTE: MOCK numbers (grounded simulator). Replace with real Graviton4 runs before submission.{% endif %}

- **Model:** `{{ c.model.id }}` ({{ c.model.params_b }}B)
- **Instance:** {{ c.instance }} @ ${{ c.usd_per_hr }}/hr
- **Winning config:** `{{ c.winning_config }}`
- **Baseline:** `{{ c.baseline_config }}`

## Operating point (meets SLO)
| Metric | Value |
|---|---|
| Request rate | {{ c.operating_point.request_rate }} req/s |
| Output throughput | {{ c.operating_point.output_throughput_tok_s }} tok/s |
| TTFT p95 | {{ c.operating_point.ttft_p95_ms }} ms |
| TPOT p95 | {{ c.operating_point.tpot_p95_ms }} ms |

## Economics
| Metric | Winning | Baseline |
|---|---|---|
| Cost / 1M output tokens | ${{ c.economics.cost_per_1m_tokens_usd }} | ${{ c.economics.baseline_cost_per_1m_tokens_usd }} |
| Tokens / $ | {{ c.economics.tokens_per_usd }} | - |
| Throughput speedup | {{ c.economics.throughput_speedup_x }}x | 1.0x |
| Perf/watt proxy (tok/s/W)* | {{ c.economics.perf_per_watt_proxy_tok_s_per_w }} | - |

*Relative-only proxy; Graviton does not expose RAPL.

{% if c.quality_guard %}## Quality guard
Perplexity {{ c.quality_guard.ppl_base }} -> {{ c.quality_guard.ppl_quant }} ({{ c.quality_guard.delta_pct }}% delta, budget {{ c.quality_guard.max_delta_pct }}%): **{{ 'PASS' if c.quality_guard.passed else 'FAIL' }}**{% endif %}

{% if c.performix_topdown %}## Arm Performix top-down (winning config)
retiring {{ c.performix_topdown.topdown.retiring }}% | backend {{ c.performix_topdown.topdown.backend_bound }}% (mem {{ c.performix_topdown.topdown.memory_bound }}%) | IPC {{ c.performix_topdown.topdown.ipc }}{% endif %}

## Example savings
At {{ '{:,.0f}'.format(c.savings_example.tokens_per_month_example|default(0)) }} output tokens/month:
serving on the tuned config costs **${{ '%.0f'|format(c.savings_example.best_usd_per_month) }}/mo** vs **${{ '%.0f'|format(c.savings_example.baseline_usd_per_month) }}/mo** baseline = **${{ '%.0f'|format(c.savings_example.usd_saved_per_month) }}/mo saved ({{ '%.0f'|format(c.savings_example.pct_saved) }}%)**.
""")


def write_cost_card(run_dir: Path, model: ModelSpec, card: dict) -> tuple[Path, Path]:
    d = run_dir / "cost_cards"
    d.mkdir(parents=True, exist_ok=True)
    jp = d / f"{model.short}.json"
    jp.write_text(json.dumps(card, indent=2), encoding="utf-8")
    mp = d / f"{model.short}.md"
    mp.write_text(_COST_CARD_MD.render(c=card), encoding="utf-8")
    return jp, mp


# --------------------------------------------------------------------------- #
# HF model card
# --------------------------------------------------------------------------- #
_MODEL_CARD = Template("""---
license: apache-2.0
base_model: {{ model.id }}
tags: [arm, aarch64, graviton, neoverse, vllm, quantization, {{ precision }}, neoserve]
---

# {{ model.short }}-{{ precision }} (NeoServe, Arm Neoverse optimized)

{{ precision|upper }} quantization of `{{ model.id }}` produced by
[NeoServe](https://github.com/AmanM006/neoserve) for cost-efficient CPU serving on
AWS Graviton4 (Neoverse V2) with vLLM (oneDNN + Arm Compute Library{% if precision == 'w4a8' %} + KleidiAI INT4{% endif %}).

## Why
On Arm cloud CPUs, {{ precision|upper }} matmul runs through i8mm/SMMLA{% if precision == 'w4a8' %}/KleidiAI INT4{% endif %} kernels,
delivering higher serving throughput per dollar than bf16 while holding quality.

## Measured (see cost card)
- Cost / 1M output tokens: **${{ card.economics.cost_per_1m_tokens_usd }}** (baseline bf16 ${{ card.economics.baseline_cost_per_1m_tokens_usd }})
- Throughput speedup vs bf16: **{{ card.economics.throughput_speedup_x }}x**
{% if card.quality_guard %}- Perplexity delta vs bf16: **{{ card.quality_guard.delta_pct }}%** (budget {{ card.quality_guard.max_delta_pct }}%){% endif %}
- Instance: {{ card.instance }} @ ${{ card.usd_per_hr }}/hr

## Serve
```bash
docker compose -f neoserve-recipe/compose.yaml up
# OpenAI-compatible endpoint on :8000
```
{% if card.mock %}
> Numbers above are from NeoServe's grounded simulator; regenerate on real Graviton4 before publishing.
{% endif %}
""")


def write_model_card(run_dir: Path, model: ModelSpec, precision: str, card: dict) -> Path:
    d = run_dir / "model_cards"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{model.short}-{precision}.md"
    p.write_text(_MODEL_CARD.render(model=model, precision=precision, card=card), encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# Tuned serving recipe (Dockerfile + compose + run.sh)
# --------------------------------------------------------------------------- #
def write_serving_recipe(run_dir: Path, cand: Candidate, model: ModelSpec,
                         instance: InstanceSpec) -> Path:
    d = run_dir / "serving_recipe" / model.short
    d.mkdir(parents=True, exist_ok=True)
    env = launch_env(cand, instance)
    serve_args = vllm_serve_args(cand, model)

    env_lines = "\n".join(f'ENV {k}="{v}"' for k, v in env.items())
    compose_env = "\n".join(f"      {k}: \"{v}\"" for k, v in env.items())
    cmd_str = " ".join(serve_args + ["--host", "0.0.0.0", "--port", "8000"])

    (d / "Dockerfile.arm64").write_text(f"""# NeoServe tuned serving image for {model.short} ({cand.precision})
# Winning config: {cand.label()}  |  target: {instance.name}
FROM --platform=linux/arm64 vllm/vllm-openai:latest
{env_lines}
EXPOSE 8000
ENTRYPOINT ["/bin/bash","-lc"]
CMD ["{cmd_str}"]
""", encoding="utf-8")

    (d / "compose.yaml").write_text(f"""services:
  neoserve:
    build:
      context: .
      dockerfile: Dockerfile.arm64
    platform: linux/arm64
    ports: ["8000:8000"]
    environment:
{compose_env}
    # Recommended instance: {instance.name} ({instance.vcpu} vCPU / {instance.mem_gb} GB)
""", encoding="utf-8")

    (d / "run.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n"
                              + "".join(f'export {k}="{v}"\n' for k, v in env.items())
                              + cmd_str + "\n", encoding="utf-8")
    return d


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #
_REPORT_HTML = Template("""<!doctype html><html><head><meta charset="utf-8">
<title>NeoServe run {{ s.run_id }}</title>
<style>
 body{font-family:system-ui,Segoe UI,Roboto,sans-serif;max-width:960px;margin:2rem auto;color:#111;line-height:1.5}
 h1{margin-bottom:.2rem} .sub{color:#666} table{border-collapse:collapse;width:100%;margin:1rem 0}
 th,td{border:1px solid #ddd;padding:.4rem .6rem;text-align:right} th:first-child,td:first-child{text-align:left}
 .mock{background:#fff3cd;border:1px solid #ffe69c;padding:.6rem;border-radius:6px}
 .win{background:#d1e7dd} code{background:#f1f1f1;padding:.1rem .3rem;border-radius:3px}
</style></head><body>
<h1>NeoServe - Arm serving optimization report</h1>
<div class="sub">run {{ s.run_id }} | {{ s.generated_at }} | mode: {{ 'MOCK (simulator)' if s.mock else 'REAL (Graviton4)' }}</div>
{% if s.mock %}<p class="mock"><b>Mock run.</b> Numbers come from NeoServe's grounded simulator so the pipeline is fully runnable offline. Re-run on real Graviton4 (<code>--real</code>) before submission.</p>{% endif %}
<p>SLO: TTFT p95 &le; {{ s.slo.ttft_p95_ms }} ms, TPOT p95 &le; {{ s.slo.tpot_p95_ms }} ms.</p>
{% for m in s.models %}
<h2>{{ m.model }} on {{ m.instance }}</h2>
<table>
<tr><th>Config</th><th>op req/s</th><th>throughput tok/s</th><th>TTFT p95 ms</th><th>TPOT p95 ms</th><th>$/1M tok</th><th>speedup</th></tr>
<tr><td>{{ m.baseline_label }} (baseline)</td><td>{{ m.baseline.request_rate }}</td><td>{{ '%.0f'|format(m.baseline.output_throughput_tok_s) }}</td><td>{{ '%.0f'|format(m.baseline.ttft_p95_ms) }}</td><td>{{ '%.0f'|format(m.baseline.tpot_p95_ms) }}</td><td>${{ '%.3f'|format(m.baseline.cost_per_1m) }}</td><td>1.0x</td></tr>
<tr class="win"><td>{{ m.best_label }} (winner)</td><td>{{ m.best.request_rate }}</td><td>{{ '%.0f'|format(m.best.output_throughput_tok_s) }}</td><td>{{ '%.0f'|format(m.best.ttft_p95_ms) }}</td><td>{{ '%.0f'|format(m.best.tpot_p95_ms) }}</td><td>${{ '%.3f'|format(m.best.cost_per_1m) }}</td><td>{{ '%.2f'|format(m.speedup) }}x</td></tr>
</table>
{% if m.quality %}<p>Quality guard: perplexity {{ m.quality.ppl_base }} -> {{ m.quality.ppl_quant }} ({{ m.quality.delta_pct }}%), {{ 'PASS' if m.quality.passed else 'FAIL' }}.</p>{% endif %}
{% if m.performix_best %}<p>Performix (winner): retiring {{ m.performix_best.topdown.retiring }}%, backend {{ m.performix_best.topdown.backend_bound }}% (mem {{ m.performix_best.topdown.memory_bound }}%), IPC {{ m.performix_best.topdown.ipc }} vs baseline retiring {{ m.performix_base.topdown.retiring }}%, IPC {{ m.performix_base.topdown.ipc }}.</p>{% endif %}
<p>Savings at {{ '{:,.0f}'.format(m.savings.tokens_per_month_example) }} tok/mo: <b>${{ '%.0f'|format(m.savings.usd_saved_per_month) }}/mo ({{ '%.0f'|format(m.savings.pct_saved) }}%)</b>.</p>
{% endfor %}
<hr><p class="sub">Generated by NeoServe. Provenance: see <code>ledger.json</code>.</p>
</body></html>""")


def write_report_html(run_dir: Path, summary: dict) -> Path:
    p = run_dir / "report.html"
    p.write_text(_REPORT_HTML.render(s=summary), encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# Provenance ledger (SHA-256 over every emitted file)
# --------------------------------------------------------------------------- #
def write_ledger(run_dir: Path) -> Path:
    entries = []
    for f in sorted(run_dir.rglob("*")):
        if f.is_file() and f.name != "ledger.json":
            h = hashlib.sha256(f.read_bytes()).hexdigest()
            entries.append({"path": str(f.relative_to(run_dir)).replace("\\", "/"),
                            "sha256": h, "bytes": f.stat().st_size})
    ledger = {"schema": "neoserve.ledger/v1", "generated_at": _now(),
              "run_dir": run_dir.name, "files": entries}
    p = run_dir / "ledger.json"
    p.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    return p
