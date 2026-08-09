"""Quality guard for quantized models.

A speed win is only a win if quality holds. Before NeoServe promotes a W8A8/W4A8
config, this module measures the perplexity delta vs the bf16 reference on a fixed
corpus and rejects the quant if it degrades beyond the model's configured budget.

REAL mode uses `lm_eval` (wikitext word-perplexity) on the Arm host. MOCK mode
returns grounded, per-precision deltas so the pipeline is testable offline.
"""
from __future__ import annotations

import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class QualityResult:
    model_id: str
    precision: str
    task: str
    ppl_base: float
    ppl_quant: float
    delta_pct: float          # (ppl_quant - ppl_base) / ppl_base * 100
    max_delta_pct: float
    passed: bool
    source: str = "mock"

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id, "precision": self.precision, "task": self.task,
            "ppl_base": self.ppl_base, "ppl_quant": self.ppl_quant,
            "delta_pct": self.delta_pct, "max_delta_pct": self.max_delta_pct,
            "passed": self.passed, "source": self.source,
        }


# Grounded typical perplexity deltas (percent worse than bf16) observed for these
# quantization schemes on 7-8B models. W8A8 is nearly lossless; W4A8 costs a little.
_MOCK_DELTA_PCT = {"bf16": 0.0, "w8a8": 0.6, "w4a8": 2.1}
# Rough bf16 wikitext word-perplexity anchors by model size band.
_MOCK_BASE_PPL = {"llama31-8b": 7.3, "qwen25-7b": 7.9, "qwen25-1p5b": 11.4}


def evaluate_quality_mock(model_id: str, model_short: str, precision: str,
                          task: str, max_delta_pct: float,
                          seed: int = 0) -> QualityResult:
    rng = random.Random(hash((model_short, precision, seed)) & 0xFFFFFFFF)
    base = _MOCK_BASE_PPL.get(model_short, 9.0)
    delta = _MOCK_DELTA_PCT.get(precision, 0.0) * rng.uniform(0.85, 1.15)
    quant = base * (1 + delta / 100.0)
    return QualityResult(
        model_id=model_id, precision=precision, task=task,
        ppl_base=round(base, 4), ppl_quant=round(quant, 4),
        delta_pct=round(delta, 3), max_delta_pct=max_delta_pct,
        passed=delta <= max_delta_pct, source="mock",
    )


def _lm_eval_ppl(model_path: str, task: str) -> float:
    """Run lm_eval word-perplexity for one model on the Arm host; return perplexity."""
    out = Path("/tmp") / f"lmeval-{abs(hash((model_path, task)))}.json"
    cmd = [
        "lm_eval", "--model", "vllm",
        "--model_args", f"pretrained={model_path},device=cpu,dtype=bfloat16",
        "--tasks", task, "--batch_size", "auto",
        "--output_path", str(out),
    ]
    subprocess.run(cmd, check=True)
    data = json.loads(out.read_text())
    res = data["results"][task]
    for key in ("word_perplexity,none", "word_perplexity", "perplexity,none", "perplexity"):
        if key in res:
            return float(res[key])
    raise KeyError(f"perplexity not found in lm_eval results for {task}: {list(res)}")


def evaluate_quality_real(model_id: str, base_model_path: str, quant_model_path: str,
                          precision: str, task: str, max_delta_pct: float) -> QualityResult:
    ppl_base = _lm_eval_ppl(base_model_path, task)
    ppl_quant = _lm_eval_ppl(quant_model_path, task)
    delta = (ppl_quant - ppl_base) / ppl_base * 100.0
    return QualityResult(
        model_id=model_id, precision=precision, task=task,
        ppl_base=round(ppl_base, 4), ppl_quant=round(ppl_quant, 4),
        delta_pct=round(delta, 3), max_delta_pct=max_delta_pct,
        passed=delta <= max_delta_pct, source="real",
    )


if __name__ == "__main__":
    for prec in ("w8a8", "w4a8"):
        r = evaluate_quality_mock("meta-llama/Llama-3.1-8B-Instruct", "llama31-8b",
                                  prec, "wikitext", 3.0)
        print(r.as_dict())
