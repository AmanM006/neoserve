"""Arm Performix (apx) integration.

The challenge rules explicitly invite using Arm Performix to produce exact,
PMU-backed benchmarks -- almost no competitor does this. NeoServe profiles the
bf16 baseline and the winning config with Performix's top-down methodology so every
speedup ships with a mechanistic "why" (e.g. baseline is backend/memory bound; the
tuned config shifts work into retiring via i8mm/SMMLA + oneDNN prepacking).

REAL mode drives the `apx` CLI against a remote Arm64 target over SSH (the same
transport Performix and the Arm MCP Server use). MOCK mode returns a coherent
top-down breakdown so the report/dashboard render offline.

Top-down level-1 categories (Arm Neoverse): retiring, bad_speculation,
frontend_bound, backend_bound (the last split into memory_bound / core_bound).
"""
from __future__ import annotations

import json
import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TopDown:
    retiring: float
    bad_speculation: float
    frontend_bound: float
    backend_bound: float
    memory_bound: float
    core_bound: float
    ipc: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class PerformixReport:
    candidate_id: str
    label: str
    topdown: TopDown
    hotspots: list[dict] = field(default_factory=list)   # [{symbol, pct, note}]
    source: str = "mock"

    def as_dict(self) -> dict:
        return {"candidate_id": self.candidate_id, "label": self.label,
                "topdown": self.topdown.as_dict(), "hotspots": self.hotspots,
                "source": self.source}


def _normalize(td: dict) -> dict:
    total = td["retiring"] + td["bad_speculation"] + td["frontend_bound"] + td["backend_bound"]
    if total <= 0:
        return td
    for k in ("retiring", "bad_speculation", "frontend_bound", "backend_bound"):
        td[k] = td[k] / total * 100.0
    return td


def profile_mock(candidate_id: str, label: str, precision: str,
                 tuned: bool, seed: int = 0) -> PerformixReport:
    """Coherent top-down: bf16 baseline is memory/backend bound; INT8/INT4 tuned
    configs move cycles into retiring and lift IPC (SMMLA/i8mm + oneDNN prepack)."""
    rng = random.Random(hash((candidate_id, seed)) & 0xFFFFFFFF)
    if not tuned:
        td = {"retiring": 28, "bad_speculation": 6, "frontend_bound": 10,
              "backend_bound": 56}
        mem_share, ipc = 0.72, 0.9
        hot = [
            {"symbol": "ggml_vec_dot_f16 / bf16 GEMM", "pct": 41.0,
             "note": "fp32/bf16 matmul, memory-bandwidth bound"},
            {"symbol": "libgomp barrier wait", "pct": 14.0,
             "note": "OMP dynamic-schedule contention (no LSE atomics)"},
            {"symbol": "malloc/page-fault", "pct": 9.0,
             "note": "glibc allocator page faults under concurrency"},
        ]
    else:
        td = {"retiring": 52, "bad_speculation": 4, "frontend_bound": 9,
              "backend_bound": 35}
        mem_share, ipc = 0.55, 1.7
        kernel = "kai_matmul_qai8 (KleidiAI INT4)" if precision == "w4a8" else \
                 "onednn_int8_smmla_gemm"
        hot = [
            {"symbol": kernel, "pct": 47.0,
             "note": "INT8/INT4 SMMLA matmul; weights prepacked at warmup"},
            {"symbol": "paged_attention_bfmmla", "pct": 12.0,
             "note": "attention via BFMMLA + poly softmax"},
            {"symbol": "mimalloc fast-path", "pct": 3.0,
             "note": "allocator no longer a hotspot"},
        ]
    td = _normalize({k: v * rng.uniform(0.97, 1.03) for k, v in td.items()})
    backend = td["backend_bound"]
    top = TopDown(
        retiring=round(td["retiring"], 1), bad_speculation=round(td["bad_speculation"], 1),
        frontend_bound=round(td["frontend_bound"], 1), backend_bound=round(backend, 1),
        memory_bound=round(backend * mem_share, 1), core_bound=round(backend * (1 - mem_share), 1),
        ipc=round(ipc * rng.uniform(0.98, 1.02), 2),
    )
    return PerformixReport(candidate_id, label, top, hot, source="mock")


# --------------------------------------------------------------------------- #
# REAL path: apx over SSH
# --------------------------------------------------------------------------- #
def build_apx_cmd(recipe: str, target: str, workload: str,
                  out_dir: str = "/tmp/neoserve-apx") -> list[str]:
    """Compose an `apx recipe run` invocation. `target` is an SSH target Performix
    connects to (user@host or a configured target name)."""
    return [
        "apx", "recipe", "run", recipe,
        "--target", target,
        "--workload", workload,
        "--output", out_dir,
        "--format", "json",
    ]


def profile_real(candidate_id: str, label: str, precision: str, target: str,
                 workload: str, recipe: str = "topdown",
                 result_dir: Optional[Path] = None) -> PerformixReport:
    """Run Performix on the Arm target and parse its JSON top-down output."""
    out_dir = str((result_dir or Path("/tmp/neoserve-apx")) / candidate_id)
    subprocess.run(build_apx_cmd(recipe, target, workload, out_dir), check=True)
    data = json.loads((Path(out_dir) / "topdown.json").read_text())
    td = data["topdown"]
    backend = float(td["backend_bound"])
    top = TopDown(
        retiring=float(td["retiring"]), bad_speculation=float(td["bad_speculation"]),
        frontend_bound=float(td["frontend_bound"]), backend_bound=backend,
        memory_bound=float(td.get("memory_bound", backend * 0.6)),
        core_bound=float(td.get("core_bound", backend * 0.4)),
        ipc=float(data.get("ipc", 0.0)),
    )
    hotspots = data.get("hotspots", [])[:6]
    return PerformixReport(candidate_id, label, top, hotspots, source="real")


if __name__ == "__main__":
    b = profile_mock("base", "bf16 default", "bf16", tuned=False)
    t = profile_mock("best", "w4a8 tuned", "w4a8", tuned=True)
    for r in (b, t):
        print(r.label, "-> retiring", r.topdown.retiring, "%, backend",
              r.topdown.backend_bound, "%, ipc", r.topdown.ipc)
