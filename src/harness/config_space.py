"""Optimization space: load models/sweep configs, generate candidate configs, and
map each candidate to the environment/flags used to launch vLLM on Graviton4.

A "candidate" is one (model, precision, lever-combo) point. The runner benchmarks
candidates under a concurrency grid and scores them on cost-at-SLO.
"""
from __future__ import annotations

import hashlib
import itertools
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import yaml


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #
@dataclass
class ModelSpec:
    id: str
    short: str
    params_b: float
    ctx: int
    precisions: list[str]
    quality_task: str = "wikitext"
    quality_max_ppl_delta_pct: float = 3.0


@dataclass
class InstanceSpec:
    name: str
    vcpu: int
    mem_gb: int
    usd_per_hr: float
    tdp_w_est: float = 0.0


def load_models(path: str | Path) -> list[ModelSpec]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [ModelSpec(**m) for m in data["models"]]


def load_sweep(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_instances(sweep: dict[str, Any]) -> list[InstanceSpec]:
    return [InstanceSpec(**i) for i in sweep.get("instances", [])]


# --------------------------------------------------------------------------- #
# Candidate model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Candidate:
    model_id: str
    model_short: str
    precision: str            # bf16 | w8a8 | w4a8
    allocator: str            # glibc | tcmalloc | mimalloc
    onednn_fpmath: str        # "" | bf16
    thread_bind: str          # auto | phys
    kv_cache_space_gb: int
    max_num_batched_tokens: int
    lse_atomics: str          # off | on

    @property
    def id(self) -> str:
        """Stable short id for a candidate (used in filenames + result index)."""
        raw = "|".join(str(v) for v in asdict(self).values())
        h = hashlib.sha256(raw.encode()).hexdigest()[:10]
        return f"{self.model_short}-{self.precision}-{h}"

    def label(self) -> str:
        parts = [self.precision, f"alloc={self.allocator}", f"bind={self.thread_bind}",
                 f"kv={self.kv_cache_space_gb}g", f"mnbt={self.max_num_batched_tokens}"]
        if self.onednn_fpmath:
            parts.append("fpmath=bf16")
        if self.lse_atomics == "on":
            parts.append("lse")
        return " ".join(parts)

    def is_baseline(self) -> bool:
        """The fair baseline: bf16 with every lever at its documented default."""
        return (self.precision == "bf16" and self.allocator == "glibc"
                and self.onednn_fpmath == "" and self.thread_bind == "auto"
                and self.lse_atomics == "off")


# --------------------------------------------------------------------------- #
# Candidate generation
# --------------------------------------------------------------------------- #
def _lever_values(sweep: dict[str, Any], name: str) -> list[Any]:
    return list(sweep["levers"][name]["values"])


def _lever_default(sweep: dict[str, Any], name: str) -> Any:
    return sweep["levers"][name]["default"]


def baseline_candidate(model: ModelSpec, sweep: dict[str, Any]) -> Candidate:
    """bf16 + all lever defaults -- the honest reference every win is measured against."""
    return Candidate(
        model_id=model.id,
        model_short=model.short,
        precision="bf16",
        allocator=_lever_default(sweep, "allocator"),
        onednn_fpmath=_lever_default(sweep, "onednn_fpmath"),
        thread_bind=_lever_default(sweep, "thread_bind"),
        kv_cache_space_gb=_lever_default(sweep, "kv_cache_space_gb"),
        max_num_batched_tokens=_lever_default(sweep, "max_num_batched_tokens"),
        lse_atomics=_lever_default(sweep, "lse_atomics"),
    )


def generate_candidates(
    models: list[ModelSpec],
    sweep: dict[str, Any],
    max_candidates: int | None = None,
) -> list[Candidate]:
    """Expand the lever grid per (model, precision).

    Always includes the bf16 baseline for each model. Honors sweep.search.strategy
    (grid -> full product; random -> sampled subset). Successive-halving handling is
    done by the runner; here we just produce the full candidate pool to prune from.
    """
    allocs = _lever_values(sweep, "allocator")
    fpmaths = _lever_values(sweep, "onednn_fpmath")
    binds = _lever_values(sweep, "thread_bind")
    kvs = _lever_values(sweep, "kv_cache_space_gb")
    mnbts = _lever_values(sweep, "max_num_batched_tokens")
    lses = _lever_values(sweep, "lse_atomics")

    strategy = sweep.get("search", {}).get("strategy", "grid")
    seed = sweep.get("search", {}).get("seed", 1234)
    rng = random.Random(seed)

    pool: list[Candidate] = []
    seen: set[str] = set()

    def add(c: Candidate) -> None:
        if c.id not in seen:
            seen.add(c.id)
            pool.append(c)

    for model in models:
        add(baseline_candidate(model, sweep))
        combos = itertools.product(model.precisions, allocs, fpmaths, binds, kvs, mnbts, lses)
        combos = list(combos)
        if strategy == "random" and max_candidates:
            rng.shuffle(combos)
            combos = combos[: max_candidates]
        for prec, alloc, fpmath, bind, kv, mnbt, lse in combos:
            add(Candidate(
                model_id=model.id, model_short=model.short, precision=prec,
                allocator=alloc, onednn_fpmath=fpmath, thread_bind=bind,
                kv_cache_space_gb=kv, max_num_batched_tokens=mnbt, lse_atomics=lse,
            ))

    if max_candidates is not None and len(pool) > max_candidates:
        # keep all baselines, sample the rest deterministically
        baselines = [c for c in pool if c.is_baseline()]
        others = [c for c in pool if not c.is_baseline()]
        rng.shuffle(others)
        pool = baselines + others[: max(0, max_candidates - len(baselines))]
    return pool


# --------------------------------------------------------------------------- #
# Candidate -> launch environment (what actually configures the Arm win)
# --------------------------------------------------------------------------- #
_ALLOCATOR_PRELOAD = {
    "glibc": "",
    "tcmalloc": "/usr/lib/aarch64-linux-gnu/libtcmalloc_minimal.so.4",
    "mimalloc": "/usr/local/lib/libmimalloc.so",
}


def launch_env(cand: Candidate, instance: InstanceSpec) -> dict[str, str]:
    """Environment variables + preloads that realize a candidate's optimizations.

    These map 1:1 to the measured vLLM-on-Arm levers so the harness is honest about
    exactly what produced each number.
    """
    env: dict[str, str] = {
        # Critical: pip's default vLLM wheel is GPU-oriented and fails device
        # inference on Arm CPUs. Force the CPU backend (Graviton build).
        "VLLM_TARGET_DEVICE": "cpu",
    }

    preload = _ALLOCATOR_PRELOAD.get(cand.allocator, "")
    if preload and Path(preload).exists():
        env["LD_PRELOAD"] = preload
    elif preload:
        # Host missing the shared lib — don't poison the process; record intent only.
        env["NEOSERVE_ALLOCATOR_REQUESTED"] = cand.allocator

    if cand.onednn_fpmath == "bf16":
        env["ONEDNN_DEFAULT_FPMATH_MODE"] = "BF16"

    if cand.thread_bind == "phys":
        # pin OMP threads to physical cores 0..vcpu-1 (Graviton = 1 core/vCPU)
        env["VLLM_CPU_OMP_THREADS_BIND"] = f"0-{max(0, instance.vcpu - 1)}"

    env["VLLM_CPU_KVCACHE_SPACE"] = str(cand.kv_cache_space_gb)
    # Cap KV to leave room for weights on smaller instances (honest capacity).
    headroom = max(4, int(instance.mem_gb) - 12)
    if cand.kv_cache_space_gb > headroom:
        env["VLLM_CPU_KVCACHE_SPACE"] = str(headroom)
    # LSE atomics are a libgomp build property; we surface it for provenance and
    # select the correct libgomp via deploy tooling.
    env["NEOSERVE_LSE_ATOMICS"] = str(cand.lse_atomics)
    return env


def vllm_serve_args(cand: Candidate, model: ModelSpec) -> list[str]:
    """CLI args for `vllm serve` for this candidate."""
    served_model = cand.model_id
    if cand.precision in ("w8a8", "w4a8"):
        # local quantized artifact from src/quantize/*
        served_model = str(Path("models") / f"{cand.model_short}-{cand.precision}")
    # Cap context a bit for CPU memory headroom during serving sweeps.
    max_len = min(model.ctx, 4096)
    # CPU backend is selected via VLLM_TARGET_DEVICE=cpu in launch_env().
    # Do NOT pass --device cpu: current vLLM treats --device/--device-ids as
    # numeric accelerator IDs and crashes with ValueError on the string "cpu".
    # Also omit --disable-log-requests (removed from newer CLIs; unknown = exit 2).
    return [
        "vllm", "serve", served_model,
        "--dtype", "bfloat16",
        "--max-model-len", str(max_len),
        "--max-num-batched-tokens", str(cand.max_num_batched_tokens),
    ]


if __name__ == "__main__":  # quick sanity check
    import json
    here = Path(__file__).resolve().parents[2]
    models = load_models(here / "configs" / "models.yaml")
    sweep = load_sweep(here / "configs" / "sweep.yaml")
    pool = generate_candidates(models, sweep, max_candidates=40)
    print(f"generated {len(pool)} candidates (showing 5):")
    for c in pool[:5]:
        print(" ", c.id, "|", c.label())
    print(json.dumps(launch_env(pool[1], load_instances(sweep)[1]), indent=2))
