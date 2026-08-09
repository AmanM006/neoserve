"""Reps aggregation, confidence intervals, and validity gates.

The strongest competitors win on *credibility*: N>=5 reps, warmup discarded,
confidence intervals, and gates that throw out thermally-throttled or noisy trials.
This module is the honesty layer.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Sequence


# Student-t 95% two-sided critical values for small n (df = n-1). Falls back to
# 1.96 for large samples. Avoids a scipy dependency.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042}


def _t95(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in _T95:
        return _T95[df]
    keys = sorted(_T95)
    if df < keys[-1]:
        # nearest tabulated df at or above
        for k in keys:
            if k >= df:
                return _T95[k]
    return 1.96


@dataclass
class Aggregate:
    """Summary of a metric across reps."""
    n: int
    mean: float
    std: float
    cv_pct: float                 # coefficient of variation (std/mean * 100)
    ci95_halfwidth: float         # +/- around the mean at 95%
    values: list[float] = field(default_factory=list)

    @property
    def ci95(self) -> tuple[float, float]:
        return (self.mean - self.ci95_halfwidth, self.mean + self.ci95_halfwidth)

    def as_dict(self) -> dict:
        lo, hi = self.ci95
        return {"n": self.n, "mean": self.mean, "std": self.std,
                "cv_pct": self.cv_pct, "ci95_lo": lo, "ci95_hi": hi}


def aggregate(values: Sequence[float]) -> Aggregate:
    vals = [float(v) for v in values if v is not None and not math.isnan(v)]
    n = len(vals)
    if n == 0:
        return Aggregate(0, float("nan"), float("nan"), float("nan"), float("nan"), [])
    mean = statistics.fmean(vals)
    std = statistics.stdev(vals) if n > 1 else 0.0
    cv = (std / mean * 100.0) if mean else 0.0
    half = (_t95(n - 1) * std / math.sqrt(n)) if n > 1 else 0.0
    return Aggregate(n, mean, std, cv, half, vals)


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile (p in [0,100]). Matches numpy default."""
    vals = sorted(float(v) for v in values)
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    rank = (p / 100.0) * (len(vals) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (rank - lo)


# --------------------------------------------------------------------------- #
# Validity gates
# --------------------------------------------------------------------------- #
@dataclass
class GateResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "reasons": list(self.reasons)}


def check_gates(
    gates: dict,
    *,
    throughput_reps: Sequence[float],
    max_cpu_temp_c: float | None = None,
    load_before_start: float | None = None,   # 1-min loadavg / vcpu
    swap_in_bytes: int | None = None,
) -> GateResult:
    """Return whether a benchmark cell is trustworthy given host conditions.

    A failed gate does not crash the run; the cell is flagged invalid and excluded
    from the winning-config decision (and shown as rejected in the report).
    """
    reasons: list[str] = []

    agg = aggregate(throughput_reps)
    max_cv = gates.get("max_run_cv_pct")
    if max_cv is not None and agg.n > 1 and agg.cv_pct > max_cv:
        reasons.append(f"throughput CV {agg.cv_pct:.1f}% > {max_cv}% (noisy)")

    temp_limit = gates.get("max_cpu_temp_c", max_cpu_temp_c)
    if max_cpu_temp_c is not None and temp_limit is not None and max_cpu_temp_c > temp_limit:
        reasons.append(f"cpu temp {max_cpu_temp_c:.0f}C > {temp_limit}C (thermal throttle)")

    load_limit = gates.get("max_load_before_start")
    if load_limit is not None and load_before_start is not None and load_before_start > load_limit:
        reasons.append(f"loadavg/vcpu {load_before_start:.2f} > {load_limit} (busy host)")

    if gates.get("require_no_swap") and swap_in_bytes:
        reasons.append(f"swap-in {swap_in_bytes} bytes during trial (memory pressure)")

    return GateResult(ok=not reasons, reasons=reasons)


if __name__ == "__main__":
    a = aggregate([101.2, 99.8, 100.5, 100.1, 100.9])
    print(a.as_dict())
    print("p95:", percentile([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 95))
    print(check_gates({"max_run_cv_pct": 15, "require_no_swap": True},
                      throughput_reps=[100, 101, 99], swap_in_bytes=0).as_dict())
