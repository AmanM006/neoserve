"""Serving economics: tokens-per-dollar, cost per 1M tokens, a perf-per-watt proxy,
and the Pareto frontier of latency vs cost under concurrency.

This is NeoServe's core differentiator. The rest of the field reports single-stream
tokens/sec; we answer the production question: *at my traffic and SLO, which config
serves tokens cheapest?*
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence


SECONDS_PER_HOUR = 3600.0


# --------------------------------------------------------------------------- #
# Point-level economics
# --------------------------------------------------------------------------- #
def tokens_per_dollar(throughput_tok_s: float, usd_per_hr: float) -> float:
    """Output tokens served per US dollar at a given sustained throughput."""
    if usd_per_hr <= 0:
        return float("nan")
    return throughput_tok_s * SECONDS_PER_HOUR / usd_per_hr


def cost_per_1m_tokens(throughput_tok_s: float, usd_per_hr: float) -> float:
    """USD to serve 1,000,000 output tokens at a sustained throughput."""
    tpd = tokens_per_dollar(throughput_tok_s, usd_per_hr)
    if not tpd or tpd != tpd:  # nan guard
        return float("nan")
    return 1_000_000.0 / tpd


def perf_per_watt(throughput_tok_s: float, tdp_w_est: float) -> float:
    """Tokens/sec per estimated watt.

    HONEST CAVEAT: Graviton does not expose RAPL, so `tdp_w_est` is a socket-share
    estimate, not a measurement. Use only for *relative* comparison between configs
    on the SAME instance; never publish as an absolute wattage figure.
    """
    if tdp_w_est <= 0:
        return float("nan")
    return throughput_tok_s / tdp_w_est


# --------------------------------------------------------------------------- #
# Goodput under an SLO
# --------------------------------------------------------------------------- #
@dataclass
class ServingPoint:
    """One measured (candidate, instance, request_rate) cell."""
    candidate_id: str
    label: str
    instance: str
    usd_per_hr: float
    request_rate: float                 # offered load (req/s)
    # measured, SLO-agnostic
    output_throughput_tok_s: float      # total output tokens/sec across all requests
    ttft_p95_ms: float
    tpot_p95_ms: float
    completed_req_s: float              # requests finished per second
    tdp_w_est: float = 0.0
    valid: bool = True
    reasons: list[str] = field(default_factory=list)

    def meets_slo(self, slo: dict) -> bool:
        return (self.ttft_p95_ms <= slo.get("ttft_p95_ms", float("inf"))
                and self.tpot_p95_ms <= slo.get("tpot_p95_ms", float("inf")))

    def cost_per_1m(self) -> float:
        return cost_per_1m_tokens(self.output_throughput_tok_s, self.usd_per_hr)

    def tokens_per_usd(self) -> float:
        return tokens_per_dollar(self.output_throughput_tok_s, self.usd_per_hr)

    def perf_watt(self) -> float:
        return perf_per_watt(self.output_throughput_tok_s, self.tdp_w_est)

    def as_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id, "label": self.label,
            "instance": self.instance, "usd_per_hr": self.usd_per_hr,
            "request_rate": self.request_rate,
            "output_throughput_tok_s": self.output_throughput_tok_s,
            "ttft_p95_ms": self.ttft_p95_ms, "tpot_p95_ms": self.tpot_p95_ms,
            "completed_req_s": self.completed_req_s,
            "cost_per_1m_tokens": self.cost_per_1m(),
            "tokens_per_usd": self.tokens_per_usd(),
            "perf_per_watt": self.perf_watt(),
            "valid": self.valid, "reasons": self.reasons,
        }


def cheapest_at_slo(points: Sequence[ServingPoint], slo: dict) -> ServingPoint | None:
    """Config with the lowest cost/1M tokens among those meeting the SLO."""
    elig = [p for p in points if p.valid and p.meets_slo(slo)]
    if not elig:
        # Prefer an SLO-meeting point even if a soft host gate failed (e.g. residual
        # loadavg), rather than returning None and collapsing to an arbitrary baseline.
        elig = [p for p in points if p.meets_slo(slo)]
    if not elig:
        return None
    return min(elig, key=lambda p: p.cost_per_1m())


def max_slo_goodput(points: Sequence[ServingPoint], slo: dict) -> ServingPoint | None:
    """Highest-throughput point that still satisfies the SLO (the config's usable
    operating point). Returns None if no point meets the SLO."""
    ok = [p for p in points if p.valid and p.meets_slo(slo)]
    if not ok:
        ok = [p for p in points if p.meets_slo(slo)]
    if not ok:
        return None
    return max(ok, key=lambda p: p.output_throughput_tok_s)


# --------------------------------------------------------------------------- #
# Pareto frontier: minimize cost/1M tokens, maximize SLO-meeting throughput
# --------------------------------------------------------------------------- #
def pareto_frontier(points: Sequence[ServingPoint], slo: dict) -> list[ServingPoint]:
    """Non-dominated set trading off cost/1M-tokens (lower better) against
    SLO-meeting throughput (higher better).

    A point A dominates B if A is no worse on both axes and strictly better on one.
    Prefer valid SLO-meeting points; fall back to any SLO-meeting points.
    """
    elig = [p for p in points if p.valid and p.meets_slo(slo)]
    if not elig:
        elig = [p for p in points if p.meets_slo(slo)]
    frontier: list[ServingPoint] = []
    for p in elig:
        dominated = False
        for q in elig:
            if q is p:
                continue
            if (q.cost_per_1m() <= p.cost_per_1m()
                    and q.output_throughput_tok_s >= p.output_throughput_tok_s
                    and (q.cost_per_1m() < p.cost_per_1m()
                         or q.output_throughput_tok_s > p.output_throughput_tok_s)):
                dominated = True
                break
        if not dominated:
            frontier.append(p)
    frontier.sort(key=lambda p: p.output_throughput_tok_s)
    return frontier


# --------------------------------------------------------------------------- #
# Traffic -> monthly cost projection + savings vs a baseline point
# --------------------------------------------------------------------------- #
@dataclass
class MonthlyProjection:
    tokens_per_month: float
    config_label: str
    instance: str
    usd_per_month: float
    instances_needed: float   # to sustain the peak token rate at SLO


def project_monthly_cost(
    point: ServingPoint,
    tokens_per_month: float,
    peak_tok_s: float | None = None,
) -> MonthlyProjection:
    """Cost to serve `tokens_per_month` at this config's cost/1M, and how many
    instances are needed to also cover a peak token rate at the SLO throughput."""
    usd = (tokens_per_month / 1_000_000.0) * point.cost_per_1m()
    inst = 1.0
    if peak_tok_s and point.output_throughput_tok_s > 0:
        inst = max(1.0, peak_tok_s / point.output_throughput_tok_s)
    return MonthlyProjection(
        tokens_per_month=tokens_per_month, config_label=point.label,
        instance=point.instance, usd_per_month=usd, instances_needed=inst,
    )


def savings_vs_baseline(best: ServingPoint, baseline: ServingPoint,
                        tokens_per_month: float) -> dict:
    """Absolute + percentage monthly savings of `best` over `baseline`."""
    b_cost = (tokens_per_month / 1_000_000.0) * baseline.cost_per_1m()
    x_cost = (tokens_per_month / 1_000_000.0) * best.cost_per_1m()
    saved = b_cost - x_cost
    pct = (saved / b_cost * 100.0) if b_cost else 0.0
    speedup = (best.output_throughput_tok_s / baseline.output_throughput_tok_s
               if baseline.output_throughput_tok_s else float("nan"))
    return {
        "baseline_usd_per_month": b_cost,
        "best_usd_per_month": x_cost,
        "usd_saved_per_month": saved,
        "pct_saved": pct,
        "throughput_speedup_x": speedup,
    }


if __name__ == "__main__":
    base = ServingPoint("base", "bf16 default", "c8g.4xlarge", 0.637, 8,
                        output_throughput_tok_s=180, ttft_p95_ms=1200, tpot_p95_ms=90,
                        completed_req_s=1.4, tdp_w_est=140)
    best = ServingPoint("w4a8", "w4a8 tuned", "c8g.4xlarge", 0.637, 8,
                        output_throughput_tok_s=900, ttft_p95_ms=800, tpot_p95_ms=60,
                        completed_req_s=7.0, tdp_w_est=140)
    slo = {"ttft_p95_ms": 2000, "tpot_p95_ms": 120}
    print("baseline $/1M:", round(base.cost_per_1m(), 3))
    print("best     $/1M:", round(best.cost_per_1m(), 3))
    print(savings_vs_baseline(best, base, 5_000_000_000))
