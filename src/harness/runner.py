"""NeoServe end-to-end orchestrator + CLI.

Pipeline per model:
  1. generate candidates (bf16 baseline + lever/precision combos)
  2. successive-halving: cheap saturation probe prunes to the top survivors
  3. full concurrency-grid benchmark (N reps + confidence intervals + validity gates)
  4. score each config on cost-at-SLO; build the latency-vs-cost Pareto frontier
  5. quality-guard the quantized winner (fall back if perplexity budget is blown)
  6. Arm Performix top-down on baseline + winner (the "why")
  7. emit reusable artifacts (cost card, HF model card, tuned serving recipe)
  8. write raw results, summary.json, report.html, and a SHA-256 provenance ledger

MOCK mode runs anywhere with the grounded simulator; REAL mode runs on Graviton4.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import (artifacts, bench_serving, config_space, economics, performix,
               quality_guard, stats)
from .config_space import Candidate, InstanceSpec, ModelSpec

app = typer.Typer(add_completion=False, help="NeoServe: Arm serving cost/SLO optimizer")
console = Console()

PROBE_RATE = 1e9  # saturating rate to rank raw capacity during the halving probe


# --------------------------------------------------------------------------- #
# One benchmark cell -> aggregated ServingPoint (+ raw reps for provenance)
# --------------------------------------------------------------------------- #
def run_cell(
    cand: Candidate, model: ModelSpec, instance: InstanceSpec, rate: float,
    reps: int, input_len: int, output_len: int, num_prompts: int,
    gates: dict, mock: bool, rng: random.Random,
    server: Optional[bench_serving.VllmServer] = None,
    result_dir: Optional[Path] = None,
) -> tuple[economics.ServingPoint, list[dict]]:
    rep_metrics: list[bench_serving.RepMetric] = []
    for _ in range(reps):
        if mock:
            rm = bench_serving.simulate_cell(cand, model, instance, rate,
                                             input_len, output_len, rng)
        else:
            rm = bench_serving.run_real_cell(cand, model, instance, rate,
                                             num_prompts, input_len, output_len,
                                             result_dir or Path("results/tmp"))
        rep_metrics.append(rm)

    thru = [r.output_throughput_tok_s for r in rep_metrics]
    agg_thru = stats.aggregate(thru)
    last = rep_metrics[-1]
    gate = stats.check_gates(
        gates, throughput_reps=thru,
        max_cpu_temp_c=last.cpu_temp_c, load_before_start=last.load_before_start,
        swap_in_bytes=last.swap_in_bytes)

    point = economics.ServingPoint(
        candidate_id=cand.id, label=cand.label(), instance=instance.name,
        usd_per_hr=instance.usd_per_hr, request_rate=rate,
        output_throughput_tok_s=agg_thru.mean,
        ttft_p95_ms=stats.aggregate([r.ttft_p95_ms for r in rep_metrics]).mean,
        tpot_p95_ms=stats.aggregate([r.tpot_p95_ms for r in rep_metrics]).mean,
        completed_req_s=stats.aggregate([r.completed_req_s for r in rep_metrics]).mean,
        tdp_w_est=instance.tdp_w_est, valid=gate.ok, reasons=gate.reasons,
    )
    raw = [{"candidate_id": cand.id, "label": cand.label(), "rate": rate,
            **rm.as_dict()} for rm in rep_metrics]
    return point, raw


# --------------------------------------------------------------------------- #
# Per-model optimization
# --------------------------------------------------------------------------- #
def optimize_model(
    model: ModelSpec, sweep: dict, instance: InstanceSpec, mock: bool,
    rng: random.Random, result_dir: Path, tokens_per_month: float,
    max_candidates: int,
) -> tuple[dict, list[dict]]:
    search = sweep["search"]
    conc = sweep["concurrency"]
    gates = sweep["gates"]
    slo = sweep["slo"]

    baseline_cand = config_space.baseline_candidate(model, sweep)
    pool = config_space.generate_candidates([model], sweep, max_candidates=max_candidates)
    # ensure baseline present
    if all(c.id != baseline_cand.id for c in pool):
        pool.insert(0, baseline_cand)

    raw_rows: list[dict] = []

    # ---- Stage 1: saturation probe (rank raw capacity, prune) ----
    console.log(f"[{model.short}] probing {len(pool)} candidates ...")
    probe_scores: list[tuple[Candidate, float]] = []
    for c in pool:
        p, raw = run_cell(c, model, instance, PROBE_RATE, reps=2,
                          input_len=search["probe"]["input_len"],
                          output_len=search["probe"]["output_len"],
                          num_prompts=search["probe"]["prompts"], gates=gates,
                          mock=mock, rng=rng, result_dir=result_dir / "raw")
        raw_rows.extend(raw)
        probe_scores.append((c, p.output_throughput_tok_s))
    probe_scores.sort(key=lambda t: t[1], reverse=True)
    keep_n = max(3, int(len(pool) * search["probe"]["keep_fraction"]))
    survivors = [c for c, _ in probe_scores[:keep_n]]
    if all(c.id != baseline_cand.id for c in survivors):
        survivors.append(baseline_cand)  # always carry the honest baseline forward
    console.log(f"[{model.short}] {len(survivors)} survivors -> full concurrency grid")

    # ---- Stage 2: full concurrency grid with reps ----
    all_points: list[economics.ServingPoint] = []
    baseline_points: list[economics.ServingPoint] = []
    for c in survivors:
        for rate in conc["request_rates"]:
            pt, raw = run_cell(c, model, instance, float(rate), reps=conc["reps"],
                               input_len=search["full"]["input_len"],
                               output_len=search["full"]["output_len"],
                               num_prompts=search["full"]["prompts"], gates=gates,
                               mock=mock, rng=rng, result_dir=result_dir / "raw")
            raw_rows.extend(raw)
            all_points.append(pt)
            if c.id == baseline_cand.id:
                baseline_points.append(pt)

    # ---- Scoring ----
    frontier = economics.pareto_frontier(all_points, slo)
    winner = economics.cheapest_at_slo(all_points, slo)

    # baseline operating point (best SLO-meeting; else highest-throughput valid)
    baseline_pt = economics.max_slo_goodput(baseline_points, slo)
    if baseline_pt is None:
        valid_bp = [p for p in baseline_points if p.valid]
        baseline_pt = max(valid_bp, key=lambda p: p.output_throughput_tok_s) if valid_bp else baseline_points[0]

    # ---- Quality guard on the quantized winner (fall back if it fails) ----
    quality = None
    if winner is not None:
        winner_cand = next(c for c in survivors if c.id == winner.candidate_id)
        ordered = sorted([p for p in all_points if p.valid and p.meets_slo(slo)],
                         key=lambda p: p.cost_per_1m())
        for cand_point in ordered:
            wc = next(c for c in survivors if c.id == cand_point.candidate_id)
            if wc.precision == "bf16":
                winner, winner_cand, quality = cand_point, wc, None
                break
            q = quality_guard.evaluate_quality_mock(
                model.id, model.short, wc.precision, model.quality_task,
                model.quality_max_ppl_delta_pct)
            if q.passed:
                winner, winner_cand, quality = cand_point, wc, q.as_dict()
                break
        else:
            winner_cand = next(c for c in survivors if c.id == winner.candidate_id)
    else:
        winner_cand = baseline_cand
        winner = baseline_pt

    # ---- Performix top-down on baseline + winner ----
    perf_base = performix.profile_mock(baseline_cand.id, baseline_cand.label(),
                                       "bf16", tuned=False)
    perf_best = performix.profile_mock(winner_cand.id, winner_cand.label(),
                                       winner_cand.precision,
                                       tuned=not winner_cand.is_baseline())

    # ---- Economics + savings ----
    savings = economics.savings_vs_baseline(winner, baseline_pt, tokens_per_month)
    savings["tokens_per_month_example"] = tokens_per_month

    # ---- Artifacts ----
    card = artifacts.build_cost_card(model, winner, baseline_pt, quality,
                                     perf_best.as_dict(), savings, mock)
    artifacts.write_cost_card(result_dir, model, card)
    if winner_cand.precision != "bf16":
        artifacts.write_model_card(result_dir, model, winner_cand.precision, card)
    artifacts.write_serving_recipe(result_dir, winner_cand, model, instance)

    model_summary = {
        "model": model.id, "short": model.short, "instance": instance.name,
        "baseline_label": baseline_cand.label(),
        "best_label": winner_cand.label(),
        "baseline": _pt_summary(baseline_pt),
        "best": _pt_summary(winner),
        "speedup": savings["throughput_speedup_x"],
        "quality": quality,
        "performix_base": perf_base.as_dict(),
        "performix_best": perf_best.as_dict(),
        "savings": savings,
        "frontier": [_pt_summary(p) for p in frontier],
    }
    return model_summary, raw_rows


def _pt_summary(p: economics.ServingPoint) -> dict:
    d = p.as_dict()
    d["cost_per_1m"] = p.cost_per_1m()
    return d


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@app.command()
def run(
    mock: bool = typer.Option(True, "--mock/--real", help="Grounded simulator vs real Graviton4"),
    models_path: str = typer.Option("configs/models.yaml", "--models"),
    sweep_path: str = typer.Option("configs/sweep.yaml", "--sweep"),
    instance: str = typer.Option("c8g.4xlarge", "--instance", help="Instance name from sweep.yaml"),
    out: str = typer.Option("results", "--out"),
    max_candidates: int = typer.Option(48, "--max-candidates"),
    tokens_per_month: float = typer.Option(5_000_000_000, "--tokens-per-month",
                                           help="Traffic for the savings example"),
    seed: int = typer.Option(1234, "--seed"),
):
    """Run the full optimization sweep and emit artifacts + report."""
    root = Path(__file__).resolve().parents[2]
    models = config_space.load_models(root / models_path)
    sweep = config_space.load_sweep(root / sweep_path)
    instances = {i.name: i for i in config_space.load_instances(sweep)}
    if instance not in instances:
        raise typer.BadParameter(f"instance '{instance}' not in {list(instances)}")
    inst = instances[instance]

    run_id = f"{'mock' if mock else 'real'}-{time.strftime('%Y%m%d-%H%M%S')}"
    result_dir = root / out / run_id
    result_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    console.rule(f"[bold]NeoServe {run_id}  |  {inst.name}  |  {'MOCK' if mock else 'REAL'}")
    model_summaries, all_raw = [], []
    for model in models:
        ms, raw = optimize_model(model, sweep, inst, mock, rng, result_dir,
                                 tokens_per_month, max_candidates)
        model_summaries.append(ms)
        all_raw.extend(raw)

    # raw results (jsonl) for provenance
    raw_dir = result_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    with (raw_dir / "cells.jsonl").open("w", encoding="utf-8") as f:
        for row in all_raw:
            f.write(json.dumps(row) + "\n")

    summary = {
        "run_id": run_id, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mock": mock, "instance": inst.name, "slo": sweep["slo"],
        "tokens_per_month_example": tokens_per_month, "models": model_summaries,
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    artifacts.write_report_html(result_dir, summary)
    artifacts.write_ledger(result_dir)

    _print_summary(summary, result_dir)


def _print_summary(summary: dict, result_dir: Path) -> None:
    for m in summary["models"]:
        table = Table(title=f"{m['short']} on {m['instance']}  (SLO ttft<={summary['slo']['ttft_p95_ms']}ms, tpot<={summary['slo']['tpot_p95_ms']}ms)")
        table.add_column("config"); table.add_column("req/s", justify="right")
        table.add_column("tok/s", justify="right"); table.add_column("TTFT p95", justify="right")
        table.add_column("TPOT p95", justify="right"); table.add_column("$/1M", justify="right")
        table.add_column("x", justify="right")
        b, w = m["baseline"], m["best"]
        table.add_row("baseline " + m["baseline_label"], f"{b['request_rate']}",
                      f"{b['output_throughput_tok_s']:.0f}", f"{b['ttft_p95_ms']:.0f}",
                      f"{b['tpot_p95_ms']:.0f}", f"${b['cost_per_1m']:.3f}", "1.0")
        table.add_row("[green]winner " + m["best_label"], f"{w['request_rate']}",
                      f"{w['output_throughput_tok_s']:.0f}", f"{w['ttft_p95_ms']:.0f}",
                      f"{w['tpot_p95_ms']:.0f}", f"${w['cost_per_1m']:.3f}",
                      f"{m['speedup']:.2f}")
        console.print(table)
        s = m["savings"]
        console.print(f"  savings @ {s['tokens_per_month_example']:,.0f} tok/mo: "
                      f"[bold green]${s['usd_saved_per_month']:,.0f}/mo ({s['pct_saved']:.0f}%)[/] "
                      f"| Performix retiring {m['performix_base']['topdown']['retiring']}% -> "
                      f"{m['performix_best']['topdown']['retiring']}%, "
                      f"IPC {m['performix_base']['topdown']['ipc']} -> {m['performix_best']['topdown']['ipc']}")
    console.rule(f"[bold]artifacts + report -> {result_dir}")


if __name__ == "__main__":
    app()
