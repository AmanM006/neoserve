"""NeoServe MCP server: expose the tuned result index to any MCP client
(Claude Code, Cursor, GitHub Copilot, etc.) so an agent can ask
"how do I serve model X cheapest on Arm at my traffic and SLO?" and get back a
concrete, benchmarked config + the exact vLLM env/flags + projected cost.

This mirrors how the Arm MCP Server surfaces migration tooling to assistants, and
makes NeoServe's results reusable, not just readable.

Run:
    NEOSERVE_RESULTS=results/canonical python -m mcp.server
Register the process as an MCP stdio server in your client.

Tools:
    list_models()                       -> models with a tuned result
    recommend_config(model, ...)        -> winning config, env, cost, savings
    get_serving_recipe(model)           -> Dockerfile/compose/run.sh for the winner
    project_cost(model, tokens_per_month)-> monthly $ + savings vs bf16 baseline
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional


def _results_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    env = os.environ.get("NEOSERVE_RESULTS")
    if env:
        p = Path(env)
        return p if p.is_absolute() else root / p
    canonical = root / "results" / "canonical"
    if canonical.exists():
        return canonical
    # else latest run
    runs = sorted((root / "results").glob("*/summary.json"))
    if not runs:
        raise FileNotFoundError("no NeoServe results found; run the harness first")
    return runs[-1].parent


def _load_summary() -> dict:
    return json.loads((_results_dir() / "summary.json").read_text(encoding="utf-8"))


def _find_model(summary: dict, model: str) -> Optional[dict]:
    for m in summary["models"]:
        if model in (m["short"], m["model"]):
            return m
    return None


# --------------------------------------------------------------------------- #
# Pure functions (also unit-testable without the MCP runtime)
# --------------------------------------------------------------------------- #
def list_models_impl() -> list[dict]:
    s = _load_summary()
    return [{"short": m["short"], "id": m["model"], "instance": m["instance"],
             "winner": m["best_label"], "cost_per_1m": m["best"]["cost_per_1m"]}
            for m in s["models"]]


def recommend_config_impl(model: str, tokens_per_month: Optional[float] = None) -> dict:
    s = _load_summary()
    m = _find_model(s, model)
    if not m:
        return {"error": f"no tuned result for '{model}'", "available": [x["short"] for x in s["models"]]}
    best, base = m["best"], m["baseline"]
    savings = dict(m["savings"])
    if tokens_per_month:
        base_cost = tokens_per_month / 1e6 * base["cost_per_1m"]
        best_cost = tokens_per_month / 1e6 * best["cost_per_1m"]
        savings = {"tokens_per_month_example": tokens_per_month,
                   "baseline_usd_per_month": base_cost, "best_usd_per_month": best_cost,
                   "usd_saved_per_month": base_cost - best_cost,
                   "pct_saved": (1 - best_cost / base_cost) * 100 if base_cost else 0,
                   "throughput_speedup_x": m["speedup"]}
    return {
        "model": m["model"], "instance": m["instance"], "mock": s.get("mock", True),
        "slo": s["slo"],
        "winning_config": m["best_label"],
        "operating_point": {"request_rate": best["request_rate"],
                            "output_throughput_tok_s": best["output_throughput_tok_s"],
                            "ttft_p95_ms": best["ttft_p95_ms"], "tpot_p95_ms": best["tpot_p95_ms"]},
        "cost_per_1m_tokens_usd": best["cost_per_1m"],
        "baseline_cost_per_1m_tokens_usd": base["cost_per_1m"],
        "quality_guard": m.get("quality"),
        "performix_ipc": {"baseline": m["performix_base"]["topdown"]["ipc"],
                          "winner": m["performix_best"]["topdown"]["ipc"]},
        "savings": savings,
    }


def get_serving_recipe_impl(model: str) -> dict:
    s = _load_summary()
    m = _find_model(s, model)
    if not m:
        return {"error": f"no tuned result for '{model}'"}
    d = _results_dir() / "serving_recipe" / m["short"]
    files = {}
    for name in ("Dockerfile.arm64", "compose.yaml", "run.sh"):
        fp = d / name
        if fp.exists():
            files[name] = fp.read_text(encoding="utf-8")
    return {"model": m["model"], "files": files}


def project_cost_impl(model: str, tokens_per_month: float) -> dict:
    return recommend_config_impl(model, tokens_per_month=tokens_per_month)["savings"]


# --------------------------------------------------------------------------- #
# MCP wiring
# --------------------------------------------------------------------------- #
def build_server():
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("neoserve")

    @mcp.tool()
    def list_models() -> list[dict]:
        """List models that have a NeoServe-tuned serving config."""
        return list_models_impl()

    @mcp.tool()
    def recommend_config(model: str, tokens_per_month: float | None = None) -> dict:
        """Recommend the cheapest SLO-meeting Arm serving config for a model.

        Args:
            model: model short name (e.g. 'llama31-8b') or full HF id.
            tokens_per_month: optional traffic to compute a monthly cost + savings.
        """
        return recommend_config_impl(model, tokens_per_month)

    @mcp.tool()
    def get_serving_recipe(model: str) -> dict:
        """Return the tuned Dockerfile/compose/run.sh for a model's winning config."""
        return get_serving_recipe_impl(model)

    @mcp.tool()
    def project_cost(model: str, tokens_per_month: float) -> dict:
        """Project monthly serving cost + savings vs the bf16 baseline."""
        return project_cost_impl(model, tokens_per_month)

    return mcp


if __name__ == "__main__":
    build_server().run()
