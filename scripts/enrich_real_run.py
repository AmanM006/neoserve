#!/usr/bin/env python3
"""Post-process a finished REAL run: attach non-mock quality + PMU/perf profiles.

Use this on runs that finished before runner.py wired REAL proof paths.

    PYTHONPATH=src python scripts/enrich_real_run.py results/real-YYYYMMDD-HHMMSS
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harness import performix, quality_guard  # noqa: E402
from harness.artifacts import write_ledger  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()
    summary_path = args.run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("mock"):
        print("refusing to enrich a MOCK run", file=sys.stderr)
        return 1

    for m in summary.get("models", []):
        short = m["short"]
        best_label = m.get("best_label", "")
        precision = "w4a8" if "w4a8" in best_label else (
            "w8a8" if "w8a8" in best_label else "bf16")
        model_id = m["model"]
        quant_path = ROOT / "models" / f"{short}-{precision}"
        # quality
        if precision != "bf16" and quant_path.exists():
            q = quality_guard.evaluate_quality_for_run(
                model_id, short, precision, "wikitext", 4.0, mock=False,
                base_model_path=model_id, quant_model_path=str(quant_path))
            m["quality"] = q.as_dict()
            print(f"quality {short} {precision}: delta={q.delta_pct}% "
                  f"passed={q.passed} source={q.source}")
        else:
            print(f"skip quality for {short} precision={precision}")

        perf_dir = args.run_dir / "performix"
        base = performix.profile_for_run(
            f"{short}-baseline", m.get("baseline_label", "bf16"), "bf16",
            mock=False, tuned=False, result_dir=perf_dir)
        best = performix.profile_for_run(
            f"{short}-best", best_label, precision,
            mock=False, tuned=precision != "bf16", result_dir=perf_dir)
        m["performix_base"] = base.as_dict()
        m["performix_best"] = best.as_dict()
        print(f"performix {short}: base={base.source} best={best.source} "
              f"ipc {base.topdown.ipc}->{best.topdown.ipc}")

        # refresh cost card performix/quality if present
        card_path = args.run_dir / "cost_cards" / f"{short}.json"
        if card_path.exists():
            card = json.loads(card_path.read_text(encoding="utf-8"))
            card["quality_guard"] = m.get("quality")
            card["performix_topdown"] = best.as_dict()
            card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_ledger(args.run_dir)
    print(f"enriched {args.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
