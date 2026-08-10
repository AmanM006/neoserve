#!/usr/bin/env python3
"""Promote a finished REAL run to results/canonical only if it shows a real win.

Requires:
  - summary.json mock == false
  - best cost_per_1m < baseline cost_per_1m (or explicit --force)
  - performix_* source != mock

Usage:
    python scripts/promote_canonical.py results/real-YYYYMMDD-HHMMSS
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--force", action="store_true",
                    help="promote even without a cost win (still requires REAL + non-mock PMU)")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    summary_path = args.run_dir / "summary.json"
    if not summary_path.exists():
        print("no summary.json", file=sys.stderr)
        return 2
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("mock"):
        print("refusing to promote MOCK run", file=sys.stderr)
        return 1

    for m in summary.get("models", []):
        for key in ("performix_base", "performix_best"):
            src = (m.get(key) or {}).get("source")
            if src == "mock":
                print(f"refusing: {m.get('short')} {key} source=mock", file=sys.stderr)
                return 1
        b = m.get("baseline") or {}
        w = m.get("best") or {}
        b_cost = b.get("cost_per_1m") or b.get("cost_per_1m_tokens")
        w_cost = w.get("cost_per_1m") or w.get("cost_per_1m_tokens")
        if b_cost is None or w_cost is None:
            print("missing cost fields", file=sys.stderr)
            return 1
        if w_cost >= b_cost and not args.force:
            print(
                f"no win for {m.get('short')}: best ${w_cost:.4f}/1M >= "
                f"baseline ${b_cost:.4f}/1M (use --force to override)",
                file=sys.stderr,
            )
            return 1
        print(f"win {m.get('short')}: ${w_cost:.4f}/1M vs baseline ${b_cost:.4f}/1M "
              f"({m.get('best_label')})")

    canon = root / "results" / "canonical"
    backup = root / "results" / f"canonical-backup-{summary.get('run_id', 'prev')}"
    if canon.exists():
        if backup.exists():
            shutil.rmtree(backup)
        shutil.copytree(canon, backup)
        print(f"backed up previous canonical -> {backup}")
        shutil.rmtree(canon)
    shutil.copytree(args.run_dir, canon, ignore=shutil.ignore_patterns("raw"))
    # keep raw JSON only (not huge serve logs)
    raw_src = args.run_dir / "raw"
    raw_dst = canon / "raw"
    raw_dst.mkdir(exist_ok=True)
    if raw_src.exists():
        for p in raw_src.glob("*.json"):
            shutil.copy2(p, raw_dst / p.name)
        cells = raw_src / "cells.jsonl"
        if cells.exists():
            shutil.copy2(cells, raw_dst / "cells.jsonl")
    from harness.artifacts import write_ledger
    write_ledger(canon)
    print(f"promoted {args.run_dir} -> {canon}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
