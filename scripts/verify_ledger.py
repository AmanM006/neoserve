#!/usr/bin/env python3
"""Re-hash every file listed in results/<run>/ledger.json and exit non-zero on drift.

Judge reproduce step:
    python scripts/verify_ledger.py results/canonical
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path, help="results/<run> directory with ledger.json")
    args = ap.parse_args()
    ledger_path = args.run_dir / "ledger.json"
    if not ledger_path.exists():
        print(f"missing {ledger_path}", file=sys.stderr)
        return 2
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    entries = ledger.get("files") or ledger.get("entries") or ledger
    if isinstance(entries, dict) and "files" not in ledger:
        # tolerate {relpath: sha} map
        file_map = entries
    elif isinstance(entries, list):
        file_map = {e["path"]: e["sha256"] for e in entries}
    else:
        file_map = {k: v for k, v in entries.items() if k != "generated_at"}

    bad = 0
    for rel, expected in file_map.items():
        if rel in ("generated_at", "schema", "run_id"):
            continue
        if isinstance(expected, dict):
            expected = expected.get("sha256") or expected.get("hash")
        path = args.run_dir / rel
        if not path.exists():
            print(f"MISSING {rel}")
            bad += 1
            continue
        got = sha256(path)
        if got != expected:
            print(f"MISMATCH {rel}\n  expected {expected}\n  got      {got}")
            bad += 1
        else:
            print(f"ok {rel}")
    if bad:
        print(f"{bad} failures", file=sys.stderr)
        return 1
    print("ledger verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
