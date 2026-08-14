import json
import hashlib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "canonical"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

class TestLedgerIntegrity(unittest.TestCase):
    def setUp(self):
        self.ledger_path = RESULTS_DIR / "ledger.json"
        self.assertTrue(self.ledger_path.exists(), f"Missing ledger at {self.ledger_path}")
        self.ledger = json.loads(self.ledger_path.read_text(encoding="utf-8"))

    def test_all_hashed_files_exist_and_match(self):
        file_map = self.ledger.get("files") or self.ledger
        if isinstance(file_map, list):
            file_map = {e["path"]: e["sha256"] for e in file_map}

        checked = 0
        for rel, expected in file_map.items():
            if rel in ("generated_at", "schema", "run_id"):
                continue
            if isinstance(expected, dict):
                expected = expected.get("sha256") or expected.get("hash")
            path = RESULTS_DIR / rel
            self.assertTrue(path.exists(), f"File in ledger missing on disk: {rel}")
            got = sha256_file(path)
            self.assertEqual(got, expected, f"SHA-256 mismatch for {rel}")
            checked += 1

        self.assertGreaterEqual(checked, 10, f"Expected at least 10 tracked files, found {checked}")

    def test_canonical_is_not_mock(self):
        summary_path = RESULTS_DIR / "summary.json"
        self.assertTrue(summary_path.exists())
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertFalse(summary.get("mock", True), "Canonical summary must have mock=False")

if __name__ == "__main__":
    unittest.main()
