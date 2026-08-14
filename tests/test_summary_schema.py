import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "canonical"

class TestSummarySchema(unittest.TestCase):
    def test_summary_structure(self):
        summary_file = RESULTS_DIR / "summary.json"
        self.assertTrue(summary_file.exists())
        data = json.loads(summary_file.read_text(encoding="utf-8"))

        self.assertIn("instance", data)
        self.assertEqual(data["instance"], "c8g.4xlarge")
        self.assertIn("models", data)
        self.assertGreaterEqual(len(data["models"]), 1)

        for m in data["models"]:
            self.assertIn("model", m)
            self.assertIn("baseline", m)
            self.assertIn("best", m)
            self.assertIn("speedup", m)
            self.assertGreater(m["speedup"], 1.0)
            self.assertIn("quality", m)
            if m.get("quality"):
                self.assertTrue(m["quality"]["passed"])
                self.assertLessEqual(m["quality"]["delta_pct"], m["quality"]["max_delta_pct"])

    def test_architecture_comparison_structure(self):
        arch_file = RESULTS_DIR / "architecture_comparison.json"
        self.assertTrue(arch_file.exists())
        data = json.loads(arch_file.read_text(encoding="utf-8"))

        self.assertIn("architectures", data)
        archs = data["architectures"]
        self.assertEqual(len(archs), 4)

        winner = [a for a in archs if a.get("winner")]
        self.assertEqual(len(winner), 1)
        self.assertIn("Graviton4", winner[0]["instance"])
        self.assertGreater(winner[0]["tokens_per_dollar"], 1_000_000)

if __name__ == "__main__":
    unittest.main()
