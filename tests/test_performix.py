import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
from src.harness.performix import profile_mock, TopDown, PerformixReport

class TestPerformixPMU(unittest.TestCase):
    def test_topdown_prior_baseline(self):
        rep = profile_mock("qwen-base", "bf16 baseline", "bf16", tuned=False, seed=42)
        self.assertEqual(rep.source, "mock")
        self.assertGreater(rep.topdown.backend_bound, 30.0)
        self.assertLess(rep.topdown.ipc, 1.2)
        # Verify sum to ~100%
        total = (
            rep.topdown.retiring
            + rep.topdown.bad_speculation
            + rep.topdown.frontend_bound
            + rep.topdown.backend_bound
        )
        self.assertAlmostEqual(total, 100.0, delta=1.5)

    def test_topdown_prior_tuned(self):
        rep = profile_mock("qwen-tuned", "w4a8 tuned", "w4a8", tuned=True, seed=42)
        self.assertGreater(rep.topdown.retiring, 45.0)
        self.assertGreater(rep.topdown.ipc, 1.4)
        symbols = [h["symbol"] for h in rep.hotspots]
        self.assertTrue(any("KleidiAI" in s for s in symbols))

if __name__ == "__main__":
    unittest.main()
