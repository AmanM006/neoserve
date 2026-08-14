import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
from src.mcp.server import (
    list_models_impl,
    recommend_config_impl,
    get_serving_recipe_impl,
    project_cost_impl,
)

class TestMCPServer(unittest.TestCase):
    def test_list_models(self):
        models = list_models_impl()
        self.assertIsInstance(models, list)
        self.assertGreaterEqual(len(models), 1)
        shorts = [m["short"] for m in models]
        self.assertIn("qwen25-1p5b", shorts)

    def test_recommend_config_qwen(self):
        rec = recommend_config_impl("qwen25-1p5b", tokens_per_month=5_000_000_000)
        self.assertNotIn("error", rec)
        self.assertEqual(rec["instance"], "c8g.4xlarge")
        self.assertIn("w4a8", rec["winning_config"])
        self.assertAlmostEqual(rec["cost_per_1m_tokens_usd"], 0.7451, places=3)
        self.assertAlmostEqual(rec["baseline_cost_per_1m_tokens_usd"], 1.4461, places=3)
        self.assertGreater(rec["savings"]["usd_saved_per_month"], 3000.0)

    def test_get_serving_recipe(self):
        recipe = get_serving_recipe_impl("qwen25-1p5b")
        self.assertNotIn("error", recipe)
        self.assertIn("files", recipe)
        self.assertIn("Dockerfile.arm64", recipe["files"])
        self.assertIn("compose.yaml", recipe["files"])

    def test_project_cost(self):
        proj = project_cost_impl("qwen25-1p5b", tokens_per_month=1_000_000_000)
        self.assertIn("usd_saved_per_month", proj)
        self.assertIn("pct_saved", proj)
        self.assertAlmostEqual(proj["pct_saved"], 48.47, places=1)

if __name__ == "__main__":
    unittest.main()
