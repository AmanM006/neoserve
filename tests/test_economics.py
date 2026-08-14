import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
import math
from src.harness.economics import (
    tokens_per_dollar,
    cost_per_1m_tokens,
    perf_per_watt,
    ServingPoint,
    pareto_frontier,
    cheapest_at_slo,
    savings_vs_baseline,
)

class TestEconomics(unittest.TestCase):
    def test_cost_per_1m_math(self):
        usd_hr = 1.00
        thru = 100.0
        tpd = tokens_per_dollar(thru, usd_hr)
        self.assertAlmostEqual(tpd, 360000.0)
        cpm = cost_per_1m_tokens(thru, usd_hr)
        self.assertAlmostEqual(cpm, 1_000_000.0 / 360000.0, places=4)

    def test_graviton4_exact_economics(self):
        usd_hr = 0.63712
        thru = 237.51781047849116
        cpm = cost_per_1m_tokens(thru, usd_hr)
        self.assertAlmostEqual(cpm, 0.7451137, places=4)

    def test_serving_point_slo_gate(self):
        slo = {"ttft_p95_ms": 3000, "tpot_p95_ms": 200}
        pt_pass = ServingPoint(
            candidate_id="c1", label="pass", instance="c8g", usd_per_hr=0.63712,
            request_rate=2.0, output_throughput_tok_s=200.0,
            ttft_p95_ms=450.0, tpot_p95_ms=110.0, completed_req_s=2.0,
            tdp_w_est=140.0, valid=True
        )
        self.assertTrue(pt_pass.meets_slo(slo))

        pt_fail = ServingPoint(
            candidate_id="c2", label="fail", instance="c8g", usd_per_hr=0.63712,
            request_rate=8.0, output_throughput_tok_s=300.0,
            ttft_p95_ms=3500.0, tpot_p95_ms=250.0, completed_req_s=8.0,
            tdp_w_est=140.0, valid=True
        )
        self.assertFalse(pt_fail.meets_slo(slo))

    def test_savings_calculation(self):
        pt_base = ServingPoint(
            candidate_id="base", label="bf16", instance="c8g", usd_per_hr=0.63712,
            request_rate=1.0, output_throughput_tok_s=122.3866,
            ttft_p95_ms=403.6, tpot_p95_ms=99.2, completed_req_s=0.95,
            tdp_w_est=140.0, valid=True
        )
        pt_best = ServingPoint(
            candidate_id="best", label="w4a8", instance="c8g", usd_per_hr=0.63712,
            request_rate=2.0, output_throughput_tok_s=237.5178,
            ttft_p95_ms=421.5, tpot_p95_ms=108.8, completed_req_s=1.85,
            tdp_w_est=140.0, valid=True
        )
        savings = savings_vs_baseline(pt_best, pt_base, tokens_per_month=5_000_000_000)
        self.assertAlmostEqual(savings["pct_saved"], 48.47, places=1)
        self.assertAlmostEqual(savings["usd_saved_per_month"], 3504.70, places=0)
        self.assertAlmostEqual(savings["throughput_speedup_x"], 1.94, places=2)

if __name__ == "__main__":
    unittest.main()
