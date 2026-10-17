"""Accounting checks for frozen C4 natural-EOS coding comparisons."""
import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from glm_completed_code_c4 import counters, overlap_seconds


class CompletedC4Test(unittest.TestCase):
    def test_overlap_uses_decode_interval_only(self):
        attempts = [dict(started_at=100+i*.01, ended_at=103+i*.01,
                         timing={"ttft_ms": 1000+i*100}) for i in range(4)]
        self.assertAlmostEqual(overlap_seconds(attempts), 1.67)
        attempts[-1].pop("timing")
        self.assertEqual(overlap_seconds(attempts), 0)

    def test_counters_sum_all_worker_series(self):
        raw = ("vllm:spec_decode_num_draft_tokens_total{worker=\"a\"} 20\n"
               "vllm:spec_decode_num_draft_tokens_total{worker=\"b\"} 30\n"
               "vllm:spec_decode_num_accepted_tokens_total 25\n"
               "vllm:num_preemptions_total 0\n")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.prom"
            path.write_text(raw)
            result = counters(path)
        self.assertEqual(result["spec_decode_num_draft_tokens_total"], 50)
        self.assertEqual(result["spec_decode_num_accepted_tokens_total"], 25)
        self.assertEqual(result["num_preemptions_total"], 0)


if __name__ == "__main__":
    unittest.main()
