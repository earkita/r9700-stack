"""Counter accounting for retained natural-EOS code completions."""
import sys
from pathlib import Path
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from glm_completed_code import decode_observation


class DecodeObservationTest(unittest.TestCase):
    def test_excludes_entire_first_streamed_burst(self):
        result = decode_observation([(1.0, 4), (1.1, 8), (1.25, 13)],
                                    {"completion_tokens": 14})
        self.assertEqual(result["first_update_tokens"], 4)
        self.assertEqual(result["last_update_tokens"], 13)
        self.assertEqual(result["final_usage_minus_last_update"], 1)
        self.assertAlmostEqual(result["decode_tps"], 36)
        self.assertEqual(len(result["update_gaps_ms"]), 2)

    def test_duplicate_counter_is_not_an_extra_token(self):
        result = decode_observation([(1.0, 4), (1.05, 4), (1.1, 6)],
                                    {"completion_tokens": 6})
        self.assertAlmostEqual(result["decode_tps"], 20)

    def test_rejects_missing_or_decreasing_counters(self):
        for events in ([(1.0, None), (2.0, 5)],
                       [(1.0, 4), (2.0, 3)],
                       [(1.0, 4)],
                       [(1.0, 4), (1.0, 5)]):
            with self.subTest(events=events), self.assertRaises(ValueError):
                decode_observation(events, {"completion_tokens": 5})


if __name__ == "__main__":
    unittest.main()
