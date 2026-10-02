"""Prometheus accounting gates, independent of a running GPU server."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from glm_spec_metrics import summarize


def snapshot(rounds, proposed, accepted, positions):
    values = dict(num_drafts=rounds, num_draft_tokens=proposed,
                  num_accepted_tokens=accepted)
    lines = [f'vllm:spec_decode_{k}_total{{model_name="glm"}} {v}'
             for k, v in values.items()]
    lines += [f'vllm:spec_decode_num_accepted_tokens_per_pos_total'
              f'{{position="{i}",model_name="glm"}} {v}'
              for i, v in enumerate(positions)]
    return '\n'.join(lines)


class SpecMetrics(unittest.TestCase):
    def test_interval_and_conditional_positions(self):
        before = snapshot(10, 30, 12, [8, 3, 1])
        after = snapshot(14, 42, 18, [11, 5, 2])
        result = summarize(before, after)
        self.assertEqual(result['acceptance'], .5)
        self.assertEqual(result['accepted_per_round'], 1.5)
        self.assertEqual([p['fraction_of_rounds'] for p in result['positions']], [.75, .5, .25])
        self.assertEqual([p['fraction_of_previous'] for p in result['positions']], [.75, 2 / 3, .5])

    def test_reset_and_missing_metrics_are_not_zero_acceptance(self):
        nonzero = snapshot(10, 30, 12, [8, 3, 1])
        for after in ('', snapshot(0, 0, 0, [0, 0, 0])):
            with self.assertRaises(ValueError):
                summarize(nonzero, after)
        with self.assertRaises(ValueError):
            summarize('', '')

    def test_idle_interval_has_no_division_by_zero(self):
        text = snapshot(10, 30, 12, [8, 3, 1])
        result = summarize(text, text)
        self.assertIsNone(result['acceptance'])
        self.assertTrue(all(p['fraction_of_previous'] is None for p in result['positions']))


if __name__ == '__main__':
    unittest.main()
