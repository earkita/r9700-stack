"""Classification and accounting checks for the bounded GLM quality runner."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

BENCH = Path(__file__).resolve().parents[1] / "bench"
sys.path.insert(0, str(BENCH))
spec = importlib.util.spec_from_file_location("glm_quality", BENCH / "glm_quality.py")
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


class Quality(unittest.TestCase):
    niah_case = {"id": "niah_8192_depth095", "scoring": "exact",
                 "expected": "R9700-NIAH-095-7DD029"}

    def test_niah_retrieval_is_separate_from_format(self):
        response = {"finish_reason": "stop", "content": self.niah_case["expected"] + "\nExtra explanation."}
        scores = quality.niah_scores(response, self.niah_case)
        self.assertEqual(scores, dict(strict_exact_match="FAIL", semantic_needle_retrieval="PASS",
            output_format_compliance="FAIL", generation_completed="PASS", backend_runtime_error="NO",
            retrieval_channels=["content"]))
        response["content"] = "R9700-NIAH-095-FFFFFF"
        scores = quality.niah_scores(response, self.niah_case)
        self.assertEqual(scores["semantic_needle_retrieval"], "FAIL")
        self.assertEqual(scores["output_format_compliance"], "PASS")

    def test_niah_reasoning_and_completion_are_independent(self):
        response = {"finish_reason": "length", "content": "", "reasoning": self.niah_case["expected"]}
        scores = quality.niah_scores(response, self.niah_case)
        self.assertEqual(scores["semantic_needle_retrieval"], "PASS")
        self.assertEqual(scores["retrieval_channels"], ["reasoning"])
        for field in ("strict_exact_match", "output_format_compliance", "generation_completed"):
            self.assertEqual(scores[field], "FAIL")
        self.assertEqual(quality.niah_scores(None, self.niah_case)["backend_runtime_error"], "UNKNOWN")
        response = {"finish_reason": "stop", "content": self.niah_case["expected"] + "ABC"}
        self.assertEqual(quality.niah_scores(response, self.niah_case)["semantic_needle_retrieval"], "FAIL")

    def test_niah_summary_has_context_cache_and_separate_denominators(self):
        row = {"phase": "on_cold", "status": "wrong_completed", "tokens": {"prompt": 8192},
               "niah": quality.niah_scores({"finish_reason": "stop",
                   "content": self.niah_case["expected"] + "\nExtra"}, self.niah_case)}
        unknown = dict(row, status="api_or_measurement_error",
                       niah=quality.niah_scores(None, self.niah_case))
        group = quality.summarize([row, unknown])["niah_groups"]["8192/on_cold"]
        self.assertEqual(group["attempts"], 2)
        self.assertEqual(group["strict_exact_match"], {"FAIL": 1, "UNKNOWN": 1})
        self.assertEqual(group["semantic_needle_retrieval"], {"PASS": 1, "UNKNOWN": 1})

    def test_only_full_matched_rerun_can_mark_non_reproducible(self):
        from glm_niah_report import rescore
        case = self.niah_case
        row = dict(case=case["id"], phase="on_cold", seed=1234, status="wrong_completed",
                   request={"temperature": 0, "cache_salt": "first"},
                   response={"finish_reason": "stop", "content": case["expected"] + "\nExtra"})
        rerun = dict(row, request={"temperature": 0, "cache_salt": "rerun"},
                     response={"finish_reason": "stop", "content": case["expected"]})
        result = rescore([row], {case["id"]: case}, [rerun])[0]
        self.assertEqual(result["reproduction"], "non-reproducible")
        self.assertEqual(result["niah"]["strict_exact_match"], "FAIL")
        self.assertEqual(result["status"], "wrong_completed")
        self.assertNotIn("niah", row)
        for invalid in ([], [dict(rerun, request={"temperature": 1})],
                        [dict(rerun, response={"finish_reason": "length", "content": case["expected"]})],
                        [dict(rerun, error="Cache metric observation failed")]):
            with self.assertRaises(ValueError):
                rescore([row], {case["id"]: case}, invalid)

    def test_incomplete_is_not_completed_wrong_or_correct(self):
        case = {"scoring": "exact", "expected": "ORBIT-7319"}
        for content in ("", "ORBIT-7319", "wrong"):
            result = {"finish_reason": "length", "content": content}
            self.assertEqual(quality.classify(result, case), "incomplete")
        self.assertEqual(quality.classify({"finish_reason": "stop", "content": ""}, case), "incomplete")

    def test_expected_substring_and_reasoning_are_not_exact_answers(self):
        case = {"scoring": "exact", "expected": "ORBIT-7319"}
        result = {"finish_reason": "stop", "content": "wrong", "reasoning": "ORBIT-7319"}
        self.assertEqual(quality.classify(result, case), "wrong_completed")
        result["content"] = "Wrong, correction: ORBIT-7319"
        self.assertEqual(quality.classify(result, case), "wrong_completed")
        result["content"] = " ORBIT-7319\n"
        self.assertEqual(quality.classify(result, case), "correct_completed")

    def test_structured_answer_rejects_extra_fields_and_duplicates(self):
        case = {"scoring": "json", "expected": {"code": "A", "count": 3}}
        for content, status in [('{"count":3,"code":"A"}', "correct_completed"),
                                ('{"count":3,"code":"A","extra":1}', "wrong_completed"),
                                ('{"count":3,"code":"B","code":"A"}', "wrong_completed")]:
            self.assertEqual(quality.classify({"finish_reason": "stop", "content": content}, case), status)

    def test_missing_reasoning_usage_is_unknown_not_zero(self):
        result = quality.token_counts({"completion_tokens": 10})
        self.assertIsNone(result["reasoning"])
        self.assertIsNone(result["non_reasoning_by_subtraction"])
        result = quality.token_counts({"completion_tokens": 10, "completion_tokens_details": {"reasoning_tokens": 8}})
        self.assertEqual(result["non_reasoning_by_subtraction"], 2)

    def test_summary_keeps_incomplete_and_errors_in_denominator(self):
        rows = [{"phase": "off", "status": status} for status in
                ("correct_completed", "wrong_completed", "incomplete", "api_or_measurement_error")]
        group = quality.summarize(rows)["groups"]["off"]
        self.assertEqual(group["attempts"], 4)
        self.assertEqual(group["correct_completed_over_attempts"], 0.25)
        self.assertEqual(group["accuracy_among_completed"], 0.5)
        self.assertEqual(group["completion_rate"], 0.5)

    def test_ttft_counts_reasoning_but_not_role_or_usage(self):
        class Response:
            def __enter__(self):
                return iter([
                    b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
                    b'data: {"choices":[{"delta":{"reasoning":"think"}}]}\n',
                    b'data: {"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}]}\n',
                    b'data: {"choices":[],"usage":{"completion_tokens":2}}\n',
                    b'data: [DONE]\n'])
            def __exit__(self, *args):
                pass
        with patch.object(quality, "read_url", return_value=Response()), \
             patch.object(quality.time, "monotonic", side_effect=[0, 1, 2, 3]):
            result, timing = quality.stream("http://test", {}, 10)
        self.assertEqual(result["reasoning"], "think")
        self.assertEqual(timing, {"ttft_ms": 1000, "first_content_ms": 2000, "end_to_end_ms": 3000})


if __name__ == "__main__":
    unittest.main()
