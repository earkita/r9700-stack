"""Opt-in fused prefill numerics and allocation regression on gfx1201."""
import gc
import json
import os
import unittest

if os.environ.get("R9700_DEEPSEEK_GPU_TEST") != "1":
    raise unittest.SkipTest("Explicit GPU test opt-in required")

import torch
from r9700_vllm.attn.deepseek_indexer_prefill import prefill_logits
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import fp8_mqa_logits_torch


class DeepseekPrefillGPU(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.version.hip)
        self.assertIn("gfx1201", torch.cuda.get_device_properties(0).gcnArchName)
        torch.manual_seed(970041)

    def inputs(self, m, n, strided=False):
        step = 2 if strided else 1
        q = torch.randn(m * step, 32, 128, device="cuda").to(torch.float8_e4m3fn)[::step]
        k = torch.randn(n * step, 128, device="cuda").to(torch.float8_e4m3fn)[::step]
        scales = torch.exp2(torch.arange(n, device="cuda") % 7 - 4.).reshape(n, 1)
        weights = torch.randn(m * step, 32, device="cuda")[::step]
        # Disjoint sequence ranges, unequal lengths, and an empty visible range.
        starts = (torch.arange(m, device="cuda", dtype=torch.int32) % 4) * (n // 4)
        ends = (starts + n // 4 - torch.arange(m, device="cuda") % 13).clamp(min=0)
        ends = torch.maximum(starts, ends).to(torch.int32)
        ends[0] = starts[0]
        return q, (k, scales), weights, starts, ends

    def reference(self, args, rows):
        q, (k, scales), weights, starts, ends = args
        # Small FP32 chunks give an independent reference without H*M*N storage.
        outputs = []
        for row in rows:
            score = q[row].float() @ k.float().T
            logits = ((score * scales.flatten()).relu() * weights[row, :, None]).sum(0)
            pos = torch.arange(k.shape[0], device="cuda")
            outputs.append(logits.masked_fill((pos < starts[row]) | (pos >= ends[row]), -float("inf")))
        return torch.stack(outputs)

    def test_masks_scaling_strides_and_topk(self):
        for m, n, strided in [(7,129,False), (32,1025,True), (512,3072,False)]:
            with self.subTest(m=m, n=n, strided=strided):
                args = self.inputs(m,n,strided)
                actual = prefill_logits(*args)
                rows = list(range(min(m,8))) + ([m-1] if m>8 else [])
                expected = self.reference(args, rows)
                torch.testing.assert_close(actual[rows], expected, atol=.004, rtol=.0002)
                self.assertTrue(torch.equal(torch.isneginf(actual[rows]), torch.isneginf(expected)))
                for i,row in enumerate(rows):
                    count = int(torch.isfinite(expected[i]).sum())
                    if not count:
                        continue
                    size = min(64,count)
                    indices = actual[row].topk(size).indices
                    torch.testing.assert_close(expected[i,indices].sort().values,
                                               expected[i].topk(size).values.sort().values,
                                               atol=.004,rtol=.0002)

    def test_reject_unsupported_dtype_and_layout(self):
        args = list(self.inputs(4,129))
        bad = args.copy(); bad[0] = bad[0].to(torch.bfloat16)
        with self.assertRaisesRegex(ValueError,"Unsupported"):
            prefill_logits(*bad)
        bad = args.copy(); bad[1] = (args[1][0],torch.ones(129,2,device="cuda")[:,0])
        with self.assertRaisesRegex(ValueError,"Unsupported"):
            prefill_logits(*bad)

    def test_peak_memory_at_oom_sized_tensor(self):
        args = self.inputs(512,3072)
        measured = {}
        # Same inputs, one compile warmup per implementation, then allocation
        # measurement. This is a kernel memory test, not a model speed benchmark.
        for name,fn in [("stock_reference",fp8_mqa_logits_torch),("fused",prefill_logits)]:
            warm = fn(*args); torch.cuda.synchronize(); del warm
            gc.collect(); torch.cuda.empty_cache()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            result = fn(*args); torch.cuda.synchronize()
            measured[name] = torch.cuda.max_memory_allocated() - baseline
            self.assertEqual(tuple(result.shape),(512,3072))
            del result
        print("prefill_peak_bytes=" + json.dumps(measured), flush=True)
        self.assertLessEqual(measured["fused"],512*3072*4 + 2*1024**2)
        self.assertGreater(measured["stock_reference"],20*measured["fused"])

    def test_long_history_memory_and_selected_rows(self):
        args = self.inputs(512,65536)
        # Compile without contaminating the measured allocation peak.
        warm = prefill_logits(*args); torch.cuda.synchronize(); del warm
        gc.collect(); torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
        actual = prefill_logits(*args); torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()-baseline
        self.assertLessEqual(peak,512*65536*4+2*1024**2)
        rows = [0,1,2,3,511]
        expected = self.reference(args,rows)
        torch.testing.assert_close(actual[rows],expected,atol=.004,rtol=.0002)
        print("long_history_peak_bytes="+str(peak),flush=True)


if __name__ == "__main__":
    unittest.main()
