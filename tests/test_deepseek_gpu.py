"""Opt-in tiny GPU correctness tests; not a full-model or performance test.

Use the pinned DeepSeek image, R9K_PLATFORM=0, VLLM_PLUGINS='',
R9700_DEEPSEEK_GPU_TEST=1. No checkpoint weights are loaded.
"""
import gc
import os
from types import SimpleNamespace
import unittest

if os.environ.get("R9700_DEEPSEEK_GPU_TEST") != "1":
    raise unittest.SkipTest("Explicit GPU test opt-in required")

import torch
import vllm
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config


def qdq_reference(x):
    shape = x.shape
    blocks = x.float().reshape(-1, shape[-1] // 32, 32)
    maximum = blocks.abs().amax(-1).clamp_min(torch.finfo(torch.float32).tiny)
    scale = torch.exp2(torch.ceil(torch.log2(maximum / 448)).clamp(-127, 127))
    return ((blocks / scale[..., None]).to(torch.float8_e4m3fn).float()
            * scale[..., None]).reshape(shape).to(torch.bfloat16)


def mxfp4_hip_reference(x):
    """CPU reference for Quark HIP's BF16 scale rounding and FP4 ties-to-even."""
    blocks = x.cpu().reshape(-1, 32)
    maximum = blocks.abs().amax(-1)
    # Round max's BF16 mantissa before selecting the power-of-two E8M0 scale.
    exponent = ((maximum.view(torch.int16).int() + 32) & 0x7f80) // 128 - 127 - 2
    scale = torch.exp2(exponent.clamp(-127, 127).float())
    normalized = blocks.float() / scale[:, None]
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6.])
    distance = (normalized.abs()[..., None] - lut).abs()
    codes = torch.arange(8)
    # At an exact midpoint prefer the code with even low mantissa bit.
    priority = codes + torch.where(codes % 2 == 0, 0, 8)
    nearest = torch.where(distance == distance.amin(-1, keepdim=True),
                          priority, 99).argmin(-1)
    return (normalized.sign() * lut[nearest] * scale[:, None]).reshape_as(x).bfloat16()


def packed_cache(seed):
    """Independent fp8_ds_mla encoder: data records, then per-block scales."""
    g = torch.Generator().manual_seed(seed)
    values = torch.randn(256, 512, generator=g).to(torch.bfloat16)
    scales = torch.tensor([.25, .5, 1, 2, 4, .5, 1]).repeat(256, 1)
    codes = (values[:, :448].float().reshape(256, 7, 64)
             / scales[..., None]).to(torch.float8_e4m3fn)
    decoded = values.clone()
    decoded[:, :448] = (codes.float() * scales[..., None]).reshape(256, 448)
    cache = torch.zeros(2, 128, 584, dtype=torch.uint8)
    for block in range(2):
        rows = slice(block * 128, (block + 1) * 128)
        flat = cache[block].flatten()
        data = flat[:128 * 576].view(128, 576)
        data[:, :448] = codes[rows].view(torch.uint8).reshape(128, 448)
        data[:, 448:] = values[rows, 448:].contiguous().view(torch.uint8)
        exponents = flat[128 * 576:].view(128, 8)
        exponents[:, :7] = (scales[rows].log2() + 127).to(torch.uint8)
    return cache.cuda(), decoded


def attention_reference(q, kv, indices, lengths, sink):
    result = []
    for row in range(q.shape[0]):
        selected = kv[indices[row, :lengths[row]].long()].float()
        scores = q[row].float() @ selected.T / (512 ** .5)
        probabilities = torch.softmax(torch.cat([scores, sink[:, None]], dim=-1), -1)
        result.append(probabilities[:, :-1] @ selected)
    return torch.stack(result).to(torch.bfloat16)


class DeepseekGPU(unittest.TestCase):
    def setUp(self):
        self.assertIn("18f8f960", vllm.__version__)
        self.assertTrue(torch.version.hip)
        self.assertTrue(torch.cuda.is_available())
        self.assertIn("gfx1201", torch.cuda.get_device_properties(0).gcnArchName)
        torch.cuda.set_device(0)
        self.config = VllmConfig(device_config=DeviceConfig(device="cuda"))
        self.config.model_config = SimpleNamespace(
            dtype=torch.bfloat16, hf_config=SimpleNamespace(model_type="deepseek_v41"))
        self.context = set_current_vllm_config(self.config)
        self.context.__enter__()
        self.addCleanup(self.context.__exit__, None, None, None)

    def test_mxfp8_gpu_qdq(self):
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
            mxfp8_e4m3_quantize, dequant_mxfp8_to_bf16,
        )
        g = torch.Generator().manual_seed(9700)
        for rows in (1, 8, 16, 512):
            x = torch.randn(rows, 5120, generator=g).to(torch.bfloat16)
            x[:, :32] = 0
            x[:, 32:64] *= 128
            expected = qdq_reference(x)
            with self.subTest(rows=rows):
                values, scales = mxfp8_e4m3_quantize(x.cuda())
                actual = dequant_mxfp8_to_bf16(values, scales).cpu()
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_linear_adapter(self):
        from vllm.model_executor.kernels.linear.mxfp8 import Mxfp8LinearLayerConfig
        from vllm.model_executor.kernels.linear.mxfp8.emulation import EmulationMxfp8LinearKernel
        from vllm.model_executor.layers.quantization.quark.schemes import QuarkOCP_MX
        from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Static, kMxfp8Dynamic
        from r9700_vllm.quant.deepseek import DeepseekQuarkLinearMethod
        g = torch.Generator().manual_seed(5120)
        layer = torch.nn.Module()
        weight = (torch.randn(128, 5120, generator=g) / 64).to(torch.bfloat16)
        layer.weight = torch.nn.Parameter(weight.cuda(), False)
        layer.scheme = QuarkOCP_MX(kMxfp8Static, kMxfp8Dynamic, scale_block_rows=32)
        layer.scheme.ocp_mx_linear = EmulationMxfp8LinearKernel(
            Mxfp8LinearLayerConfig(weight_shape=(128, 5120)))
        method = DeepseekQuarkLinearMethod.__new__(DeepseekQuarkLinearMethod)
        for rows in (1, 8, 16, 512):
            with self.subTest(rows=rows):
                x = torch.randn(rows, 5120, generator=g).to(torch.bfloat16)
                expected = torch.nn.functional.linear(qdq_reference(x).float(), weight.float())
                actual = method.apply(layer, x.cuda()).cpu().float()
                torch.testing.assert_close(actual, expected, atol=.02, rtol=.01)

    def test_shared_mxfp4_odd_rows(self):
        from quark.torch.kernel.mx.hip import qdq_mxfp4_hip
        from vllm.model_executor.kernels.linear.mxfp4.base import MxFp4LinearLayerConfig
        from vllm.model_executor.kernels.linear.mxfp4.emulation import EmulationMxfp4LinearKernel
        from vllm.model_executor.layers.quantization.quark.schemes import QuarkOCP_MX
        from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Static, kMxfp4Dynamic
        from r9700_vllm.quant.deepseek import DeepseekQuarkLinearMethod
        from test_glm_numerics import unpack_reference
        torch.manual_seed(41017)
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(torch.randint(256, (5120, 144),
            device="cuda", dtype=torch.uint8), False)
        layer.weight_scale = torch.nn.Parameter(torch.randint(119, 123, (5120, 9),
            device="cuda", dtype=torch.uint8), False)
        layer.scheme = QuarkOCP_MX(kMxfp4Static, kMxfp4Dynamic)
        layer.scheme.ocp_mx_linear = EmulationMxfp4LinearKernel(
            MxFp4LinearLayerConfig(activation_quant_key=kMxfp4Dynamic))
        method = DeepseekQuarkLinearMethod.__new__(DeepseekQuarkLinearMethod)
        weight = unpack_reference(layer.weight, layer.weight_scale)
        bias = torch.randn(5120, device="cuda", dtype=torch.bfloat16) / 16
        # Reproduce the real 17-token prompt failure before testing the adapter.
        with self.assertRaisesRegex(RuntimeError, "multiple of 64"):
            layer.scheme.apply_weights(layer, torch.ones(17, 288,
                device="cuda", dtype=torch.bfloat16))
        for rows in (1, 2, 3, 8, 17, 31, 128, 511, 512):
            for with_bias in (False, True):
                with self.subTest(rows=rows, bias=with_bias):
                    x = torch.randn(rows, 288, device="cuda", dtype=torch.bfloat16)
                    x[0, :32] = 0
                    x[:, 32:64] *= 64
                    b = bias if with_bias else None
                    qdq = mxfp4_hip_reference(x).cuda()
                    # An independent even-row stock call must agree exactly
                    # with the CPU reference, including each original row.
                    stock_qdq = qdq_mxfp4_hip(torch.cat((x, x), dim=0))[:rows]
                    torch.testing.assert_close(stock_qdq, qdq, atol=0, rtol=0)
                    expected = torch.nn.functional.linear(qdq.float(), weight,
                        b.float() if b is not None else None).bfloat16()
                    actual = method.apply(layer, x, b)
                    self.assertEqual(actual.shape, (rows, 5120))
                    torch.testing.assert_close(actual, expected, atol=.03, rtol=.01)

    def test_exact_uva_all_devices(self):
        from vllm.models.deepseek_v41.common.engram import _allocate_huge_page_storage
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
        self.assertEqual(torch.cuda.device_count(), 8)
        for device in range(8):
            with self.subTest(device=device), torch.cuda.device(device):
                host = _allocate_huge_page_storage(3 * 1024 * 1024)
                self.assertIsNotNone(host)
                self.assertTrue(host.is_pinned())
                values = host.view(torch.float32)
                values.copy_(torch.arange(values.numel(), dtype=torch.float32) % 113)
                view = get_accelerator_view_from_cpu_tensor(values)
                self.assertTrue(view.is_cuda)
                # Exercise the device view after dropping explicit host references.
                expected = values[:257].clone() + 1
                del host, values
                gc.collect()
                actual = (view[:257] + 1).cpu()
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                torch.cuda.synchronize()
                del view
                gc.collect()

    def test_real_expert_offloader(self):
        from r9700_vllm.compat.deepseek import install_engram_guard, install_expert_pinning
        from vllm.model_executor.offloader.uva import UVAOffloader
        install_engram_guard()
        install_expert_pinning()
        module = torch.nn.Module()
        module.experts = torch.nn.Linear(513, 1024, bias=False, dtype=torch.bfloat16, device="cuda")
        expected = module.experts.weight.detach().cpu().clone()
        offloader = UVAOffloader(2 * 1024**2, {"experts"})
        stock = torch.Tensor.pin_memory
        self.assertIs(offloader._maybe_offload_to_cpu(module), module)
        self.assertIs(torch.Tensor.pin_memory, stock)
        self.assertTrue(module.experts.weight._vllm_is_uva_offloaded)
        self.assertEqual(offloader.cpu_offload_bytes, expected.numel() * expected.element_size())
        actual = (module.experts.weight + 0).cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_host_expert_tp8_copies_match_stock_bytes(self):
        from r9700_vllm.compat.deepseek import offload_parameters
        from r9700_vllm.compat.deepseek_loading import load_host_expert
        from r9700_vllm.compat.deepseek_load_buffer import BufferedLoadCopies
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
        from vllm.model_executor.layers.fused_moe.routed_experts import FusedMoeWeightScaleSupported
        from vllm.models.deepseek_v41.common.engram import _allocate_huge_page_storage
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
        layer = RoutedExperts.__new__(RoutedExperts)
        torch.nn.Module.__init__(layer)
        layer.quant_config = None
        layer.quant_method = SimpleNamespace()
        layer.expert_map_manager = SimpleNamespace(map_global_to_local=lambda n: n)
        generator = torch.Generator().manual_seed(410512)
        for rank in range(8):
            layer.moe_config = SimpleNamespace(tp_rank=rank, is_act_and_mul=True,
                tp_shard_with_padding=False, moe_parallel_config=SimpleNamespace(tp_size=8))
            # Real TP8 dimensions, two experts including an untouched sentinel.
            for scale in (False, True):
                for projection in ('w13', 'w2'):
                    with self.subTest(rank=rank, scale=scale, projection=projection):
                        width = (160 if scale else 2560) if projection == 'w13' else (9 if scale else 144)
                        shape = (2, 576 if projection == 'w13' else 5120, width)
                        initial = torch.full(shape, 37, dtype=torch.uint8, device='cuda')
                        stock = torch.nn.Parameter(initial.clone(), False)
                        buffered = torch.nn.Parameter(initial.clone(), False)
                        module = torch.nn.Module()
                        module.register_parameter('weight', torch.nn.Parameter(initial, False))
                        budget = SimpleNamespace(cpu_offload_bytes=0, cpu_offload_max_bytes=1,
                                                 cpu_offload_params=set())
                        offload_parameters(budget, module, '', _allocate_huge_page_storage,
                                           get_accelerator_view_from_cpu_tensor)
                        candidate = module.weight
                        stock.quant_method = candidate.quant_method = FusedMoeWeightScaleSupported.GROUP.value
                        buffered.quant_method = stock.quant_method
                        pointer = candidate.data_ptr()
                        name = projection + ('_weight_scale' if scale else '_weight')
                        for shard in (('w1', 'w3') if projection == 'w13' else ('w2',)):
                            src_shape = (2304, width) if projection == 'w13' else (5120, width * 8)
                            src = torch.randint(0, 256, src_shape, dtype=torch.uint8, generator=generator)
                            args = dict(shard_id=shard, expert_id=1, return_success=True)
                            self.assertTrue(RoutedExperts.weight_loader(layer, stock, src, name, **args))
                            with BufferedLoadCopies(64 * 1024) as staging:
                                self.assertTrue(RoutedExperts.weight_loader(
                                    layer, buffered, src, name, **args))
                                self.assertGreater(staging.copies, 0)
                                count = staging.copies
                                self.assertTrue(load_host_expert(RoutedExperts.weight_loader, layer,
                                                                candidate, src, name, **args))
                                self.assertEqual(staging.copies, count)  # host path unchanged
                        torch.cuda.synchronize()
                        self.assertEqual(candidate.data_ptr(), pointer)
                        self.assertEqual(candidate.device.type, 'cuda')
                        torch.testing.assert_close(candidate.cpu(), stock.cpu(), rtol=0, atol=0)
                        torch.testing.assert_close(buffered.cpu(), stock.cpu(), rtol=0, atol=0)
                        torch.testing.assert_close(candidate._r9700_host_view, stock.cpu(), rtol=0, atol=0)
                        self.assertTrue(torch.all(candidate._r9700_host_view[0] == 37))
                        # Keep the tiny test bounded even on allocators with delayed GC.
                        del stock, buffered, candidate, module, initial
                        gc.collect()

    def test_sparse_attention(self):
        from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
            rocm_sparse_attn_decode, rocm_sparse_attn_prefill,
        )
        cache, kv = packed_cache(41)
        extra_cache, extra_kv = packed_cache(42)
        g = torch.Generator().manual_seed(43)
        for rows in (1, 2, 8, 16):
            q = torch.randn(rows, 8, 512, generator=g).to(torch.bfloat16)
            indices = torch.full((rows, 128), -1, dtype=torch.int32)
            lengths = torch.tensor([(1, 17, 65, 127)[i % 4] for i in range(rows)], dtype=torch.int32)
            for i, length in enumerate(lengths):
                indices[i, :length] = torch.randperm(256, generator=g)[:length]
            sink = torch.linspace(-1, 1, 8)
            for compressed in (False, True):
                with self.subTest(rows=rows, compressed=compressed):
                    if compressed:
                        reference_kv = torch.cat([kv, extra_kv])
                        reference_indices = torch.full((rows, 256), -1, dtype=torch.int32)
                        for i, length in enumerate(lengths):
                            reference_indices[i, :length] = indices[i, :length]
                            reference_indices[i, length:2*length] = indices[i, :length] + 256
                        reference_lengths = lengths * 2
                    else:
                        reference_kv, reference_indices, reference_lengths = kv, indices, lengths
                    expected = attention_reference(q, reference_kv, reference_indices, reference_lengths, sink)
                    out = torch.empty_like(q, device="cuda")
                    rocm_sparse_attn_decode(
                        q=q.cuda(), kv_cache=extra_cache if compressed else None,
                        swa_k_cache=cache, swa_only=not compressed,
                        topk_indices=indices.cuda() if compressed else None,
                        topk_lens=lengths.cuda() if compressed else None,
                        swa_indices=indices.cuda(), swa_lens=lengths.cuda(),
                        swa_ragged_indices=None, swa_ragged_indptr=None,
                        topk_ragged_indices=None, topk_ragged_indptr=None,
                        attn_sink=sink.cuda(), scale=512**-.5, head_dim=512,
                        nope_head_dim=448, rope_head_dim=64, output=out)
                    torch.testing.assert_close(out.cpu(), expected, atol=.03, rtol=.02)

                    rocm_sparse_attn_prefill(
                        q=q.cuda(), kv=reference_kv[:, None, :].cuda(),
                        indices=reference_indices.cuda(), topk_length=reference_lengths.cuda(),
                        scale=512**-.5, head_dim=512, nope_head_dim=448,
                        rope_head_dim=64, attn_sink=sink.cuda(), output=out)
                    torch.testing.assert_close(out.cpu(), expected, atol=.03, rtol=.02)

    def test_swa_cache_writer(self):
        # Exercise the actual C++ writer used by the model, not only manually
        # packed reader fixtures. Identity RoPE isolates the cache quantization.
        from vllm import _custom_ops  # noqa: F401
        g = torch.Generator().manual_seed(584)
        q = torch.randn(17, 8, 512, generator=g).to(torch.bfloat16)
        kv = torch.randn(17, 512, generator=g).to(torch.bfloat16)
        cache = torch.zeros(2, 128 * 584, dtype=torch.uint8, device="cuda")
        slots = torch.arange(120, 137, device="cuda", dtype=torch.int64)
        positions = torch.arange(17, device="cuda", dtype=torch.int64)
        cos_sin = torch.cat([torch.ones(17, 32), torch.zeros(17, 32)], dim=-1).cuda()
        actual_q = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q.cuda(), kv.cuda(), cache, slots, positions, cos_sin,
            16, 1e-6, 128, False, False, True, False)
        torch.testing.assert_close(actual_q[:, :8].cpu(), q, rtol=0, atol=0)
        self.assertTrue((actual_q[:, 8:] == 0).all())
        cache = cache.cpu()
        reconstructed = []
        for slot in range(120, 137):
            block, row = divmod(slot, 128)
            data = cache[block, row * 576:(row + 1) * 576]
            scales = torch.exp2(cache[block, 128 * 576 + row * 8:128 * 576 + row * 8 + 7].float() - 127)
            nope = data[:448].view(torch.float8_e4m3fn).float().view(7, 64) * scales[:, None]
            rope = data[448:].view(torch.bfloat16)
            reconstructed.append(torch.cat([nope.flatten(), rope.float()]))
        actual = torch.stack(reconstructed).to(torch.bfloat16)
        values = kv[:, :448].float().view(17, 7, 64)
        scale = torch.exp2(torch.ceil(torch.log2(values.abs().amax(-1) / 448)))
        expected_nope = ((values / scale[..., None]).to(torch.float8_e4m3fn).float()
                         * scale[..., None]).reshape(17, 448).to(torch.bfloat16)
        expected = torch.cat([expected_nope, kv[:, 448:]], -1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
