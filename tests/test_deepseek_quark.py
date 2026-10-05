"""CPU-only checks in the pinned DeepSeek image; no model weights or GPU context.

Run with R9K_PLATFORM=0 VLLM_PLUGINS='' python3 -m unittest discover
-s tests -p test_deepseek_quark.py. These are layout checks, not serving validation.
"""

from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
    import vllm
except ImportError as exc:
    raise unittest.SkipTest("Run inside the DeepSeek vLLM image") from exc

from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.quark.quark import (
    QuarkConfig, QuarkLinearMethod,
)
from vllm.model_executor.layers.quantization.quark.schemes import QuarkOCP_MX
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp4Dynamic, kMxfp4Static, kMxfp8Dynamic, kMxfp8Static,
)
from vllm.models.deepseek_v41.quant_config import DeepseekV4FP8Config


def checkpoint_quant():
    mx4 = dict(dtype="fp4", qscheme="per_group", group_size=32,
               scale_format="e8m0", is_dynamic=False)
    return {
        "quant_method": "quark",
        "export": {"kv_cache_group": [], "pack_method": "reorder"},
        "global_quant_config": {"weight": mx4,
                                "input_tensors": {**mx4, "is_dynamic": True}},
        "layer_type_quant_config": {},
        "layer_quant_config": {"layers.0.attn.wkv": {
            "weight": dict(dtype="fp8_e4m3", qscheme="per_block",
                           block_size=[32, 32], scale_type="float8_e8m0fnu",
                           symmetric=True, is_dynamic=False),
            "input_tensors": dict(dtype="fp8_e4m3", qscheme="per_group",
                                  group_size=32, symmetric=True, is_dynamic=True),
        }},
        "exclude": [],
    }


class BareLinear(LinearBase):
    def __init__(self):
        torch.nn.Module.__init__(self)


class DeepseekQuark(unittest.TestCase):
    def setUp(self):
        if "18f8f960" not in vllm.__version__:
            self.skipTest("Requires the independently pinned 18f8f960 candidate")
        self.config = VllmConfig(device_config=DeviceConfig(device="cpu"))
        self.config.model_config = SimpleNamespace(
            dtype=torch.bfloat16, hf_config=SimpleNamespace(model_type="deepseek_v41"))
        self.context = set_current_vllm_config(self.config)
        self.context.__enter__()
        self.addCleanup(self.context.__exit__, None, None, None)
        # Test scheme selection on CPU, not GPU capability. Hardware qualification
        # is separate and must not be inferred from these mocked checks.
        # AMD model imports inspect the architecture even when only the weight
        # name mapper is requested. Supply metadata without opening a GPU.
        architecture = patch.object(torch.cuda, "get_device_properties",
                                    return_value=SimpleNamespace(gcnArchName="gfx1201"))
        architecture.start()
        self.addCleanup(architecture.stop)
        self.capability = patch.object(QuarkConfig, "_check_scheme_supported", return_value=True)
        self.capability.start()
        self.addCleanup(self.capability.stop)

    def test_quark_is_not_rewritten(self):
        quant = checkpoint_quant()
        original = deepcopy(quant)
        for requested in (None, "quark"):
            self.assertIsNone(DeepseekV4FP8Config.override_quantization_method(
                quant, requested, SimpleNamespace(model_type="deepseek_v41")))
        self.assertEqual(quant, original)

    @unittest.skipUnless(os.environ.get("DEEPSEEK_MODEL_CONFIG"), "Optional local checkpoint")
    def test_all_checkpoint_overrides(self):
        source = json.loads(Path(os.environ["DEEPSEEK_MODEL_CONFIG"]).read_text())
        raw = source["quantization_config"]
        quant = QuarkConfig.from_config(deepcopy(raw))
        self.assertEqual(len(raw["layer_quant_config"]), 226)
        for prefix in raw["layer_quant_config"]:
            with self.subTest(prefix=prefix):
                weight, activation, method = quant.get_quant_method_target(prefix, LinearBase)
                self.assertEqual((weight, activation), (kMxfp8Static, kMxfp8Dynamic))
                self.assertIs(method, QuarkLinearMethod)
        from vllm.models.deepseek_v41.amd.vl_model import DeepseekV41ForCausalLM
        quant.packed_modules_mapping = DeepseekV41ForCausalLM.packed_modules_mapping
        quant.apply_vllm_mapper(DeepseekV41ForCausalLM.hf_to_vllm_mapper.get_rename_mapper())
        for suffix in ("fused_wqa_wkv", "wq_b", "wo_a", "wo_b"):
            with self.subTest(fused=suffix):
                weight, activation, method = quant.get_quant_method_target(
                    "language_model.model.layers.0.attn." + suffix, LinearBase)
                self.assertEqual((weight, activation), (kMxfp8Static, kMxfp8Dynamic))
                self.assertIs(method, QuarkLinearMethod)

    def test_dense_mxfp8_and_expert_w4a4_dispatch(self):
        quant = QuarkConfig.from_config(checkpoint_quant())
        weight, activation, method = quant.get_quant_method_target(
            "layers.0.attn.wkv", LinearBase)
        self.assertEqual((weight, activation), (kMxfp8Static, kMxfp8Dynamic))
        self.assertIs(method, QuarkLinearMethod)
        layer = BareLinear()
        self.assertIsInstance(quant.get_quant_method(layer, "layers.0.attn.wkv"),
                              QuarkLinearMethod)
        self.assertIsInstance(layer.scheme, QuarkOCP_MX)
        self.assertEqual(layer.scheme.scale_block_rows, 32)
        weight, activation, _ = quant.get_quant_method_target(
            "layers.0.ffn.experts", RoutedExperts)
        self.assertEqual((weight, activation), (kMxfp4Static, kMxfp4Dynamic))

    def test_scale_expansion_before_tp_slicing(self):
        # Each rank must see the scales for its own rows. Distinct scale codes
        # catch accidental repeat/tile reversal and expansion after TP slicing.
        scheme = QuarkOCP_MX(kMxfp8Static, kMxfp8Dynamic, scale_block_rows=32)
        scale = torch.arange(16 * 4, dtype=torch.uint8).reshape(16, 4) + 90
        for rank in range(8):
            captured = []

            def loader(param, expanded):
                captured.append(expanded.chunk(8, dim=0)[rank])

            scheme._expanding_scale_loader(loader)(None, scale.view(torch.float8_e8m0fnu))
            expected = scale[rank * 2:rank * 2 + 2].repeat_interleave(32, dim=0)
            torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)

    def test_engram_scale_mapping(self):
        from vllm.models.deepseek_v41.amd.model import _make_deepseek_v4_weights_mapper
        mapper = _make_deepseek_v4_weights_mapper("fp4", "weight_scale")
        mapped = list(mapper.apply([
            ("layers.1.engram.embed.weight_scale", torch.empty(0)),
            ("layers.1.engram.embed.scale", torch.empty(0)),
        ]))
        self.assertEqual([n for n, _ in mapped],
                         ["model.layers.1.engram.embed_tokens.weight_scale_inv"] * 2)

    def test_vl_mapping_available_before_model_construction(self):
        from vllm.models.deepseek_v41.amd.vl_model import DeepseekV41ForCausalLM
        cls = DeepseekV41ForCausalLM
        self.assertIn("fused_wqa_wkv", cls.packed_modules_mapping)
        mapper = cls.hf_to_vllm_mapper.get_rename_mapper()
        self.assertEqual(mapper.apply_list(["layers.0.attn.wkv"]),
                         ["language_model.model.layers.0.attn.wkv"])

    def test_scoped_adapter_dispatch(self):
        from r9700_vllm.quant.deepseek import DeepseekQuarkConfig, DeepseekQuarkLinearMethod
        quant = checkpoint_quant()
        cls = DeepseekQuarkConfig
        self.assertIsNone(cls.override_quantization_method(
            quant, None, SimpleNamespace(model_type="deepseek_v41")))
        with self.assertRaisesRegex(ValueError, "requires a DeepSeek"):
            cls.override_quantization_method(quant, cls.get_name(),
                                             SimpleNamespace(model_type="glm5_next"))
        config = cls.from_config(quant)
        layer = BareLinear()
        self.assertIsInstance(config.get_quant_method(layer, "layers.0.attn.wkv"),
                              DeepseekQuarkLinearMethod)
        shared = BareLinear()
        self.assertIsInstance(config.get_quant_method(shared, "layers.0.ffn.shared_experts.down_proj"),
                              DeepseekQuarkLinearMethod)
        self.assertEqual((shared.scheme.weight_quant_key, shared.scheme.activation_quant_key),
                         (kMxfp4Static, kMxfp4Dynamic))
        self.config.model_config.hf_config.model_type = "glm5_next"
        with self.assertRaisesRegex(ValueError, "other models"):
            config.get_quant_method(BareLinear(), "layers.0.attn.wkv")

    def test_emulated_linear_preserves_activation_rounding(self):
        from vllm.model_executor.kernels.linear.mxfp8 import Mxfp8LinearLayerConfig
        from vllm.model_executor.kernels.linear.mxfp8.emulation import EmulationMxfp8LinearKernel
        from r9700_vllm.quant.deepseek import DeepseekQuarkLinearMethod
        layer = BareLinear()
        layer.weight = torch.nn.Parameter(torch.eye(64, dtype=torch.bfloat16), False)
        layer.scheme = QuarkOCP_MX(kMxfp8Static, kMxfp8Dynamic, scale_block_rows=32)
        layer.scheme.ocp_mx_linear = EmulationMxfp8LinearKernel(
            Mxfp8LinearLayerConfig(weight_shape=(64, 64)))
        method = DeepseekQuarkLinearMethod(QuarkConfig.from_config(checkpoint_quant()))
        # Two different exponent groups plus an all-zero row. Identity weights
        # isolate activation rounding instead of conflating it with GEMM error.
        x = torch.linspace(-1.7, 2.3, 128, dtype=torch.float32).view(2, 64)
        x[:, 32:] *= 128
        x = torch.cat([x, torch.zeros(1, 64)]).to(torch.bfloat16)
        blocks = x.float().reshape(3, 2, 32)
        maximum = blocks.abs().amax(-1).clamp_min(torch.finfo(torch.float32).tiny)
        scale = torch.exp2(torch.ceil(torch.log2(maximum / 448)).clamp(-127, 127))
        expected = ((blocks / scale[..., None]).to(torch.float8_e4m3fn).float()
                    * scale[..., None]).reshape_as(x).to(torch.bfloat16)
        stock = layer.scheme.apply_weights(layer, x)
        self.assertFalse(torch.equal(stock, expected), "Fixture must expose missing QDQ")
        actual = method.apply(layer, x)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_native_linear_is_not_double_quantized(self):
        from r9700_vllm.quant.deepseek import DeepseekQuarkLinearMethod
        seen = []
        layer = BareLinear()
        layer.scheme = QuarkOCP_MX(kMxfp8Static, kMxfp8Dynamic, scale_block_rows=32)
        layer.scheme.ocp_mx_linear = SimpleNamespace(
            apply_weights=lambda layer, x, bias: seen.append(x) or x)
        method = DeepseekQuarkLinearMethod(QuarkConfig.from_config(checkpoint_quant()))
        x = torch.ones(1, 64, dtype=torch.bfloat16)
        self.assertIs(method.apply(layer, x), x)
        self.assertIs(seen[0], x)

    def test_engram_guard_refuses_rounded_fallback(self):
        from r9700_vllm.compat.deepseek import install_engram_guard
        from vllm.models.deepseek_v41.common import engram
        with patch.object(engram, "_allocate_huge_page_storage", return_value=None):
            install_engram_guard()
            wrapped = engram._allocate_huge_page_storage
            install_engram_guard()
            self.assertIs(wrapped, engram._allocate_huge_page_storage)
            with self.assertRaisesRegex(RuntimeError, "refusing rounded"):
                wrapped(4096)
            self.config.model_config.hf_config.model_type = "qwen4_exp"
            self.assertIsNone(wrapped(4096))
        table = torch.empty(4096, dtype=torch.uint8)
        with patch.object(engram, "_allocate_huge_page_storage", return_value=table):
            install_engram_guard()
            self.assertIs(engram._allocate_huge_page_storage(4096), table)

    @patch.dict(os.environ, R9K_DEEPSEEK_OFFLOAD_MATRICES="0", R9K_DEEPSEEK_OFFLOAD_FIRST_LAYER="0")
    def test_single_copy_selection_accounting_and_no_recopy(self):
        from r9700_vllm.compat.deepseek import offload_parameters
        module = torch.nn.Module()
        module.experts = torch.nn.Linear(6, 5, bias=False)
        module.other = torch.nn.Linear(6, 5, bias=False)
        expected = module.experts.weight.detach().clone()
        other_ptr = module.other.weight.data_ptr()
        offloader = SimpleNamespace(cpu_offload_params={"experts"},
                                    cpu_offload_bytes=0,cpu_offload_max_bytes=1)
        from unittest.mock import Mock
        allocate = Mock(side_effect=lambda size: torch.empty(size,dtype=torch.uint8))
        view = Mock(side_effect=lambda x: x)
        # No Tensor.to('cpu') or pin_memory intermediate is permitted.
        with patch.object(torch.Tensor,'to',side_effect=AssertionError('second copy')), \
             patch.object(torch.Tensor,'pin_memory',side_effect=AssertionError('second copy')):
            offload_parameters(offloader,module,"",allocate,view)
            offload_parameters(offloader,module,"",allocate,view)
        allocate.assert_called_once_with(120)
        self.assertEqual(offloader.cpu_offload_bytes,120)
        self.assertEqual(module.other.weight.data_ptr(),other_ptr)
        self.assertTrue(module.experts.weight._vllm_is_uva_offloaded)
        torch.testing.assert_close(module.experts.weight,expected,atol=0,rtol=0)

    @patch.dict(os.environ, R9K_DEEPSEEK_OFFLOAD_MATRICES="0", R9K_DEEPSEEK_OFFLOAD_FIRST_LAYER="0")
    def test_single_copy_failure_does_not_spend_budget(self):
        from r9700_vllm.compat.deepseek import offload_parameters
        module=torch.nn.Linear(8,8,bias=False)
        original=module.weight.detach().clone()
        offloader=SimpleNamespace(cpu_offload_params=set(),cpu_offload_bytes=0,
                                  cpu_offload_max_bytes=4096)
        with self.assertRaisesRegex(RuntimeError,'refusing rounded'):
            offload_parameters(offloader,module,"",lambda n:None,lambda x:x)
        self.assertEqual(offloader.cpu_offload_bytes,0)
        torch.testing.assert_close(module.weight,original)

    def test_streaming_text_loader_finalizes_once_after_all_weights(self):
        from r9700_vllm.compat.deepseek_loading import install_optimized_loading
        from vllm.models.deepseek_v41.amd.vl_model import DeepseekV41ForCausalLM
        from vllm.model_executor.models.utils import StageMissingLayer, WeightsMapper
        events = []

        class LanguageModel(torch.nn.Module):
            def load_weights(self, weights):
                names = set()
                events.append('begin')
                for name, value in weights:
                    events.append(name)
                    names.add(name)
                events.append('finalize')
                return names

        def source():
            # Deliberately interleave vision and LM, and head/model groups.
            for name in ('language_model.model.a', 'vision.weight',
                         'language_model.lm_head.weight', 'image_start',
                         'language_model.model.b'):
                events.append('yield:' + name)
                yield name, torch.ones(1)

        model = DeepseekV41ForCausalLM.__new__(DeepseekV41ForCausalLM)
        torch.nn.Module.__init__(model)
        model.vision = StageMissingLayer('vision')
        model.aligner = StageMissingLayer('vision')
        model.language_model = LanguageModel()
        model.hf_to_vllm_mapper = WeightsMapper()
        with patch.dict(os.environ, R9K_DEEPSEEK_LOAD='stream'), \
             patch.object(DeepseekV41ForCausalLM, 'load_weights', DeepseekV41ForCausalLM.load_weights), \
             patch.object(RoutedExperts, 'weight_loader', RoutedExperts.weight_loader):
            install_optimized_loading()
            result = model.load_weights(source())
        self.assertEqual(result, {'language_model.model.a', 'language_model.model.b',
                                  'language_model.lm_head.weight'})
        self.assertEqual(events.count('begin'), 1)
        self.assertEqual(events.count('finalize'), 1)
        self.assertEqual(events[-1], 'finalize')
        self.assertLess(events.index('model.a'), events.index('yield:vision.weight'))
        self.assertTrue(model._weights_finalized)

    def test_vision_stream_copies_interleaved_tensors_and_finalizes_once(self):
        from r9700_vllm.compat.deepseek_loading import install_optimized_loading
        from vllm.models.deepseek_v41.amd.vl_model import DeepseekV41ForCausalLM
        from vllm.model_executor.models.utils import WeightsMapper
        events = []

        class LanguageModel(torch.nn.Module):
            def load_weights(self, weights):
                names = set()
                for name, value in weights:
                    events.append(name)
                    names.add(name)
                events.append('finalize')
                return names

        model = DeepseekV41ForCausalLM.__new__(DeepseekV41ForCausalLM)
        torch.nn.Module.__init__(model)
        model.vision = torch.nn.Linear(2, 2, bias=False)
        model.aligner = torch.nn.Linear(2, 2, bias=False)
        model.image_start = torch.nn.Parameter(torch.zeros(2))
        model.language_model = LanguageModel()
        model.hf_to_vllm_mapper = WeightsMapper()

        def source():
            yield 'language_model.model.a', torch.ones(1)
            self.assertEqual(events, ['model.a'])  # no checkpoint buffering
            yield 'vision.weight', torch.full((2, 2), 3.)
            torch.testing.assert_close(model.vision.weight, torch.full((2, 2), 3.))
            yield 'language_model.model.b', torch.ones(1)
            yield 'aligner.weight', torch.full((2, 2), 5.)
            yield 'image_start', torch.full((2,), 7.)

        with patch.dict(os.environ, R9K_DEEPSEEK_LOAD='stream'), \
             patch.object(DeepseekV41ForCausalLM, 'load_weights', DeepseekV41ForCausalLM.load_weights), \
             patch.object(RoutedExperts, 'weight_loader', RoutedExperts.weight_loader):
            install_optimized_loading()
            result = model.load_weights(source())
        self.assertEqual(result, {'language_model.model.a', 'language_model.model.b',
                                  'vision.weight', 'aligner.weight', 'image_start'})
        self.assertEqual(events, ['model.a', 'model.b', 'finalize'])
        torch.testing.assert_close(model.aligner.weight, torch.full((2, 2), 5.))
        torch.testing.assert_close(model.image_start, torch.full((2,), 7.))
        self.assertTrue(model._weights_finalized)

    def test_text_stream_rejects_unexpected_names(self):
        from r9700_vllm.compat.deepseek_loading import text_weights
        with self.assertRaisesRegex(RuntimeError, 'Unexpected'):
            list(text_weights([('unknown.weight', torch.ones(1))]))

    def test_streaming_keeps_other_models_loader_and_is_idempotent(self):
        from r9700_vllm.compat.deepseek_loading import install_optimized_loading
        from vllm.models.deepseek_v41.amd.vl_model import DeepseekV41ForCausalLM
        with patch.dict(os.environ, R9K_DEEPSEEK_LOAD='stream'), \
             patch.object(DeepseekV41ForCausalLM, 'load_weights', return_value={'stock'}) as original, \
             patch.object(RoutedExperts, 'weight_loader', RoutedExperts.weight_loader):
            # Mock attributes must be concrete: a Mock fabricates truthy markers.
            original._r9700_ds_stream = False
            install_optimized_loading()
            first = DeepseekV41ForCausalLM.load_weights
            install_optimized_loading()
            self.assertIs(DeepseekV41ForCausalLM.load_weights, first)
            model = SimpleNamespace(vision=torch.nn.Identity(), aligner=torch.nn.Identity())
            self.config.model_config.hf_config.model_type = 'other'
            self.assertEqual(first(model, iter(())), {'stock'})
            original.assert_called_once()

    def test_host_copy_restores_alias_after_error_and_checks_geometry(self):
        from r9700_vllm.compat.deepseek_loading import load_host_expert
        parameter = torch.nn.Parameter(torch.zeros(2, 8, dtype=torch.uint8), False)
        parameter._r9700_host_view = torch.ones_like(parameter)
        pointer = parameter.data_ptr()

        def fail(layer, param, source):
            self.assertEqual(param.data_ptr(), parameter._r9700_host_view.data_ptr())
            raise ValueError('load failed')

        with self.assertRaisesRegex(ValueError, 'load failed'):
            load_host_expert(fail, None, parameter, torch.ones_like(parameter))
        self.assertEqual(parameter.data_ptr(), pointer)
        parameter._r9700_host_view = torch.ones(8, 2, dtype=torch.uint8)
        with self.assertRaisesRegex(RuntimeError, 'alias no longer matches'):
            load_host_expert(fail, None, parameter, torch.ones_like(parameter))

    def test_other_models_keep_original_offloader(self):
        from r9700_vllm.compat.deepseek import install_expert_pinning
        from vllm.model_executor.offloader.uva import UVAOffloader
        from unittest.mock import Mock
        original=Mock(return_value='stock')
        with patch.object(UVAOffloader,'_maybe_offload_to_cpu',original):
            install_expert_pinning()
            self.config.model_config.hf_config.model_type='glm5_next'
            self.assertEqual(UVAOffloader._maybe_offload_to_cpu(None,None), 'stock')


if __name__ == "__main__":
    unittest.main()
