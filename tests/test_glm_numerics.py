"""Reference gates for the upstream GLM W4A4 baseline on gfx1201.

These exercise the pinned runtime, not r9700's incompatible W4A8 experts.
Run in the ROCm image, separately from performance measurements.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm required")


def unpack_reference(packed, scales):
    # Quark reorder: low nibble first, contiguous groups of 32 K elements.
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6,
                        -0., -.5, -1, -1.5, -2, -3, -4, -6], device=packed.device)
    scale = torch.exp2(scales.float() - 127)
    scale = torch.where(scales == 255, float("nan"), scale)
    return lut[codes] * scale.repeat_interleave(32, dim=-1)


@pytest.mark.parametrize("width", [256, 4096])
def test_quark_mxfp4_unpack(width):
    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import dequant_mxfp4
    # Every byte value, both nibble positions, and varying group scales.
    packed = torch.arange(256, device="cuda").to(torch.uint8).repeat(width // 256).reshape(2, width // 2)
    scales = torch.arange(2 * width // 32, device="cuda").reshape(2, width // 32)
    scales = (scales % 28 + 108).to(torch.uint8)
    actual = dequant_mxfp4(packed, scales, torch.bfloat16)
    ref = unpack_reference(packed, scales).to(torch.bfloat16)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, ref, rtol=0, atol=0)


def test_quark_mxfp4_scale_extremes():
    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import dequant_mxfp4
    # Code 2 is exactly +1, avoiding overflow from a large E2M1 mantissa.
    scales = torch.tensor([[0, 1, 126, 127, 128, 252, 253, 254]], device="cuda", dtype=torch.uint8).repeat(2, 1)
    packed = torch.full((2, 128), 0x22, device="cuda", dtype=torch.uint8)
    actual = dequant_mxfp4(packed, scales, torch.bfloat16)
    ref = unpack_reference(packed, scales).to(torch.bfloat16)
    torch.testing.assert_close(actual, ref, rtol=0, atol=0, equal_nan=True)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="Pinned Quark HIP interprets reserved E8M0 255 as Inf instead of NaN")
def test_quark_reserved_nan_scale():
    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import dequant_mxfp4
    packed = torch.full((2, 128), 0x22, device="cuda", dtype=torch.uint8)
    scales = torch.full((2, 8), 255, device="cuda", dtype=torch.uint8)
    actual = dequant_mxfp4(packed, scales, torch.bfloat16)
    assert torch.isnan(actual).all()


@pytest.mark.parametrize("rows", [1, 32])
def test_fp8_group128_activation(rows):
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
    torch.manual_seed(73)
    x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)
    x[:, :128] = 0
    x[:, 128:256] *= 10000  # scaling must prevent FP8 overflow
    q, scale = per_token_group_quant_fp8(x, 128, use_ue8m0=False)
    grouped = x.float().reshape(rows, -1, 128)
    ref_scale = grouped.abs().amax(-1).clamp_min(1e-10) / torch.finfo(q.dtype).max
    ref_q = (grouped / ref_scale[..., None]).clamp(-448, 448).to(q.dtype).reshape_as(x)
    assert torch.isfinite(q.float()).all() and torch.isfinite(scale).all()
    torch.testing.assert_close(scale, ref_scale, rtol=1e-6, atol=0)
    # Reciprocal multiplication vs division can choose opposite neighbours at
    # an FP8 midpoint. Require nearest rounding, allowing only FP32 arithmetic
    # error around that midpoint, rather than requiring the same tie direction.
    normalized = (grouped.double() / ref_scale.double()[..., None]).reshape_as(x)
    error = (q.double() - normalized).abs()
    nearest_error = (ref_q.double() - normalized).abs()
    assert torch.all(error <= nearest_error + normalized.abs() * 2e-5 + 1e-6)


@pytest.mark.parametrize("rows", [1, 32])
def test_glm_sigmoid_routing(rows):
    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import grouped_topk
    torch.manual_seed(81)
    hidden = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)
    logits = torch.randn(rows, 288, device="cuda", dtype=torch.float32)
    bias = torch.randn(288, device="cuda", dtype=torch.float32) * .1
    values, ids = grouped_topk(hidden, logits, 8, True, num_expert_group=1,
                              topk_group=1, scoring_func="sigmoid",
                              routed_scaling_factor=2.5, e_score_correction_bias=bias)
    scores = logits.sigmoid()
    selected = (scores + bias).topk(8, dim=-1).indices
    torch.testing.assert_close(ids.sort(-1).values.long(), selected.sort(-1).values)
    ref = scores.gather(1, ids.long())
    ref = ref / ref.sum(-1, keepdim=True) * 2.5
    assert values.dtype == torch.float32 and torch.isfinite(values).all()
    torch.testing.assert_close(values, ref, rtol=1e-6, atol=1e-7)


def test_glm_clamped_swiglu():
    from vllm.model_executor.layers.fused_moe.utils import swiglu_limit_func
    torch.manual_seed(91)
    x = (torch.randn(32, 512, device="cuda") * 20).to(torch.bfloat16)
    out = torch.empty(32, 256, device="cuda", dtype=x.dtype)
    swiglu_limit_func(out, x, 10.0)
    # torch.compile fuses the activation and multiply: round to BF16 once.
    gate = x[:, :256].float().clamp(max=10)
    up = x[:, 256:].float().clamp(-10, 10)
    ref = (torch.nn.functional.silu(gate) * up).to(x.dtype)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [1, 16])
def test_w4a4_expert_subset_against_torch(rows):
    """Eight routed experts at TP8 widths, not a whole 288-expert/TP8 model.

    Reference GEMMs/unpacking/reduction are independent PyTorch operations;
    activation QDQ intentionally uses the same upstream Quark primitive.
    """
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe import FusedMoEConfig
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEParallelConfig, RoutingMethodType, ocp_mx_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.experts.ocp_mx_emulation_moe import (
        OCP_MXQuantizationEmulationTritonExperts,
    )
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend, make_mxfp4_moe_kernel,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import quant_dequant_mxfp4
    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(torch.accelerator.current_device_index())
    torch.manual_seed(101)
    experts, hidden, intermediate = 8, 4096, 256
    w1 = torch.randint(256, (experts, 2 * intermediate, hidden // 2), device="cuda", dtype=torch.uint8)
    w2 = torch.randint(256, (experts, hidden, intermediate // 2), device="cuda", dtype=torch.uint8)
    s1 = torch.full((experts, 2 * intermediate, hidden // 32), 121, device="cuda", dtype=torch.uint8)
    s2 = torch.full((experts, hidden, intermediate // 32), 119, device="cuda", dtype=torch.uint8)
    x = (torch.randn(rows, hidden, device="cuda") * 4).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(experts, device="cuda") for _ in range(rows)]).int()
    weights = torch.rand(rows, experts, device="cuda")
    weights = weights / weights.sum(-1, keepdim=True) * 2.5
    quant = ocp_mx_moe_quant_config(quant_dtype="mxfp4", weight_dtype="mxfp4",
                                  w1_scale=s1, w2_scale=s2, gemm1_clamp_limit=10.0)
    config = FusedMoEConfig(num_experts=experts, experts_per_token=experts,
                           hidden_dim=hidden, intermediate_size=intermediate,
                           num_local_experts=experts, num_logical_experts=experts,
                           moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
                           activation=MoEActivation.SILU, in_dtype=x.dtype, device="cuda:0",
                           routing_method=RoutingMethodType.Renormalize)
    with set_current_vllm_config(VllmConfig()):
        kernel = make_mxfp4_moe_kernel(moe_quant_config=quant, moe_config=config,
                                     mxfp4_backend=Mxfp4MoeBackend.EMULATION,
                                     experts_cls=OCP_MXQuantizationEmulationTritonExperts,
                                     routing_tables=None)
        actual = kernel.apply(hidden_states=x, w1=w1, w2=w2, topk_weights=weights,
                              topk_ids=ids, activation=MoEActivation.SILU,
                              global_num_experts=experts, expert_map=None,
                              apply_router_weight_on_input=False)
    xq = quant_dequant_mxfp4(x).float()
    dw1, dw2 = unpack_reference(w1, s1), unpack_reference(w2, s2)
    parts = torch.empty(rows, experts, hidden, device="cuda", dtype=x.dtype)
    for expert in range(experts):
        gu = (xq @ dw1[expert].T).to(x.dtype).float()
        gate, up = gu.chunk(2, -1)
        act = (torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10)).to(x.dtype)
        # Quark's HIP primitive requires at least 512 output elements. Runtime
        # batches all eight routed slots; pad this per-expert reference at M=1.
        aq = quant_dequant_mxfp4(act.repeat(2, 1) if rows == 1 else act)[:rows].float()
        # Triton applies routing weights before the GEMM2 BF16 store.
        slots = (ids == expert).int().argmax(-1)
        weighted = (aq @ dw2[expert].T) * weights.gather(1, slots[:, None])
        parts[torch.arange(rows, device="cuda"), slots] = weighted.to(x.dtype)
    reference = parts.float().sum(1).to(x.dtype)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, rtol=1e-2, atol=1e-2)
