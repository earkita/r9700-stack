"""GPU gates for the experimental grouped-WMMA C1 W4A4 path.

Run separately from the model server. Exact activation transport first, then
per-projection references and full routed output against pinned emulation.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm required")


@pytest.mark.parametrize("family,expected", [("glm5_next", "candidate"), ("qwen", "stock")])
def test_adapter_forward_without_construction_context(monkeypatch, family, expected):
    from types import SimpleNamespace
    import vllm.config.vllm as config_module
    import vllm.model_executor.layers.fused_moe.experts.ocp_mx_emulation_moe as upstream
    import r9700_vllm.moe.w4a4 as implementation
    import r9700_vllm.compat.glm_moe as adapter

    class FakeExperts:
        def __init__(self):
            self.w1_scale_val = self.w2_scale_val = torch.tensor([117, 128], dtype=torch.uint8)
        def apply(self, *args):
            return "stock"

    context = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=family)))
    monkeypatch.setattr(config_module, "get_current_vllm_config_or_none", lambda: context)
    monkeypatch.setattr(upstream, "OCP_MXQuantizationEmulationTritonExperts", FakeExperts)
    monkeypatch.setattr(implementation, "kernel", lambda: None)
    monkeypatch.setattr(implementation, "apply_c1", lambda *args: "candidate")
    monkeypatch.setattr(adapter, "eligible", lambda *args: True)
    monkeypatch.setenv("R9K_GLM_MOE", "w4a4")
    assert adapter.patch()
    ex = FakeExperts()
    context = None  # eager forward runs after the construction context exits
    args = [None] * 15
    args[1] = SimpleNamespace(shape=(1, 4096))
    assert ex.apply(*args) == expected


@pytest.mark.parametrize("unsupported", [None, "batch", "batch_optin", "batch_large", "packed32", "packed33", "ep", "router_input", "activation", "clamp", "w4a8", "lora", "bias"])
def test_adapter_gate(unsupported):
    from types import SimpleNamespace
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import OCP_MX_Scheme
    from r9700_vllm.compat.glm_moe import eligible
    def tensor(shape, dtype):
        return torch.empty(shape, dtype=dtype, device="meta")
    ex = SimpleNamespace(ocp_mx_scheme=OCP_MX_Scheme.w_mxfp4_a_mxfp4,
                         activation_config=SimpleNamespace(clamp_limit=10.0),
                         w1_scale_val=tensor((288, 512, 128), torch.uint8),
                         w2_scale_val=tensor((288, 4096, 8), torch.uint8))
    args = dict(experts=ex, hidden_states=tensor((1, 4096), torch.bfloat16),
                w1=tensor((288, 512, 2048), torch.uint8), w2=tensor((288, 4096, 128), torch.uint8),
                topk_weights=tensor((1, 8), torch.float32), topk_ids=tensor((1, 8), torch.int32),
                activation=MoEActivation.SILU, global_num_experts=288, expert_map=None,
                a1q_scale=None, a2_scale=None, apply_router_weight_on_input=False)
    if unsupported == "batch": args["hidden_states"] = tensor((2, 4096), torch.bfloat16)
    if unsupported in ("batch_optin", "batch_large"):
        rows = 8 if unsupported == "batch_optin" else 9
        args.update(hidden_states=tensor((rows, 4096), torch.bfloat16),
                    topk_ids=tensor((rows, 8), torch.int32),
                    topk_weights=tensor((rows, 8), torch.float32), max_rows=8)
    if unsupported in ("packed32", "packed33"):
        rows = 32 if unsupported == "packed32" else 33
        args.update(hidden_states=tensor((rows, 4096), torch.bfloat16),
                    topk_ids=tensor((rows, 8), torch.int32),
                    topk_weights=tensor((rows, 8), torch.float32), max_rows=32)
    if unsupported == "ep": args["expert_map"] = tensor((288,), torch.int32)
    if unsupported == "router_input": args["apply_router_weight_on_input"] = True
    if unsupported == "activation": args["activation"] = MoEActivation.GELU
    if unsupported == "clamp": ex.activation_config.clamp_limit = None
    if unsupported == "w4a8": ex.ocp_mx_scheme = OCP_MX_Scheme.w_mxfp4_a_fp8
    if unsupported == "lora": ex._lora_context = object()
    if unsupported == "bias": ex.w1_bias = tensor((512,), torch.bfloat16)
    assert eligible(**args) == (unsupported in (None, "batch_optin", "packed32"))


@pytest.mark.parametrize("batch", ["isolated", "grouped"])
def test_32_rows_rejects_unqualified_backends(monkeypatch, batch):
    from r9700_vllm.compat.glm_moe import patch
    monkeypatch.setenv("R9K_GLM_MOE", "w4a4")
    monkeypatch.setenv("R9K_GLM_W4A4_MAX_ROWS", "32")
    monkeypatch.setenv("R9K_GLM_W4A4_BATCH", batch)
    with pytest.raises(ValueError, match="requires.*packed"):
        patch()


def test_existing_w4a8_gemm_unchanged():
    from test_moe_mxfp4 import run_case
    ok, _ = run_case(8, 2, 2, 512, 4096, 4096, 256, 700)
    assert ok


def test_qdq_transport_codes_and_extremes():
    from r9700_vllm.moe.w4a4 import encode_qdq
    values = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                           -0., -.5, -1., -1.5, -2., -3., -4., -6.], device="cuda")
    exponents = torch.tensor([-132, -126, -120, -60, 0, 60, 120], device="cuda")
    x = (values.repeat(2)[None] * torch.exp2(exponents.float())[:, None]).to(torch.bfloat16)
    q, s = encode_qdq(x)
    torch.testing.assert_close(q.float() * s, x.float(), rtol=0, atol=0)


@pytest.mark.parametrize("rows,width", [(1, 4096), (8, 256)])
def test_qdq_transport_is_exact(rows, width):
    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import quant_dequant_mxfp4
    from r9700_vllm.moe.w4a4 import encode_qdq
    torch.manual_seed(701)
    x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    groups = x.view(-1, 32)
    exponents = torch.arange(groups.shape[0], device="cuda") % 61 - 30
    groups.mul_(torch.exp2(exponents.float())[:, None])
    groups[0].zero_()
    qdq = quant_dequant_mxfp4(x)
    q, s = encode_qdq(qdq)
    reconstructed = (q.float().view(-1, 32) * s.flatten()[:, None]).view_as(qdq)
    assert torch.isfinite(reconstructed).all()
    torch.testing.assert_close(reconstructed, qdq.float(), rtol=0, atol=0)


@pytest.mark.parametrize("n,k,row_divisor", [(512, 4096, 8), (4096, 256, 1)])
def test_packed_gemm_matches_reference(n, k, row_divisor):
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import quant_dequant_mxfp4
    from r9700_vllm.moe.w4a4 import encode_qdq, gemm
    from test_glm_numerics import unpack_reference

    torch.manual_seed(702)
    experts, slots = 16, 8
    w = torch.randint(256, (experts, n, k // 2), device="cuda", dtype=torch.uint8)
    s = torch.randint(117, 129, (experts, n, k // 32), device="cuda", dtype=torch.uint8)
    ids = torch.tensor([[15, 1, 12, 3, 10, 5, 8, 7]], device="cuda", dtype=torch.int32)
    alignment = moe_align_block_size(ids, 16, experts, None)
    x = torch.randn(slots // row_divisor, k, device="cuda", dtype=torch.bfloat16)
    xq = quant_dequant_mxfp4(x)
    q, scales = encode_qdq(xq)
    weights = torch.tensor([[0., .01, .03, .1, .2, .4, .76, 1.]], device="cuda") if row_divisor == 1 else None
    out = torch.empty(slots, n, device="cuda", dtype=torch.bfloat16)
    gemm(q, scales, w, s, out, alignment, slots, row_divisor, weights)
    refs = []
    for slot, expert in enumerate(ids[0].tolist()):
        ref = xq[slot // row_divisor].float() @ unpack_reference(w[expert], s[expert]).T
        if weights is not None:
            ref *= weights[0, slot]
        refs.append(ref.to(torch.bfloat16))
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.stack(refs), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("batch", ["isolated", "grouped", "packed"])
@pytest.mark.parametrize("experts,seed,input_scale,capture,rows", [
    (8, 703, 1, False, 1), (288, 704, 4, False, 1), (288, 705, 32, False, 1),
    (288, 706, 4, True, 1), (8, 707, 1, False, 2), (288, 708, 32, False, 4),
    (288, 709, 4, False, 8), (288, 710, 4, True, 8),
])
def test_moe_matches_full_emulation(experts, seed, input_scale, capture, rows, batch):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe import FusedMoEConfig
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEParallelConfig, RoutingMethodType, ocp_mx_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.experts.ocp_mx_emulation_moe import (
        OCP_MXQuantizationEmulationTritonExperts as Upstream,
    )
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend, make_mxfp4_moe_kernel
    from vllm.v1.worker.workspace import init_workspace_manager
    from r9700_vllm.moe.w4a4 import apply_c1, apply_isolated_rows, apply_grouped
    from r9700_vllm.moe.packed_gemv import apply as apply_packed

    class Candidate(Upstream):
        def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation,
                  global_num_experts, expert_map, a1q_scale, a2_scale, workspace13, workspace2,
                  expert_tokens_meta, apply_router_weight_on_input):
            impl = (apply_packed if batch == "packed" else apply_grouped if batch == "grouped" else
                    apply_c1 if hidden_states.shape[0] == 1 else apply_isolated_rows)
            impl(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation)

    init_workspace_manager(torch.accelerator.current_device_index())
    torch.manual_seed(seed)
    hidden, intermediate, topk = 4096, 256, 8
    w1 = torch.randint(256, (experts, 512, 2048), device="cuda", dtype=torch.uint8)
    w2 = torch.randint(256, (experts, 4096, 128), device="cuda", dtype=torch.uint8)
    # Nonuniform group scales, spanning the checkpoint's observed range.
    s1 = torch.randint(117, 129, (experts, 512, 128), device="cuda", dtype=torch.uint8)
    s2 = torch.randint(117, 129, (experts, 4096, 8), device="cuda", dtype=torch.uint8)
    x = (torch.randn(rows, hidden, device="cuda") * input_scale).to(torch.bfloat16)
    def routes():
        result = torch.stack([torch.randperm(experts, device="cuda")[:topk] for _ in range(rows)]).int()
        if rows > 1:
            result[-1].copy_(result[0])  # Repeated experts must keep distinct row scales.
        return result
    ids = routes()
    weights = torch.rand(rows, topk, device="cuda")
    weights *= 2.5 / weights.sum(dim=1, keepdim=True)
    config = FusedMoEConfig(num_experts=experts, experts_per_token=topk,
                           hidden_dim=hidden, intermediate_size=intermediate,
                           num_local_experts=experts, num_logical_experts=experts,
                           moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
                           activation=MoEActivation.SILU, in_dtype=x.dtype, device="cuda:0",
                           routing_method=RoutingMethodType.Renormalize)
    kernels = []
    for cls in (Upstream, Candidate):
        quant = ocp_mx_moe_quant_config(quant_dtype="mxfp4", weight_dtype="mxfp4",
                                      w1_scale=s1, w2_scale=s2, gemm1_clamp_limit=10.0)
        with set_current_vllm_config(VllmConfig()):
            kernel = make_mxfp4_moe_kernel(moe_quant_config=quant, moe_config=config,
                                         mxfp4_backend=Mxfp4MoeBackend.EMULATION,
                                         experts_cls=cls, routing_tables=None)
            kernels.append(kernel)

    def run(kernel):
        return kernel.apply(hidden_states=x, w1=w1, w2=w2, topk_weights=weights,
                            topk_ids=ids, activation=MoEActivation.SILU,
                            global_num_experts=experts, expert_map=None,
                            apply_router_weight_on_input=False)

    with set_current_vllm_config(VllmConfig()):
        if capture:
            # Compile and allocate workspace before capture, on a side stream.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    run(kernels[1])
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = run(kernels[1])

        for replay in range(3 if capture else 1):
            if replay:
                # Same addresses, different inputs AND routes: replay must not
                # retain the capture-time expert selection or activation data.
                x.copy_(torch.randn_like(x) * input_scale * (replay + 1))
                ids.copy_(routes())
                weights.copy_(torch.rand_like(weights))
                weights.mul_(2.5 / weights.sum(dim=1, keepdim=True))
            reference = run(kernels[0]).clone()
            if capture:
                graph.replay()
                actual = captured.clone()
            else:
                actual = run(kernels[1]).clone()
            assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
            torch.testing.assert_close(actual, reference, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("rows", [16, 32])
@pytest.mark.parametrize("input_scale", [1, 4, 32])
@pytest.mark.parametrize("capture", [False, True])
def test_packed_large_batch_reference(rows, input_scale, capture):
    # Full TP8 expert shapes, clamp=10 and per-group E8M0 scales. The helper
    # changes both inputs and expert routes across three graph replays.
    test_moe_matches_full_emulation(288, 800 + rows, input_scale, capture, rows, "packed")


@pytest.mark.parametrize("rows", [9, 17, 31])
def test_packed_partial_batch_reference(rows):
    test_moe_matches_full_emulation(288, 900 + rows, 4, True, rows, "packed")
