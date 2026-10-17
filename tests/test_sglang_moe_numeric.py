"""Isolated gfx1201 AITER MXFP4 MoE arithmetic diagnostic.

Run inside the pinned SGLang image with one GPU. This uses synthetic weights;
it cannot qualify the full GLM checkpoint or end-to-end generation.
"""

import json
import os

import torch
import torch.nn.functional as F

from aiter.fused_moe import fused_moe
from aiter.ops.enum import QuantType
from aiter.utility.fp4_utils import e8m0_shuffle


VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    device="cuda", dtype=torch.float32,
)


def dequant(packed, scale):
    n, k2 = packed.shape
    lo = VALUES[(packed & 15).long()]
    hi = VALUES[(packed >> 4).long()]
    out = torch.empty((n, 2 * k2), device=packed.device)
    out[:, ::2], out[:, 1::2] = lo, hi
    return (out.reshape(n, -1, 32)
            * torch.exp2(scale.float() - 127).unsqueeze(-1)).reshape(n, -1)


def main():
    torch.manual_seed(20261015)
    m, hidden, inter, experts = 25, 4096, 256, 288
    x = torch.randn((m, hidden), device="cuda", dtype=torch.bfloat16) * 0.2
    w1 = torch.zeros((experts, 2 * inter, hidden // 2), device="cuda", dtype=torch.uint8)
    w2 = torch.zeros((experts, hidden, inter // 2), device="cuda", dtype=torch.uint8)
    w1[0].random_(0, 256)
    w2[0].random_(0, 256)
    w1[1:8] = w1[0]
    w2[1:8] = w2[0]
    s1 = torch.full((experts, 2 * inter, hidden // 32), 120,
                    device="cuda", dtype=torch.uint8)
    s2 = torch.full((experts, hidden, inter // 32), 120,
                    device="cuda", dtype=torch.uint8)

    dense1 = dequant(w1[0], s1[0]).bfloat16()
    dense2 = dequant(w2[0], s2[0]).bfloat16()
    gate = x @ dense1[:inter].T
    up = x @ dense1[inter:].T
    expected = (F.silu(gate.float()) * up.float()).bfloat16() @ dense2.T

    shuffled_s1 = e8m0_shuffle(s1.view(experts * 2 * inter, -1)).view(experts, 2 * inter, -1)
    shuffled_s2 = e8m0_shuffle(s2.view(experts * hidden, -1)).view(experts, hidden, -1)
    ids = torch.arange(8, device="cuda", dtype=torch.int64).repeat(m, 1)
    weights = torch.full((m, 8), 1 / 8, device="cuda", dtype=torch.float32)
    override = {}
    if os.getenv("MOE_A8") == "1":
        override = {"quant_dtype_a": torch.float8_e4m3fn,
                    "quant_dtype_a2": torch.float8_e4m3fn}
    if os.getenv("MOE_A16") == "1":
        override = {"quant_dtype_a": torch.bfloat16,
                    "quant_dtype_a2": torch.bfloat16}
    if os.getenv("MOE_GEMV") == "1":
        import importlib.util
        spec = importlib.util.spec_from_file_location("packed_gemv", "/tmp/packed_gemv.py")
        packed_gemv = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(packed_gemv)
        gu = packed_gemv.gemv(x, w1, s1, ids.flatten(), ids.shape[1])
        gate, up = gu.float().split(inter, dim=-1)
        mid = (F.silu(gate) * up).bfloat16().contiguous()
        down = packed_gemv.gemv(mid, w2, s2, ids.flatten(), 1,
                                weights.flatten())
        actual = down.view(m, 8, hidden).sum(dim=1)
    else:
        actual = fused_moe(
            x, w1.view(torch.float4_e2m1fn_x2),
            w2.view(torch.float4_e2m1fn_x2), weights, ids,
            quant_type=QuantType.per_1x32,
            w1_scale=shuffled_s1, w2_scale=shuffled_s2,
            **override,
        )
    torch.cuda.synchronize()
    e, a = expected.float(), actual.float()
    result = {
        "expected_norm": e.norm().item(),
        "actual_norm": a.norm().item(),
        "shuffled_s1_unique": shuffled_s1.unique().tolist(),
        "shuffled_s2_unique": shuffled_s2.unique().tolist(),
        "cosine": F.cosine_similarity(e.flatten(), a.flatten(), dim=0).item(),
        "max_abs_error": (e - a).abs().max().item(),
        "relative_l2": ((e - a).norm() / e.norm()).item(),
        "finite": bool(torch.isfinite(a).all()),
    }
    print(json.dumps(result, sort_keys=True))
    assert result["finite"]
    assert result["cosine"] > 0.8, result


if __name__ == "__main__":
    main()
