"""Bounded GLM dense FP8 gate; run explicitly on an idle, PCIe-audited GPU.

Raw timings are kernel diagnostics, not model decode throughput.
"""
import json
import torch
from r9700_vllm.kernels import fp8 as F8
from r9700_vllm.kernels.glm_fp8 import gemm
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8, w8a8_triton_block_scaled_mm,
)


def relative(a, b):
    return ((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-30)).item()


def timing(fn):
    for _ in range(5):
        fn()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    times = []
    for _ in range(3):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(100):
            graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b)*10)
    return times


def main():
    torch.manual_seed(1234)
    failures = 0
    for n, k in ((3072, 4096), (4096, 1536)):
        w = torch.randn(n, k+256, device="cuda").to(torch.float8_e4m3fn)[:, :k]
        packed = F8.permute_fp8(w)
        scales = torch.rand(n//128, k//128, device="cuda")*.02+.001
        for m in (1, 5, 10, 15, 16, 17, 20, 31, 32):
            # Nontrivial leading stride exercises row offsets in the second tile.
            x = torch.randn(m, k+8, device="cuda", dtype=torch.bfloat16)[:, :k]
            cfg = F8.pick_cfg("fp8block", n, k, m)
            def stock():
                q, s = per_token_group_quant_fp8(x.contiguous(), 128, use_ue8m0=False)
                return w8a8_triton_block_scaled_mm(q, w, s, scales, [128,128], torch.bfloat16)
            def fused():
                return gemm(x, packed, scales, n, k)
            q, s = F8.quant_group128_fp8(x)
            ref = F8.gemm_fp8_block(q, s, packed, scales, n, k, None, *cfg)
            y = fused()
            stock_error, hip_error = relative(y, stock()), relative(y, ref)
            ok = bool(torch.isfinite(y).all()) and stock_error < .01 and hip_error < 1e-5
            # Capture/replay with changed inputs, including the partial second tile.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                fused()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                yg = fused()
            x.mul_(.5)
            graph.replay()
            ok = ok and torch.equal(yg, fused())
            # Quark epsilon and zero-input behavior are part of the contract.
            for factor in (0., 1e-12):
                tiny = x * factor
                tq, ts = per_token_group_quant_fp8(tiny, 128, use_ue8m0=False)
                tr = w8a8_triton_block_scaled_mm(tq, w, ts, scales, [128,128], torch.bfloat16)
                ty = gemm(tiny, packed, scales, n, k)
                ok = ok and bool(torch.isfinite(ty).all()) and relative(ty, tr) < .01
            x = x.contiguous()  # Timed paths use the same layout without an extra copy.
            result = dict(n=n, k=k, m=m, stock_relative=stock_error,
                          unfused_relative=hip_error, passed=bool(ok),
                          stock_us=timing(stock), fused_us=timing(fused))
            print(json.dumps(result), flush=True)
            failures += not ok
    if failures:
        raise SystemExit(f"{failures} failed cases")


if __name__ == "__main__":
    main()
