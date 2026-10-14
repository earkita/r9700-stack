"""GLM Quark dynamic group128 FP8: epsilon matches pinned vLLM (1e-10)."""
import ctypes
import torch
from . import fp8


def bind():
    lib = fp8._L()
    name = "r9k_glm_gemm_fp8_block_qa"
    if not hasattr(lib, name):
        raise RuntimeError("Rebuild kernels: missing GLM Quark dense FP8 entry")
    f = getattr(lib, name)
    f.restype = ctypes.c_int
    f.argtypes = [ctypes.c_long, ctypes.c_int] + [ctypes.c_long]*3 + [ctypes.c_int]*7 + [ctypes.c_long]
    return f


def gemm(x, packed, scales, n, k):
    m = x.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    wv, sk, npw = fp8.pick_cfg("fp8block", n, k, m)
    while sk > 1 and k % (128*sk):
        sk //= 2
    rc = bind()(x.data_ptr(), x.stride(0), packed.data_ptr(), scales.data_ptr(), out.data_ptr(),
                m, k, n, out.stride(0), wv, sk, npw, torch.cuda.current_stream().cuda_stream)
    if rc:
        raise RuntimeError(f"GLM dense FP8 failed ({rc}), shape={tuple(x.shape)}")
    return out
