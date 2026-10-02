"""Opt-in GLM head projection for 1–8 rows using the FP32-output router GEMV.

The upstream operands are BF16 tensors promoted exactly to FP32. Reuse those
original BF16 values with the router's FP32 FMA/reduction/output path; never
round the projection output to BF16. Only reduction order can differ.
"""
from functools import wraps
from types import FunctionType


class _HeadTorch:
    def __init__(self, original, indexer, hidden, weight, gemv):
        self.original, self.indexer = original, indexer
        self.hidden, self.weight, self.gemv = hidden, weight, gemv

    def __getattr__(self, name):
        return getattr(self.original, name)

    def mm(self, x, w, *args, **kwargs):
        if w is self.indexer._wp_fp32 and not args and not kwargs:
            return self.gemv(self.hidden, self.weight, out_bf16=False, split=8)
        return self.original.mm(x, w, *args, **kwargs)


def _wrap_forward(original, gemv):
    @wraps(original)
    def forward(self, hidden_states, *args, **kwargs):
        import torch
        weight = self.wk_weights_proj.weight
        if (hidden_states.ndim != 2 or hidden_states.shape[1] != 4096
                or not 1 <= hidden_states.shape[0] <= 8 or hidden_states.dtype != torch.bfloat16
                or weight.dtype != torch.bfloat16 or weight.shape != (160, 4096)
                or self.head_dim != 128 or not hidden_states.is_contiguous()
                or not weight.is_contiguous()):
            return original(self, hidden_states, *args, **kwargs)
        # Give this one upstream function a private torch binding. Keep its
        # code, globals and fallback untouched; no process-wide torch patch,
        # no duplicated forward, and no retained mutable request state.
        local_torch = _HeadTorch(torch, self, hidden_states, weight[128:], gemv)
        adapted = FunctionType(original.__code__, original.__globals__ | {"torch": local_torch},
                               original.__name__, original.__defaults__, original.__closure__)
        adapted.__kwdefaults__ = original.__kwdefaults__
        return adapted(self, hidden_states, *args, **kwargs)
    forward._r9700_glm_head_gemv = True
    return forward


def patch() -> bool:
    import os
    if os.environ.get("R9K_GLM_INDEXER_GEMV", "0") == "0":
        return False
    if os.environ["R9K_GLM_INDEXER_GEMV"] != "1":
        raise ValueError("R9K_GLM_INDEXER_GEMV must be 0 or 1")
    import inspect
    import torch
    from vllm.platforms import current_platform
    from .gate import check, vllm_commit
    if (not current_platform.is_rocm()
            or torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0] != "gfx1201"
            or not vllm_commit().startswith("e97573215")):
        raise RuntimeError("GLM head GEMV requires gfx1201 and audited vLLM e97573215")
    from vllm.models.glm5next.common.attention import Indexer
    if getattr(Indexer.forward, "_r9700_glm_head_gemv", False):
        return True
    if inspect.getsource(Indexer.forward).count("torch.mm(hidden_states.float(), self._wp_fp32)") != 1:
        raise RuntimeError("GLM head projection changed; revalidate the adapter")
    from r9700_vllm.router import available, router_gemm
    if not available():
        raise RuntimeError("GLM head GEMV requires the existing r9k_router_gemm kernel")
    check("glm_head_gemv")
    Indexer.forward = _wrap_forward(Indexer.forward, router_gemm)
    from vllm.logger import init_logger
    init_logger("vllm.r9700_vllm").info("r9700: GLM indexer head projection uses FP32-output router GEMV for 1–8 rows")
    return True
