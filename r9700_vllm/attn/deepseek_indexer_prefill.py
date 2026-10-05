"""DeepSeek-only fused prefill index logits on the pinned gfx1201 runtime.

Call AITER's portable Triton kernel directly, without enabling its global
dispatch. It reduces the heads in registers and only allocates [M, padded N]
FP32 logits; the PyTorch reference materializes several [H, M, N] tensors.
The caller retains upstream key gathering, chunking, masks and top-k.
"""
import inspect

import torch


def check_kernel_interface():
    from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits

    if list(inspect.signature(fp8_mqa_logits).parameters) != [
        "Q", "KV", "kv_scales", "weights", "cu_starts", "cu_ends", "clean_logits"
    ]:
        raise RuntimeError("DeepSeek AITER prefill interface changed; revalidate adapter")


def prefill_logits(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke):
    from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits

    k, scales = kv
    if (q.ndim != 3 or q.shape[1:] != (32, 128)
            or k.ndim != 2 or k.shape[1] != 128
            or q.dtype != torch.float8_e4m3fn or k.dtype != q.dtype
            or weights.shape != q.shape[:2] or weights.dtype != torch.float32
            or scales.shape not in ((k.shape[0],), (k.shape[0], 1))
            or scales.dtype != torch.float32 or not scales.is_contiguous()
            or any(x.shape != (q.shape[0],) or x.dtype != torch.int32
                   or not x.is_contiguous() for x in (cu_seqlen_ks, cu_seqlen_ke))
            or q.device.type != "cuda"
            or any(x.device != q.device for x in
                   (k, scales, weights, cu_seqlen_ks, cu_seqlen_ke))):
        raise ValueError("Unsupported DeepSeek fused prefill indexer inputs")
    # clean_logits initializes masked positions to -inf. Returning an unmasked
    # allocation would let unrelated sequence keys enter the downstream top-k.
    return fp8_mqa_logits(q, k, scales, weights, cu_seqlen_ks, cu_seqlen_ke,
                         clean_logits=True)
