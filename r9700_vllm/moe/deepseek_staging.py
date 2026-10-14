"""Exact whole-projection staging for DeepSeek's packed UVA experts.

Strategy follows devin-lai/DeepSeek-V4.1-Flash-Accel staging.py at 4a4e88e:
large eager batches copy immutable packed weights, small batches retain UVA.
This implementation also handles independently offloaded Quark scale tensors.
"""
from contextlib import contextmanager
from functools import lru_cache
import logging
import os

import torch

log = logging.getLogger("vllm.r9700_vllm.deepseek_staging")
_reported = set()


@lru_cache(maxsize=1)
def stage_threshold():
    value = int(os.environ.get("R9K_DEEPSEEK_STAGE_MIN_TOKENS", "0"))
    if value < 0:
        raise ValueError("R9K_DEEPSEEK_STAGE_MIN_TOKENS must be nonnegative")
    return value


def host_backed(tensor):
    return bool(getattr(tensor, "_vllm_is_uva_offloaded", False)
                and getattr(tensor, "_r9700_host_bytes", 1) > 0)


def report_once(kind, weight, tokens):
    key = (kind, weight.device)
    if key not in _reported:
        _reported.add(key)
        log.warning("DeepSeek whole-projection staging %s: device=%s bytes=%d query_tokens=%d",
                    kind, weight.device, weight.nbytes, tokens)


@contextmanager
def staged_projection(weight, scales, query_tokens):
    threshold = stage_threshold()
    pair = (weight, scales)
    staged = []
    if (threshold and query_tokens >= threshold and any(host_backed(t) for t in pair)
            and not torch.cuda.is_current_stream_capturing()):
        if os.environ.get("R9K_DEEPSEEK_STAGE_BUFFER", "0") == "1":
            with reserved_projection(weight, scales, query_tokens) as result:
                yield result
            return
        try:
            for tensor in pair:
                staged.append(tensor.clone(memory_format=torch.contiguous_format)
                              if host_backed(tensor) else tensor)
        except torch.OutOfMemoryError:
            staged.clear()
            report_once("allocation fallback to packed UVA", weight, query_tokens)
        else:
            pair = tuple(staged)
            report_once("active", weight, query_tokens)
    try:
        yield pair
    finally:
        # Lifetime ends on the same stream as GEMM; no persistent full copy.
        staged.clear()


class ProjectionBuffer:
    """One persistent GPU staging arena, visible to vLLM's memory profiling.

    A stream event protects reuse if eager calls switch streams. No host copy
    is retained. Allocation failure happens before touching the caller's data.
    """
    def __init__(self):
        import threading
        self.lock = threading.Lock()
        self.storage = None
        self.finished = None

    @contextmanager
    def stage(self, pair, query_tokens):
        with self.lock:
            stream = torch.cuda.current_stream(pair[0].device)
            if self.finished is not None:
                stream.wait_event(self.finished)
            needed = sum(t.nbytes for t in pair if host_backed(t))
            try:
                if self.storage is None or self.storage.numel() < needed:
                    replacement = torch.empty(needed, device=pair[0].device, dtype=torch.uint8)
                    self.storage = replacement
            except torch.OutOfMemoryError:
                report_once('buffer allocation fallback to packed UVA', pair[0], query_tokens)
                yield pair
                return
            offset = 0
            staged = []
            for tensor in pair:
                if host_backed(tensor):
                    view = self.storage[offset:offset+tensor.nbytes].view(tensor.dtype).view(tensor.shape)
                    view.copy_(tensor)
                    staged.append(view)
                    offset += tensor.nbytes
                else:
                    staged.append(tensor)
            report_once('reserved buffer active', pair[0], query_tokens)
            try:
                yield tuple(staged)
            finally:
                if self.finished is None:
                    self.finished = torch.cuda.Event()
                self.finished.record(stream)


_buffers = {}


def reserved_projection(weight, scales, query_tokens):
    key = weight.device
    if key not in _buffers:
        _buffers[key] = ProjectionBuffer()
    return _buffers[key].stage((weight, scales), query_tokens)
