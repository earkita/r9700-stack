"""Bounded host staging, active only inside DeepSeek checkpoint loading.

Stage the already-sharded source of each CPU -> HIP copy. Do not clone a
checkpoint tensor (Engram files can exceed host headroom), patch Tensor methods,
or retain checkpoint storage. CPU -> CPU offload copies remain unchanged.
"""
import torch
from torch.utils._python_dispatch import TorchDispatchMode


def copy_tiles(destination, source, max_elements):
    """Yield matching views, including column shards and broadcast sources."""
    if max_elements < 1:
        raise ValueError("The load buffer must hold at least one source element")
    source = source.expand(destination.shape)
    if destination.numel() <= max_elements:
        yield destination, source
        return
    axis = max(range(destination.ndim), key=lambda i: destination.shape[i])
    plane = destination.numel() // destination.shape[axis]
    step = max(1, max_elements // plane)
    for start in range(0, destination.shape[axis], step):
        size = min(step, destination.shape[axis] - start)
        yield from copy_tiles(destination.narrow(axis, start, size),
                              source.narrow(axis, start, size), max_elements)


class BufferedLoadCopies(TorchDispatchMode):
    """Thread-local synchronous staging; buffer is released even on failure."""

    def __init__(self, buffer_bytes):
        super().__init__()
        if buffer_bytes < 16 or buffer_bytes % 16:
            raise ValueError("Load buffer bytes must be a positive multiple of 16")
        self.buffer_bytes = buffer_bytes
        self.buffer = None
        self.copies = self.tiles = self.staged_bytes = self.peak_tile_bytes = 0

    def __exit__(self, *args):
        try:
            return super().__exit__(*args)
        finally:
            self.buffer = None

    def _copy(self, destination, source):
        if source.layout != torch.strided or destination.layout != torch.strided:
            raise RuntimeError("DeepSeek buffered loading requires strided tensors")
        if self.buffer is None:
            # Ordinary anonymous RAM avoids file-backed writable mmap transfers.
            # One allocation per load, reused for all dtypes and TP slices.
            self.buffer = torch.empty(self.buffer_bytes, dtype=torch.uint8,
                                      device="cpu", pin_memory=False)
        capacity = self.buffer_bytes // source.element_size()
        for target, value in copy_tiles(destination, source, capacity):
            size = value.numel() * value.element_size()
            staging = self.buffer[:size].view(value.dtype).view(value.shape)
            staging.copy_(value)
            # Explicitly synchronous: the next tile must not overwrite a source
            # buffer still being read by HIP, even if upstream requested async.
            target.copy_(staging, non_blocking=False)
            self.tiles += 1
            self.staged_bytes += size
            self.peak_tile_bytes = max(self.peak_tile_bytes, size)
        self.copies += 1
        return destination

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func == torch.ops.aten.copy_.default:
            destination, source = args[:2]
            if destination.device.type == "cuda" and source.device.type == "cpu":
                return self._copy(destination, source)
        elif func == torch.ops.aten._to_copy.default:
            source = args[0]
            device = kwargs.get("device")
            if (source.device.type == "cpu" and device is not None
                    and torch.device(device).type == "cuda"):
                options = {k: v for k, v in kwargs.items() if k != "non_blocking"}
                destination = torch.empty_like(source, **options)
                return self._copy(destination, source)
        return func(*args, **kwargs)
