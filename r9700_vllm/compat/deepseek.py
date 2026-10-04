"""Pinned DeepSeek gates and model-scoped exact host allocation."""


def _is_deepseek():
    from vllm.config import get_current_vllm_config

    cfg = get_current_vllm_config()
    hf = getattr(cfg.model_config, "hf_config", None)
    return getattr(hf, "model_type", None) in ("deepseek_v41", "deepseek_v41_text")


def install_engram_guard():
    from functools import wraps
    from vllm.models.deepseek_v41.common import engram

    original = engram._allocate_huge_page_storage
    if getattr(original, "_r9700_deepseek_exact", False) is True:
        return

    @wraps(original)
    def allocate(num_bytes):
        result = original(num_bytes)
        if result is None and _is_deepseek():
            # The ordinary pinned fallback can round Engram from 189 to 264 GiB.
            # Exact size is required; huge-page coverage itself remains optional.
            raise RuntimeError("DeepSeek exact Engram registration failed; refusing rounded pinned fallback")
        return result

    allocate._r9700_deepseek_exact = True
    engram._allocate_huge_page_storage = allocate


def install_expert_pinning():
    """Use the same upstream exact allocator during DeepSeek UVA offloading.

    Like the stack's exact_pinning construction scope, this temporarily replaces
    Tensor.pin_memory during single-threaded worker initialization. It does not
    change expert placement, copies, forward execution or other models.
    """
    from functools import wraps
    import torch
    from vllm.model_executor.offloader.uva import UVAOffloader
    from vllm.models.deepseek_v41.common import engram

    original = UVAOffloader._maybe_offload_to_cpu
    if getattr(original, "_r9700_deepseek_exact", False) is True:
        return

    @wraps(original)
    def offload(self, module, prefix=""):
        if not _is_deepseek():
            return original(self, module, prefix)
        if not self.pin_memory or not self.uva_offloading:
            raise RuntimeError("DeepSeek expert offload requires pinned UVA storage")
        stock = torch.Tensor.pin_memory

        def pin(tensor, *args, **kwargs):
            if tensor.device.type != "cpu":
                return stock(tensor, *args, **kwargs)
            if tensor.numel() == 0:
                return tensor
            packed = engram._allocate_huge_page_storage(tensor.numel() * tensor.element_size())
            if packed is None:
                raise RuntimeError("DeepSeek expert registration failed; refusing rounded pinned fallback")
            # Views retain the upstream allocator's registration and mmap owner.
            result = packed.view(tensor.dtype).view(tensor.shape)
            result.copy_(tensor)
            return result

        torch.Tensor.pin_memory = pin
        try:
            return original(self, module, prefix)
        finally:
            torch.Tensor.pin_memory = stock

    offload._r9700_deepseek_exact = True
    UVAOffloader._maybe_offload_to_cpu = offload


def register():
    import torch
    from vllm.platforms import current_platform
    from .gate import vllm_commit

    if not current_platform.is_rocm() or not vllm_commit().startswith("18f8f960"):
        raise RuntimeError("DeepSeek adapter requires ROCm vLLM 18f8f960")
    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0]
    if arch != "gfx1201":
        raise RuntimeError("DeepSeek adapter is scoped to gfx1201")
    from ..quant import deepseek  # noqa: F401
    install_engram_guard()
    install_expert_pinning()
