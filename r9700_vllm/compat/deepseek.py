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


def offload_parameters(offloader, module, prefix, allocate, accelerator_view):
    """One exact host allocation and one copy per selected parameter.

    Allocator and view functions are explicit to test selection/accounting
    without GPU allocation. No pageable intermediate or global Torch patch.
    """
    import os
    import re
    first_layer = int(os.environ.get("R9K_DEEPSEEK_OFFLOAD_FIRST_LAYER", "0"))
    if not 0 <= first_layer < 40:
        raise ValueError("DeepSeek first offload layer must be in [0,39]")
    prefix = prefix.rstrip(".") + "." if prefix else ""
    matrices_only = os.environ.get("R9K_DEEPSEEK_OFFLOAD_MATRICES", "0") == "1"
    for name, parameter in module.named_parameters():
        if first_layer:
            match = re.search(r"(?:^|\.)layers\.(\d+)\.", prefix + name)
            if not match or not first_layer <= int(match[1]) < 40:
                continue
        if matrices_only and (name.rsplit(".", 1)[-1] not in ("w13_weight", "w2_weight")
                              or "experts" not in (prefix + name).split(".")):
            continue
        if offloader.cpu_offload_bytes >= offloader.cpu_offload_max_bytes:
            break
        if getattr(parameter, "_vllm_is_uva_offloaded", False):
            continue
        if offloader.cpu_offload_params and not any(
            f".{part}." in f".{prefix}{name}."
            for part in offloader.cpu_offload_params
        ):
            continue
        size = parameter.numel() * parameter.element_size()
        if not size:
            continue
        storage = allocate(size)
        if storage is None:
            raise RuntimeError("DeepSeek expert registration failed; refusing rounded fallback")
        host = storage.view(parameter.dtype).view(parameter.shape)
        host.copy_(parameter.detach())
        parameter.data = accelerator_view(host)
        # Keep a CPU alias of the SAME allocation for checkpoint loading.
        # copy_ through the accelerator alias otherwise invokes HIP even for
        # a destination that physically lives in host RAM.
        parameter._r9700_host_view = host
        parameter._vllm_is_uva_offloaded = True
        parameter._r9700_host_bytes = size
        offloader.cpu_offload_bytes += size


def install_expert_pinning():
    """DeepSeek-only single-copy exact allocation; keep upstream forward path."""
    from functools import wraps
    from vllm.model_executor.offloader.uva import UVAOffloader
    from vllm.models.deepseek_v41.common import engram
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    original = UVAOffloader._maybe_offload_to_cpu
    if getattr(original, "_r9700_deepseek_exact", False) is True:
        return

    @wraps(original)
    def offload(self, module, prefix=""):
        if not _is_deepseek():
            return original(self, module, prefix)
        if not self.pin_memory or not self.uva_offloading:
            raise RuntimeError("DeepSeek expert offload requires pinned UVA storage")
        first = next(module.parameters(), None)
        if first is None or first.device.type == "cpu":
            return module
        offload_parameters(self, module, prefix, engram._allocate_huge_page_storage,
                           get_accelerator_view_from_cpu_tensor)
        return module

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
    from .deepseek_dspark import install as install_dspark
    install_dspark()
    install_engram_guard()
    install_expert_pinning()
    from .deepseek_loading import install_optimized_loading
    install_optimized_loading()
    from .deepseek_indexer import install
    install()
    from .deepseek_loading import install as install_loading
    install_loading()
    from .deepseek_moe import install as install_moe
    install_moe()

    from ..moe.deepseek_expert_profile import install as install_profile
    install_profile()

    from ..moe.deepseek_placement import install as install_placement
    install_placement()
