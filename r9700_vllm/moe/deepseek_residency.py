"""HIP VMM transport for unchanged, partially host-resident expert tensors."""
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def extension():
    import torch
    from torch.utils.cpp_extension import load, ROCM_HOME
    if not torch.version.hip or ROCM_HOME is None:
        raise RuntimeError("DeepSeek partial residency requires HIP")
    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0]
    if arch != "gfx1201":
        raise RuntimeError("DeepSeek partial residency is qualified only for gfx1201")
    root = Path(__file__).resolve().parents[2]
    cache = root / ".runtime/cache/deepseek-residency"
    cache.mkdir(parents=True, exist_ok=True)
    # CPU-only extension calling HIP; no GPU code and no NVIDIA dependency.
    return load(name="r9700_deepseek_residency",
                sources=[str(Path(__file__).with_name("csrc") / "deepseek_residency.cpp")],
                build_directory=str(cache), with_cuda=False,
                extra_include_paths=[str(Path(ROCM_HOME) / "include")],
                extra_cflags=["-O2", "-D__HIP_PLATFORM_AMD__"],
                extra_ldflags=[f"-L{ROCM_HOME}/lib", "-lamdhip64"], verbose=False)


def make_partially_resident(source, page_on_device):
    return extension().make_partially_resident(source, page_on_device, 0)
