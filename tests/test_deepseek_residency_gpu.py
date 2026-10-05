"""Small, explicit GPU tests of the mixed-memory ownership contract."""
import os
import gc
import pytest

if os.environ.get("R9700_DEEPSEEK_GPU_TEST") != "1":
    pytest.skip("Explicit GPU test opt-in required", allow_module_level=True)
import torch
from r9700_vllm.moe.deepseek_residency import extension, make_partially_resident


@pytest.mark.parametrize("host_first", [False, True])
def test_mixed_bytes_views_and_device_reader(host_first):
    grain=extension().page_size(0,0)
    source=(torch.arange(3*grain+32,device='cuda',dtype=torch.int32)%251).to(torch.uint8)
    flags=torch.tensor([not host_first,host_first,not host_first,host_first])
    mixed=make_partially_resident(source,flags)
    torch.testing.assert_close(mixed,source,rtol=0,atol=0)
    view=mixed[grain-16:grain+16]
    del mixed
    gc.collect()
    torch.testing.assert_close(view,source[grain-16:grain+16],rtol=0,atol=0)
    torch.testing.assert_close(view.to(torch.int32)*3,source[grain-16:grain+16].to(torch.int32)*3)


def test_invalid_page_plan_is_rejected():
    source=torch.ones(4096,device='cuda',dtype=torch.uint8)
    with pytest.raises(RuntimeError,match="entries"):
        make_partially_resident(source,torch.ones(7,dtype=torch.bool))
    with pytest.raises(RuntimeError,match="CPU bool"):
        make_partially_resident(source,torch.ones(1,dtype=torch.int32))
