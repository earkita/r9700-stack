import pytest


@pytest.mark.parametrize("shape", [(1, 4096), (2, 4096), (1, 0)])
def test_rejects_cpu_and_wrong_shape(shape):
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    from r9700_sglang.bf16_dense import linear_one_row
    x = torch.zeros(shape, dtype=torch.bfloat16)
    w = torch.zeros((128, 4096), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="one-row BF16 CUDA"):
        linear_one_row(x, w)
