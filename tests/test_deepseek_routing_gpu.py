"""Live graph padding must not bias the hot/cold calibration counters."""
import os
import pytest
if os.environ.get('R9700_DEEPSEEK_GPU_TEST') != '1':
    pytest.skip('Explicit GPU test opt-in required',allow_module_level=True)
import torch
from r9700_vllm.moe.deepseek_expert_profile import count_routes


def test_route_histogram_replay_uses_live_padding_and_ids():
    ids=torch.zeros(24,6,device='cuda',dtype=torch.int32)
    valid=torch.ones(24,device='cuda',dtype=torch.bool)
    bank=torch.zeros(385,device='cuda',dtype=torch.int64)
    def call():
        counts,calls=count_routes(ids,valid)
        bank[:384].add_(counts);bank[384].add_(calls)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):call()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):call()
    bank.zero_()
    expected=torch.zeros(385,dtype=torch.int64)
    for n in (1,6,19,0,24):
        ids.copy_(torch.randint(384,ids.shape,device='cuda',dtype=torch.int32))
        valid.copy_(torch.arange(24,device='cuda')<n)
        graph.replay()
        expected[:384].add_(torch.bincount(ids[:n].cpu().long().flatten(),minlength=384))
        expected[384]+=int(n>0)
        torch.testing.assert_close(bank.cpu(),expected,atol=0,rtol=0)
