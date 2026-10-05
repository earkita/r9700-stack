"""Staging eligibility, rollback and exact offload selection without GPUs."""
import os
from types import SimpleNamespace
from unittest.mock import patch
import pytest

torch = pytest.importorskip("torch")
from r9700_vllm.moe.deepseek_staging import staged_projection, stage_threshold
from r9700_vllm.compat.deepseek import offload_parameters


@pytest.fixture(autouse=True)
def threshold():
    with patch.dict(os.environ, R9K_DEEPSEEK_STAGE_MIN_TOKENS="256",R9K_DEEPSEEK_STAGE_BUFFER="0",R9K_DEEPSEEK_OFFLOAD_FIRST_LAYER="0"):
        stage_threshold.cache_clear()
        yield
        stage_threshold.cache_clear()


@pytest.mark.parametrize("rows,capture,copy", [(255,False,False),(256,False,True),(512,True,False)])
def test_exact_staging_and_capture_bypass(rows,capture,copy):
    weight=torch.arange(64,dtype=torch.uint8)
    weight._vllm_is_uva_offloaded=True
    scales=torch.ones(2,dtype=torch.uint8)
    with patch.object(torch.cuda,"is_current_stream_capturing",return_value=capture):
        with staged_projection(weight,scales,rows) as (w,s):
            assert torch.equal(w,weight)
            assert (w.data_ptr()!=weight.data_ptr()) == copy
            assert s is scales


def test_partial_allocation_failure_restores_both_originals():
    w=torch.ones(32,dtype=torch.uint8);s=torch.ones(1,dtype=torch.uint8)
    w._vllm_is_uva_offloaded=s._vllm_is_uva_offloaded=True
    clone=torch.Tensor.clone
    def fail_second(self,**kwargs):
        if self is s:raise torch.OutOfMemoryError("test allocation failure")
        return clone(self,**kwargs)
    with patch.object(torch.cuda,"is_current_stream_capturing",return_value=False), \
         patch.object(torch.Tensor,"clone",fail_second):
        with staged_projection(w,s,256) as (a,b):
            assert a is w and b is s


def test_gemm_errors_are_not_swallowed_as_allocation_fallback():
    w=torch.ones(32,dtype=torch.uint8);s=torch.ones(1,dtype=torch.uint8)
    w._vllm_is_uva_offloaded=True
    with patch.object(torch.cuda,"is_current_stream_capturing",return_value=False):
        with pytest.raises(torch.OutOfMemoryError,match="GEMM"):
            with staged_projection(w,s,256):
                raise torch.OutOfMemoryError("GEMM failed")


def test_only_routed_matrices_offloaded():
    module=torch.nn.Module()
    for group in ("experts","shared_experts"):
        bank=torch.nn.Module()
        for name in ("w13_weight","w2_weight","w13_weight_scale","w2_weight_scale"):
            bank.register_parameter(name,torch.nn.Parameter(torch.ones(32,dtype=torch.uint8),requires_grad=False))
        setattr(module,group,bank)
    budget=SimpleNamespace(cpu_offload_params={"experts"},cpu_offload_bytes=0,cpu_offload_max_bytes=1000)
    with patch.dict(os.environ,R9K_DEEPSEEK_OFFLOAD_MATRICES="1"):
        offload_parameters(budget,module,"model.layers.20.ffn",
            lambda n:torch.empty(n,dtype=torch.uint8),lambda x:x)
    assert budget.cpu_offload_bytes==64
    for name,p in module.named_parameters():
        expected=name in ("experts.w13_weight","experts.w2_weight")
        assert bool(getattr(p,"_vllm_is_uva_offloaded",False))==expected


@pytest.mark.parametrize('index,expected',[(19,False),(20,True),(39,True),(40,False)])
def test_reference_layer_range_is_explicit(index,expected):
    module=torch.nn.Module();module.experts=torch.nn.Module()
    module.experts.w13_weight=torch.nn.Parameter(torch.ones(8,dtype=torch.uint8),requires_grad=False)
    budget=SimpleNamespace(cpu_offload_params={'experts'},cpu_offload_bytes=0,cpu_offload_max_bytes=64)
    with patch.dict(os.environ,R9K_DEEPSEEK_OFFLOAD_MATRICES='1',R9K_DEEPSEEK_OFFLOAD_FIRST_LAYER='20'):
        offload_parameters(budget,module,f'layers.{index}.ffn',lambda n:torch.empty(n,dtype=torch.uint8),lambda x:x)
    assert bool(budget.cpu_offload_bytes)==expected
