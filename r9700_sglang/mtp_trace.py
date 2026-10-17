# SPDX-License-Identifier: Apache-2.0
"""Opt-in, bounded layer snapshots for MTP correctness; never throughput.

Only HIP0, one requested logical position, at most 128 snapshots per module.
No tensor or model output is modified. CPU copies deliberately synchronize.
"""
import os
from pathlib import Path

import torch


def install(model):
    directory = os.environ.get("R9K_SGLANG_MTP_TRACE_DIR")
    if not directory or torch.cuda.current_device() != 0:
        return
    position = int(os.environ["R9K_SGLANG_MTP_TRACE_POSITION"])
    if position < 0:
        raise ValueError("Trace position must be nonnegative")
    out = Path(directory);out.mkdir(parents=True, exist_ok=True)
    counters = {}
    active = {}

    def layer_pre(module, args, kwargs):
        positions = kwargs.get("positions", args[0] if args else None)
        fb = kwargs.get("forward_batch", args[2] if len(args)>2 else None)
        active.clear()
        if positions is None or fb is None:
            return
        values = positions.detach().cpu().tolist()
        if position not in values:
            return
        row = values.index(position)
        active.update(row=row, positions=values, mode=str(fb.forward_mode), layer=module.layer_id)

    def save(name):
        def hook(module, args, kwargs, output):
            if not active:
                return
            count = counters.get(name, 0)
            if count >= 128:
                return
            row = active['row']
            values = output if isinstance(output,tuple) else (output,)
            tensors = {f'output_{i}':v[row].detach().cpu().clone()
                       for i,v in enumerate(values) if isinstance(v,torch.Tensor)
                       and v.ndim>=2 and v.shape[0]>row}
            inputs = kwargs.get('hidden_states',args[1] if len(args)>1 else None)
            if isinstance(inputs,torch.Tensor) and inputs.ndim>=2 and inputs.shape[0]>row:
                tensors['input']=inputs[row].detach().cpu().clone()
            torch.save(dict(**active,**tensors),out/f'{name}-{count:03d}.pt')
            counters[name]=count+1
        return hook

    for index,layer in enumerate(model.layers):
        layer.register_forward_pre_hook(layer_pre,with_kwargs=True)
        layer.self_attn.register_forward_hook(save(f'layer{index:02d}-attn'),with_kwargs=True)
        layer.register_forward_hook(save(f'layer{index:02d}-out'),with_kwargs=True)


def attention(q_nope,q_rope,kv,indices,result,layer,forward_batch):
    directory=os.environ.get('R9K_SGLANG_MTP_TRACE_DIR')
    if not directory or torch.cuda.current_device()!=0:
        return
    positions=forward_batch.positions.detach().cpu().tolist()
    position=int(os.environ['R9K_SGLANG_MTP_TRACE_POSITION'])
    if position not in positions or layer.layer_id!=3:
        return
    row=positions.index(position);root=Path(directory)
    files=list(root.glob('attention3-*.pt'));number=len(files)
    if number>=32:return
    selected=indices[row];selected=selected[selected>=0].long()
    data=dict(mode=str(forward_batch.forward_mode),positions=positions,row=row,
              q_nope=q_nope[row].detach().cpu(),q_rope=q_rope[row].detach().cpu(),
              physical_indices=selected.cpu(),selected_kv=kv[selected].detach().cpu(),
              result=result.reshape(q_nope.shape[0], q_nope.shape[1], layer.v_head_dim)[row].detach().cpu(),scale=layer.scaling,d_v=layer.v_head_dim,
              kv_shape=list(kv.shape))
    from sglang.srt.model_executor.forward_context import get_req_to_token_pool
    req_pool=get_req_to_token_pool()
    request=int(forward_batch.req_pool_indices[0].item())
    data['expected_prefix_physical']=req_pool.req_to_token[request,:position+1].detach().cpu()
    torch.save(data,root/f'attention3-{number:03d}.pt')
