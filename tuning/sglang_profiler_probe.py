#!/usr/bin/env python3
"""Exclusive one-GPU Kineto capture check, never a performance measurement.

Triangulate direct execution, spawn before HIP initialization and spawn after
parent HIP initialization. The latter is a candidate cause from PyTorch issue
182373; this local reproduction must confirm it before attributing the fault.
"""
import argparse
from collections import Counter
import ctypes
import json
import importlib.util
import multiprocessing
import os
from pathlib import Path


def profiler_environment():
    # Whitelist profiler/device settings only; never dump authentication env.
    # C++ setenv() does not update Python's cached os.environ mapping.
    pointer=ctypes.POINTER(ctypes.c_char_p).in_dll(ctypes.CDLL(None), 'environ')
    result={};i=0
    while pointer[i] is not None:
        key,sep,value=pointer[i].decode(errors='replace').partition('=');i+=1
        if sep and (key.startswith(('KINETO_', 'ROCPROFILER_', 'ROCP_', 'ROCTRACER_',
                                    'HSA_', 'TORCH_PROFILER_'))
                    or key in ('HIP_VISIBLE_DEVICES','LD_PRELOAD','LD_LIBRARY_PATH')):
            result[key]=value
    return result


def cpu_ready():
    """Start the forkserver while it still has no imported HIP runtime."""
    pass


def worker(out):
    import torch
    assert torch.version.hip and torch.cuda.device_count() == 1
    assert torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] == 'gfx1201'
    (out/'worker-environment.json').write_text(json.dumps(profiler_environment(),indent=2)+'\n')
    x = torch.ones((256,256), device='cuda', dtype=torch.bfloat16)
    y = torch.empty_like(x)
    for _ in range(5):torch.mm(x,x,out=y)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=False,with_stack=False) as capture:
        for _ in range(32):torch.mm(x,x,out=y)
        torch.cuda.synchronize()
    assert bool((y == 256).all()), 'Matrix control failed independently of profiling'
    trace=out/'trace.json';capture.export_chrome_trace(str(trace))
    events=json.loads(trace.read_text())['traceEvents']
    categories=Counter(e.get('cat','unknown') for e in events)
    kernels=sum(categories[k] for k in ('kernel','hip_kernel'))
    result=dict(status='GPU capture present' if kernels else 'GPU capture MISSING',
                scope='trace fidelity only; no model speed or kernel timing claim',
                torch=torch.__version__,hip=torch.version.hip,
                matrix_control_exact=True,expected_matrix_calls=32,
                kernel_events=kernels,categories=dict(categories),
                gpu_time=None if not kernels else 'not summarized by this probe')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--mode',choices=['direct','spawn-clean','spawn-parent-init',
                                         'forkserver-early','forkserver-late',
                                         'spawn-parent-init-clean-hsa','spawn-parent-init-clean-both'],required=True)
    args=parser.parse_args();out=args.out
    out.mkdir(parents=True,exist_ok=False)
    (out/'before-parent.json').write_text(json.dumps(profiler_environment(),indent=2)+'\n')
    if args.mode == 'direct':
        worker(out)
        return
    import torch
    context=multiprocessing.get_context('forkserver' if args.mode.startswith('forkserver') else 'spawn')
    if args.mode == 'forkserver-early':
        ready=context.Process(target=cpu_ready);ready.start();ready.join()
        assert ready.exitcode == 0
    before=profiler_environment()
    if args.mode != 'spawn-clean':torch.cuda.init()
    after=profiler_environment()
    (out/'parent-init-environment.json').write_text(json.dumps(dict(before=before,after=after,
        changed={k:dict(before=before.get(k),after=after.get(k)) for k in before.keys()|after.keys()
                 if before.get(k)!=after.get(k)},cuda_initialized=torch.cuda.is_initialized()),indent=2)+'\n')
    child=context.Process(target=worker,args=(out,))
    if '-clean-' in args.mode:
        # Use the exact proposed startup helper, not a second implementation.
        spec=importlib.util.spec_from_file_location('child_env',Path(__file__).parents[1]/'r9700_sglang/profile_worker_env.py')
        child_env=importlib.util.module_from_spec(spec);spec.loader.exec_module(child_env)
        os.environ['R9K_SGLANG_PROFILE_CHILD_ENV']=args.mode.rsplit('-',1)[1]
        with child_env.profile_child_environment():
            during=profiler_environment();child.start()
        (out/'child-start-env.json').write_text(json.dumps(dict(during=during,after=profiler_environment()),indent=2)+'\n')
    else:
        child.start()
    (out/'child.json').write_text(json.dumps(dict(pid=child.pid,mode=args.mode))+'\n')
    child.join(timeout=120)
    if child.is_alive():
        (out/'observation-timeout.json').write_text(json.dumps(dict(pid=child.pid,status='still live; not restarted or killed'))+'\n')
        child.join()
    if child.exitcode != 0:raise RuntimeError(f'Profiler control child exited {child.exitcode}')


if __name__ == '__main__':main()
