"""Immutable expert-page planning for DeepSeek Quark EP8.

Page ranking follows DeepSeek-V4.1-Flash-Accel (4a4e88e): normalize each
calibration corpus, take worst-case hotness across corpora, place coldest
pages on host. The runtime must supply a matching checkpoint/layout identity.
This module never invents a uniform routing profile.
"""
import json
import math
from pathlib import Path

MATRICES = ("w13_weight", "w2_weight")


def page_weights(hotness, nbytes, page):
    """Byte-weighted mean expert hotness, including pages straddling experts."""
    if not hotness or nbytes <= 0 or page <= 0 or nbytes % len(hotness):
        raise ValueError("Invalid expert/page geometry")
    stride = nbytes // len(hotness)
    result = []
    for start in range(0, nbytes, page):
        end = min(start + page, nbytes)
        total = 0.0
        for expert in range(start // stride, (end - 1) // stride + 1):
            overlap = min(end, (expert + 1) * stride) - max(start, expert * stride)
            total += hotness[expert] * overlap
        result.append(total / (end - start))
    return result


def read_profiles(paths, identity, phase="decode", counter="calls_hit"):
    if phase not in ("prefill", "decode", "all") or counter not in ("slots", "calls_hit"):
        raise ValueError("Unsupported routing counter")
    combined = {}
    for path in paths:
        data = json.loads(Path(path).read_text())
        if data.get("schema") != 1 or data.get("identity") != identity:
            raise ValueError(f"Routing profile identity mismatch: {path}")
        if data.get("all_requests_completed") is not True:
            raise ValueError("Calibration requests did not all complete")
        rows = data.get("rows", [])
        if sorted(row["layer"] for row in rows) != list(range(40)):
            raise ValueError("Routing calibration must cover every backbone layer exactly once")
        corpus = {}
        for row in rows:
            counters = row[phase]
            calls = counters["calls"]
            values = counters[counter]
            if calls <= 0 or len(values) != 384 or any(v < 0 or not math.isfinite(v) for v in values):
                raise ValueError("Missing or invalid routing observations")
            corpus[row["layer"]] = [v / calls for v in values]
        mean = sum(map(sum, corpus.values())) / (40 * 384)
        if not mean:
            raise ValueError("Empty routing calibration")
        for layer, values in corpus.items():
            normalized = [v / mean for v in values]
            combined[layer] = [max(a, b) for a, b in zip(combined[layer], normalized)] if layer in combined else normalized
    if not combined:
        raise ValueError("At least one measured routing profile is required")
    return combined


def build_plan(hotness, expert_map, geometry, page_bytes, budget_bytes):
    """Return per-layer/matrix host-page sets; never exceed rounded budget."""
    if page_bytes <= 0 or budget_bytes < 0 or sorted(hotness) != list(range(40)):
        raise ValueError("Invalid page budget or missing layers")
    if set(geometry) != set(MATRICES) or len(expert_map) != 384:
        raise ValueError("Expected DeepSeek EP8 matrices and global expert map")
    owned = sorted(v for v in expert_map if v >= 0)
    if owned != list(range(48)):
        raise ValueError("EP8 must own exactly 48 distinct local experts")
    candidates = []
    for layer, values in sorted(hotness.items()):
        local = [0.] * 48
        if len(values) != 384 or any(v < 0 or not math.isfinite(v) for v in values):
            raise ValueError("Invalid expert hotness")
        for global_id, local_id in enumerate(expert_map):
            if local_id >= 0:
                local[local_id] = values[global_id]
        for name, nbytes in geometry.items():
            candidates.extend((value, layer, name, i) for i, value in enumerate(page_weights(local, nbytes, page_bytes)))
    candidates.sort()
    count = budget_bytes // page_bytes
    if count > len(candidates):
        raise ValueError("Offload budget exceeds expert storage")
    plan = {(layer, name): set() for layer in hotness for name in MATRICES}
    for _, layer, name, index in candidates[:count]:
        plan[layer, name].add(index)
    return plan


def migration_order(jobs):
    """Balance cumulative RAM/VRAM change instead of filling either pool first.

    Job[0] is new GPU bytes minus old GPU bytes. With equal total host
    budgets, the cumulative change stays within one projection; the copy
    itself needs at most one additional projection of transient headroom.
    """
    pending = list(jobs)
    balance = 0
    ordered = []
    while pending:
        index = min(range(len(pending)),
                    key=lambda i: (abs(balance + pending[i][0]), pending[i][1:3]))
        job = pending.pop(index)
        balance += job[0]
        ordered.append(job)
    return ordered


def release_reload_host_aliases(model):
    """Reload meta tensors must not retain the old pinned expert allocation.

    vLLM copies parameter __dict__ into meta tensors before loading. The
    private CPU alias is real storage, not metadata; keeping it there defeats
    releasing each old projection during VMM migration.
    """
    from vllm.model_executor.model_loader.reload.layerwise import LAYERWISE_INFO
    removed = 0
    for layer in model.modules():
        info = LAYERWISE_INFO.get(layer)
        if info is None:
            continue
        if info.kernel_tensors is not None:
            raise RuntimeError('DeepSeek placement requires initial load, not layerwise reload')
        for bank in info.restore_metadata:
            for tensor in bank.values():
                if hasattr(tensor, '_r9700_host_view'):
                    del tensor._r9700_host_view
                    for name in ('_r9700_host_bytes', '_vllm_is_uva_offloaded'):
                        if hasattr(tensor, name):
                            delattr(tensor, name)
                    removed += 1
    model._r9700_placed_requires_restart = True
    return removed


def install_reload_guard():
    from functools import wraps
    from vllm.model_executor.model_loader import reload
    from vllm.model_executor.model_loader.reload import layerwise, torchao_decorator
    original = layerwise.initialize_layerwise_reload
    if getattr(original, '_r9700_placed_guard', False):
        return
    @wraps(original)
    def initialize(model, *args, **kwargs):
        if any(getattr(layer, '_r9700_placed_requires_restart', False) for layer in model.modules()):
            raise RuntimeError('DeepSeek HIP hot/cold weights require a server restart; in-place reload is not qualified')
        return original(model, *args, **kwargs)
    initialize._r9700_placed_guard = True
    for module in (reload, layerwise, torchao_decorator):
        if getattr(module, 'initialize_layerwise_reload', None) is original:
            module.initialize_layerwise_reload = initialize


def place_model(model, config, profiles):
    """Migrate one packed projection at a time, before KV/graph allocation.

    Any failure aborts startup. A partly placed model is never exposed as a
    working fallback. Scales remain resident and are not touched here.
    """
    import re
    import torch
    from vllm.model_executor.offloader import get_offloader
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.logger import init_logger
    from .deepseek_expert_profile import identity
    from .deepseek_residency import extension, make_partially_resident
    if not config.parallel_config.enable_expert_parallel or config.parallel_config.tensor_parallel_size != 8:
        raise RuntimeError("DeepSeek hot/cold placement requires EP8/TP8")
    layers = {}
    for module in model.modules():
        if not isinstance(module, RoutedExperts) or getattr(module, 'global_num_experts', None) != 384 or not all(hasattr(module,n) for n in MATRICES):
            continue
        match = re.search(r'(?:^|\.)layers\.(\d+)\.', getattr(module,'layer_name',''))
        if not match or int(match[1]) in layers:
            raise RuntimeError("Ambiguous routed expert layer during placement")
        layers[int(match[1])] = module
    if sorted(layers) != list(range(40)):
        raise RuntimeError("Placement requires all 40 finalized backbone layers")
    expected = {'w13_weight': (48,4608,2560), 'w2_weight': (48,5120,1152)}
    expert_map = layers[0].expert_map.cpu().tolist()
    for layer in layers.values():
        if layer.expert_map.cpu().tolist() != expert_map:
            raise RuntimeError("Layer-dependent EP map is not qualified")
        for name, shape in expected.items():
            value = getattr(layer,name)
            if tuple(value.shape) != shape or value.dtype != torch.uint8 or not value.is_contiguous():
                raise RuntimeError("Unqualified Quark EP weight layout")
        for name in ('w13_weight_scale', 'w2_weight_scale'):
            value = getattr(layer, name, None)
            if value is None or getattr(value, '_vllm_is_uva_offloaded', False):
                raise RuntimeError("Placement requires resident Quark expert scales")
    device = layers[0].w13_weight.device.index
    page = int(extension().page_size(device,0))
    offloader = get_offloader()
    budget = max(int(offloader.cpu_offload_bytes), int(offloader.cpu_offload_max_bytes))
    geometry = {name: getattr(layers[0],name).nbytes for name in MATRICES}
    hot = read_profiles(profiles, identity(config))
    plan = build_plan(hot,expert_map,geometry,page,budget)
    logger = init_logger('vllm.r9700_vllm')
    jobs = []
    for index, layer in layers.items():
        for name in MATRICES:
            parameter = getattr(layer,name)
            pages = (parameter.nbytes+page-1)//page
            flags = torch.ones(pages,dtype=torch.bool)
            if plan[index,name]: flags[list(plan[index,name])] = False
            old_host = getattr(parameter,'_r9700_host_bytes',parameter.nbytes if getattr(parameter,'_vllm_is_uva_offloaded',False) else 0)
            # Positive delta frees host RAM but consumes VRAM; negative does
            # the inverse. Keep both pools balanced across projection copies.
            delta_gpu = old_host - len(plan[index,name])*page
            jobs.append((delta_gpu,index,name,flags))
    if abs(sum(job[0] for job in jobs)) >= page:
        raise RuntimeError('DeepSeek placement requires the same actual host-byte budget; refusing unbounded migration')
    removed = release_reload_host_aliases(model)
    logger.info('DeepSeek placement START device=%d projections=%d budget_bytes=%d; balanced RAM/VRAM migration',
                device,len(jobs),budget)
    logger.info('DeepSeek placement released %d loader metadata host aliases', removed)
    complete = []
    host_bytes = 0
    try:
        balance = 0
        for delta,index,name,flags in migration_order(jobs):
            parameter = getattr(layers[index],name)
            torch.cuda.empty_cache()
            replacement = make_partially_resident(parameter.detach(),flags)
            parameter.data = replacement
            if hasattr(parameter,'_r9700_host_view'): del parameter._r9700_host_view
            parameter._vllm_is_uva_offloaded = True
            parameter._r9700_host_bytes = len(plan[index,name])*page
            host_bytes += parameter._r9700_host_bytes
            complete.append((index,name))
            balance += delta
            if len(complete) % 10 == 0:
                logger.info('DeepSeek placement progress device=%d projections=%d/80 gpu_delta_bytes=%d',
                            device,len(complete),balance)
        torch.cuda.synchronize(device)
    except Exception as exc:
        raise RuntimeError(f"DeepSeek placement PARTIAL on rank/device {device}: {len(complete)}/80 projections; refusing startup") from exc
    logger.info('DeepSeek placement COMPLETE device=%d projections=%d host_bytes=%d budget_bytes=%d page_bytes=%d',
                device,len(complete),host_bytes,budget,page)


def install():
    import os
    paths = os.environ.get('R9K_DEEPSEEK_EXPERT_PLACEMENT','')
    if not paths:
        return
    install_reload_guard()
    from functools import wraps
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    original = DefaultModelLoader.load_model
    if getattr(original,'_r9700_placement',False):
        return
    @wraps(original)
    def load(self,vllm_config,model_config,prefix=''):
        hf = getattr(model_config,'hf_config',None)
        if ('DSparkV41DraftModel' in getattr(model_config,'architectures',[])
                or getattr(hf,'model_type',None) not in ('deepseek_v41','deepseek_v41_text')):
            return original(self,vllm_config,model_config,prefix)
        # Validate calibration before allocating the checkpoint, not afterwards.
        profiles = [p.strip() for p in paths.split(',') if p.strip()]
        from .deepseek_expert_profile import identity
        read_profiles(profiles,identity(vllm_config))
        if os.environ.get('R9K_DEEPSEEK_OFFLOAD_MATRICES') != '1':
            raise RuntimeError('Hot/cold placement requires matrix-only offload')
        model = original(self,vllm_config,model_config,prefix)
        place_model(model,vllm_config,profiles)
        return model
    load._r9700_placement = True
    DefaultModelLoader.load_model = load
