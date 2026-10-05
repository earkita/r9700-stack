"""Scoped DeepSeek streaming/host copies and bounded loader diagnostics."""
from functools import wraps
import os
import time


def text_weights(mapped):
    """One contiguous LM group without retaining/sorting the checkpoint.

    Only used when both vision modules are StageMissingLayer. The mapper
    still owns all renaming and exclusion (including MTP), as upstream does.
    """
    for name, tensor in mapped:
        if name.startswith("language_model."):
            yield name, tensor
        elif name.startswith(("vision.", "aligner.")) or name in (
            "image_start", "image_end", "image_newline"
        ):
            continue
        else:
            raise RuntimeError(f"Unexpected DeepSeek text-only weight: {name}")


def load_host_expert(original, layer, param, loaded_weight, *args, **kwargs):
    """Reuse upstream TP slicing with a CPU alias; no extra tensor allocation.

    Loading is synchronous and single-threaded per worker. Restore the GPU
    alias even on error; forward and post-load hooks always see that alias.
    Packed weights and raw E8M0 bytes are copied without conversion.
    """
    import torch
    host = getattr(param, "_r9700_host_view", None)
    if host is None or loaded_weight.device.type != "cpu":
        return original(layer, param, loaded_weight, *args, **kwargs)
    if param.dtype != torch.uint8 or loaded_weight.dtype != torch.uint8:
        return original(layer, param, loaded_weight, *args, **kwargs)
    if (host.device.type != "cpu" or host.shape != param.shape
            or host.dtype != param.dtype or host.stride() != param.stride()):
        raise RuntimeError("DeepSeek host alias no longer matches the expert parameter")
    accelerator = param.data
    try:
        param.data = host
        return original(layer, param, loaded_weight, *args, **kwargs)
    finally:
        param.data = accelerator


def install_optimized_loading():
    """Called only from the pinned ROCm/gfx1201 DeepSeek registration.

    R9K_DEEPSEEK_LOAD=stock retains the old path for controlled comparisons.
    Vision tensors load immediately; the LM still finalizes once at EOF.
    """
    mode = os.environ.get("R9K_DEEPSEEK_LOAD", "stock")
    if mode == "stock":
        return
    if mode != "stream":
        raise ValueError("R9K_DEEPSEEK_LOAD must be stock or stream")
    from .deepseek import _is_deepseek
    from vllm.model_executor.models.utils import AutoWeightsLoader, StageMissingLayer
    from vllm.models.deepseek_v41.amd.vl_model import DeepseekV41ForCausalLM
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.logger import init_logger
    original = DeepseekV41ForCausalLM.load_weights
    if getattr(original, "_r9700_ds_stream", False):
        return
    log = init_logger("vllm.r9700_vllm")
    buffer_mib = int(os.environ.get("R9K_DEEPSEEK_LOAD_BUFFER_MIB", "0"))
    if not 0 <= buffer_mib <= 256:
        raise ValueError("R9K_DEEPSEEK_LOAD_BUFFER_MIB must be 0..256")

    def load_stream(self, weights):
        mapped = self.hf_to_vllm_mapper.apply(weights)
        loader = AutoWeightsLoader(self)
        vision_loaded = set()
        if (isinstance(self.vision, StageMissingLayer)
                and isinstance(self.aligner, StageMissingLayer)):
            mapped = text_weights(mapped)
        else:
            # The pinned ViT/aligner use ordinary parameter weight loaders,
            # without group-level finalization. Load each tensor immediately;
            # never retain the full checkpoint just to sort its root groups.
            source = mapped

            def language_weights():
                for name, tensor in source:
                    if name.startswith("language_model."):
                        yield name, tensor
                    elif name.startswith(("vision.", "aligner.")) or name in (
                            "image_start", "image_end", "image_newline"):
                        vision_loaded.update(loader.load_weights([(name, tensor)]))
                    else:
                        raise RuntimeError(f"Unexpected DeepSeek vision weight: {name}")

            mapped = language_weights()
        result = loader.load_weights(mapped) | vision_loaded
        # One delegation to language_model means one finalization, after EOF.
        self._weights_finalized = True
        return result

    @wraps(original)
    def stream(self, weights):
        if not _is_deepseek():
            return original(self, weights)
        if not buffer_mib:
            return load_stream(self, weights)
        from vllm.config import get_current_vllm_config
        if get_current_vllm_config().load_config.safetensors_load_strategy == "eager":
            raise ValueError("Buffered DeepSeek loading requires lazy safetensors; "
                             "eager still allocates whole shards per rank")
        from .deepseek_load_buffer import BufferedLoadCopies
        log.info("r9700: DeepSeek checkpoint staging buffer=%d MiB per rank", buffer_mib)
        with BufferedLoadCopies(buffer_mib * 1024**2) as staging:
            result = load_stream(self, weights)
        log.info("r9700: DeepSeek staging finished: copies=%d tiles=%d "
                 "bytes=%d largest_tile=%d; host buffer released",
                 staging.copies, staging.tiles, staging.staged_bytes,
                 staging.peak_tile_bytes)
        return result

    load = RoutedExperts.weight_loader

    @wraps(load)
    def host_copy(self, param, loaded_weight, *args, **kwargs):
        if not _is_deepseek():
            return load(self, param, loaded_weight, *args, **kwargs)
        return load_host_expert(load, self, param, loaded_weight, *args, **kwargs)

    stream._r9700_ds_stream = True
    DeepseekV41ForCausalLM.load_weights = stream
    RoutedExperts.weight_loader = host_copy
    log.info("r9700: DeepSeek weights stream without sorting; vision loads immediately; offloaded experts use CPU aliases")


def install():
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    from vllm.logger import init_logger
    from .deepseek import _is_deepseek
    original = DefaultModelLoader.get_all_weights
    if getattr(original, "_r9700_ds_loading", False):
        return
    log = init_logger("vllm.r9700_vllm")

    @wraps(original)
    def weights(self, *args, **kwargs):
        source = original(self, *args, **kwargs)
        if not _is_deepseek():
            yield from source
            return
        start = report = time.monotonic()
        count = size = 0
        reader_time = consumer_time = worst = 0.
        worst_name = ""
        iterator = iter(source)
        while True:
            begin = time.monotonic()
            try:
                name, value = next(iterator)
            except StopIteration:
                break
            ready = time.monotonic()
            reader_time += ready - begin
            count += 1
            size += value.numel() * value.element_size()
            yield name, value
            now = time.monotonic()
            spent = now - ready
            consumer_time += spent
            if spent > worst:
                worst, worst_name = spent, name
            if now - report >= 30:
                log.info("DeepSeek load: yielded=%d logical_GiB=%.2f elapsed=%.1fs "
                         "iterator=%.1fs consumer=%.1fs slowest=%.3fs tensor=%s",
                         count,size/2**30,now-start,reader_time,consumer_time,worst,worst_name)
                report = now
                worst = 0.
        log.info("DeepSeek loader iterator exhausted: tensors=%d elapsed=%.1fs; "
                 "downstream buffered copies/finalization may remain",count,time.monotonic()-start)

    weights._r9700_ds_loading = True
    DefaultModelLoader.get_all_weights = weights

    # AutoWeightsLoader may buffer/reorder tensors after the iterator exhausts.
    # Measure the actual expert loader too; iterator timing alone misses that.
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    load = RoutedExperts.weight_loader
    totals = [time.monotonic(), 0, 0., 0., "", None, None, False]

    @wraps(load)
    def expert_weight(self, param, loaded_weight, *args, **kwargs):
        if not _is_deepseek():
            return load(self,param,loaded_weight,*args,**kwargs)
        begin = time.monotonic()
        try:
            return load(self,param,loaded_weight,*args,**kwargs)
        finally:
            now = time.monotonic()
            elapsed = now - begin
            totals[1] += 1
            totals[2] += elapsed
            if elapsed > totals[3]:
                totals[3] = elapsed
                totals[4] = str(args[0] if args else kwargs.get("weight_name", ""))
                totals[5:8] = [str(loaded_weight.device), str(param.device),
                               getattr(param, "_vllm_is_uva_offloaded", False)]
            if now - totals[0] >= 30:
                log.info("DeepSeek expert load: calls=%d loader_seconds=%.1f "
                         "slowest=%.3fs weight=%s source=%s target=%s uva=%s",
                         totals[1],totals[2],totals[3],totals[4],
                         *totals[5:8])
                totals[:] = [now,0,0.,0.,"",None,None,False]

    RoutedExperts.weight_loader = expert_weight
