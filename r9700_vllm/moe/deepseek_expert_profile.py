"""Opt-in routing calibration; disabled during performance measurements.

Only TP rank zero records GLOBAL routes, before EP dispatch. Phases come from
attention scheduler metadata, never a query-row threshold. Explicit reset/dump
commands run outside forward and synchronize counters under a lock.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time


def identity(config):
    root = Path(config.model_config.model)
    names = ("config.json", "model.safetensors.index.json")
    return {
        "checkpoint": {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in names},
        "layout": "quark-mxfp4-group32-ep8-h5120-i2304-e384",
        "vllm": "18f8f960",
        "eager": config.model_config.enforce_eager,
        "max_num_seqs": config.scheduler_config.max_num_seqs,
        "max_num_batched_tokens": config.scheduler_config.max_num_batched_tokens,
        "speculative_tokens": getattr(config.speculative_config, "num_speculative_tokens", 0) or 0,
        "speculative_method": getattr(config.speculative_config, "method", None),
        "dense_dequant_at_load": os.environ.get("VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD", "1") == "1",
        "stage_min_tokens": int(os.environ.get("R9K_DEEPSEEK_STAGE_MIN_TOKENS", "0")),
        "reserved_staging": os.environ.get("R9K_DEEPSEEK_STAGE_BUFFER", "0") == "1",
    }


def phase_ranges(metadata, rows):
    """Decode precedes prefill in upstream metadata; ignore graph padding."""
    if not isinstance(metadata, dict):
        return None  # Synthetic memory/warmup run, not calibration traffic.
    candidates = set()
    for value in metadata.values():
        d, p = getattr(value, "num_decode_tokens", None), getattr(value, "num_prefill_tokens", None)
        if isinstance(d, int) and isinstance(p, int):
            candidates.add((d, p))
    if len(candidates) != 1:
        raise RuntimeError("Cannot classify DeepSeek routing from scheduler metadata")
    decode, prefill = candidates.pop()
    if min(decode, prefill) < 0 or not 0 < decode + prefill <= rows:
        raise RuntimeError("DeepSeek routing/metadata token count mismatch")
    return {"all": (0, decode + prefill), "decode": (0, decode), "prefill": (decode, decode + prefill)}


class Recorder:
    def __init__(self, output, config):
        import torch
        self.output = Path(output)
        self.identity = identity(config)
        self.lock = threading.RLock()
        self.counters = {}
        self.device = torch.cuda.current_device()
        self.epoch = 0
        self.command = None
        self.output.parent.mkdir(parents=True, exist_ok=True)
        install_runner_lock(self.lock)
        threading.Thread(target=self.control, daemon=True).start()

    def record(self, layer, ids, context):
        import torch
        with self.lock:
            if layer not in self.counters:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Routing counters require eager warmup before capture")
                self.counters[layer] = {phase: torch.zeros(2,385,dtype=torch.int64,device=ids.device)
                                        for phase in ('all','decode','prefill')}
            phases = phase_ranges(context.attn_metadata, ids.shape[0])
            if phases is None:
                return
            mapping = context.slot_mapping
            slots = next((v for k,v in mapping.items() if k.endswith('.swa_cache')
                          and v.ndim == 1 and v.shape[0] >= ids.shape[0]), None) if isinstance(mapping,dict) else None
            if slots is None:
                raise RuntimeError("Routing calibration needs live SWA slot mappings to exclude graph padding")
            valid = slots[:ids.shape[0]] >= 0
            bank = self.counters[layer]
            for phase, (start, end) in phases.items():
                if start == end:
                    continue
                counts, calls = count_routes(ids[start:end], valid[start:end])
                bank[phase][0,:384].add_(counts)
                bank[phase][1,:384].add_(counts > 0)
                bank[phase][:,384].add_(calls)

    def control(self):
        import torch
        torch.cuda.set_device(self.device)
        request = Path(str(self.output) + ".control")
        while True:
            time.sleep(.5)
            try:
                if not request.exists():
                    continue
                command = json.loads(request.read_text())
                if command == self.command:
                    continue
                # Banks are created inside the inference-mode model forward.
                # This control thread must enter the same mode to reset them.
                with self.lock, torch.inference_mode():
                    torch.cuda.synchronize()
                    if command["action"] == "reset":
                        for bank in self.counters.values():
                            for counter in bank.values(): counter.zero_()
                        torch.cuda.synchronize()
                        self.epoch += 1
                    elif command["action"] != "snapshot":
                        raise ValueError("Control action must be reset or snapshot")
                    rows = []
                    for layer, bank in sorted(self.counters.items()):
                        row = {"layer": layer}
                        for phase in ("all", "prefill", "decode"):
                            data = bank[phase].cpu().tolist() if phase in bank else [[0]*385]*2
                            row[phase] = {"calls": data[0][384], "slots": data[0][:384], "calls_hit": data[1][:384]}
                        rows.append(row)
                payload = {"schema": 1, "identity": self.identity, "epoch": self.epoch,
                           "command": command, "rows": rows, "timing_valid": False}
                temporary = self.output.with_suffix('.tmp')
                temporary.write_text(json.dumps(payload, allow_nan=False))
                temporary.replace(self.output)
                self.command = command
            except Exception as exc:
                from vllm.logger import init_logger
                init_logger("vllm.r9700_vllm").error("Routing snapshot failed: %s", exc)
                self.command = command if 'command' in locals() else None


def install():
    output = os.environ.get("R9K_DEEPSEEK_EXPERT_PROFILE")
    if not output:
        return
    from functools import wraps
    import inspect
    from vllm.model_executor.layers.quantization.quark.quark_moe import QuarkOCP_MX_MoEMethod
    from vllm.config.vllm import get_current_vllm_config_or_none
    from vllm.forward_context import get_forward_context
    from vllm.distributed import get_tensor_model_parallel_rank
    original = QuarkOCP_MX_MoEMethod.apply
    if getattr(original, '_r9700_routing', False):
        return
    expected = ['self','layer','x','topk_weights','topk_ids','shared_experts','shared_experts_input']
    if list(inspect.signature(original).parameters) != expected:
        raise RuntimeError("Quark routing interface changed")
    state = []
    original_init = QuarkOCP_MX_MoEMethod.__init__
    @wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        config = get_current_vllm_config_or_none()
        hf = getattr(getattr(config, 'model_config', None), 'hf_config', None)
        self._r9700_profile_cfg = config if getattr(hf, 'model_type', None) in ('deepseek_v41', 'deepseek_v41_text') else None
    QuarkOCP_MX_MoEMethod.__init__ = initialize
    @wraps(original)
    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        config = getattr(self, "_r9700_profile_cfg", None)
        if config is not None and layer.global_num_experts == 384 and get_tensor_model_parallel_rank() == 0:
            if not config.parallel_config.enable_expert_parallel:
                raise RuntimeError("DeepSeek routing calibration requires EP8")
            match = re.search(r'(?:^|\.)layers\.(\d+)\.', layer.layer_name)
            if not match or not 0 <= int(match[1]) < 40:
                raise RuntimeError("Unrecognized backbone MoE layer name")
            if not state: state.append(Recorder(output, config))
            state[0].record(int(match[1]), topk_ids, get_forward_context())
        return original(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input)
    apply._r9700_routing = True
    QuarkOCP_MX_MoEMethod.apply = apply


def count_routes(ids, valid):
    """Fixed-size GPU histogram; invalid/padded rows cannot count as experts."""
    import torch
    selected = torch.where(valid[:,None], ids.long(), 384).flatten()
    counts = torch.zeros(385, dtype=torch.int64, device=ids.device)
    counts.scatter_add_(0,selected,torch.ones_like(selected))
    return counts[:384], valid.any().to(torch.int64)


def install_runner_lock(lock):
    # Graph replay bypasses Python MoE hooks. Serialize diagnostic snapshots
    # against model execution too, then synchronize before copying counters.
    # Installed only when the explicitly requested DeepSeek recorder starts.
    from functools import wraps
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    original = GPUModelRunner.execute_model
    if getattr(original,'_r9700_routing_snapshot_lock',False):
        return
    @wraps(original)
    def execute(self,*args,**kwargs):
        with lock:
            return original(self,*args,**kwargs)
    execute._r9700_routing_snapshot_lock = True
    GPUModelRunner.execute_model = execute
