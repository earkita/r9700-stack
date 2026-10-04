"""CPU-only header audit; a successful audit does NOT qualify a serving backend."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import struct

GIB = 1 << 30
DTYPE_BYTES = {"U8": 1, "F8_E4M3": 1, "F8_E8M0": 1, "BF16": 2, "F32": 4}


def read_headers(path):
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f"Truncated header: {path.name}")
        length = struct.unpack("<Q", raw)[0]
        if not 2 <= length <= min(100_000_000, path.stat().st_size - 8):
            raise ValueError(f"Invalid header length: {path.name}")
        data = json.loads(stream.read(length))
    data.pop("__metadata__", None)
    end = path.stat().st_size - length - 8
    ranges = []
    for name, tensor in data.items():
        start, stop = tensor["data_offsets"]
        shape = tensor["shape"]
        if not all(isinstance(n, int) and n >= 0 for n in shape):
            raise ValueError(f"Invalid shape: {name}")
        if tensor["dtype"] not in DTYPE_BYTES:
            raise ValueError(f"Unsupported dtype: {name}: {tensor['dtype']}")
        if not 0 <= start <= stop <= end:
            raise ValueError(f"Truncated tensor: {name}")
        if stop - start != math.prod(shape) * DTYPE_BYTES[tensor["dtype"]]:
            raise ValueError(f"Invalid byte count: {name}")
        ranges.append((start, stop))
    cursor = 0
    for start, stop in sorted(ranges):
        if start != cursor:
            raise ValueError(f"Overlapping or non-contiguous payload: {path.name}")
        cursor = stop
    if cursor != end:
        raise ValueError(f"Unindexed payload bytes: {path.name}")
    return data


def component(name):
    # Draft experts must not inflate the backbone's offload estimate.
    if name.startswith("mtp."):
        return "draft"
    if name.startswith(("vision.", "aligner.", "image")):
        return "vision"
    if ".engram.embed." in name:
        return "engram_tables"
    if ".engram." in name:
        return "engram_projections"
    if ".ffn.experts." in name:
        return "routed_experts"
    if ".ffn.shared_experts." in name:
        return "shared_experts"
    return "other"


def check_scales(tensors):
    for name, scale in tensors.items():
        if not name.endswith(".weight_scale"):
            continue
        weight_name = name.removesuffix("_scale")
        if weight_name not in tensors:
            raise ValueError(f"Missing weight for {name}")
        weight = tensors[weight_name]
        rows, cols = weight["shape"]
        if weight["dtype"] == "U8":
            expected, dtype = [rows, math.ceil(cols * 2 / 32)], "U8"
        elif ".engram.embed." in name:
            expected, dtype = [rows, math.ceil(cols / 32)], "F8_E8M0"
        elif weight["dtype"] == "F8_E4M3":
            expected, dtype = [math.ceil(rows / 32), math.ceil(cols / 32)], "F8_E8M0"
        else:
            raise ValueError(f"Unsupported quantized weight: {weight_name}")
        if scale["shape"] != expected or scale["dtype"] != dtype:
            raise ValueError(f"Invalid scale layout: {name}; expected {expected}/{dtype}")
    for name, weight in tensors.items():
        if name.endswith(".weight") and weight["dtype"] in ("U8", "F8_E4M3"):
            if name + "_scale" not in tensors:
                raise ValueError(f"Missing scale for {name}")


def memory_estimates(groups, tables, tp, offload_gib):
    if tp <= 0 or not math.isfinite(offload_gib) or offload_gib < 0:
        raise ValueError("TP must be positive; offload must be finite and nonnegative")
    offload = offload_gib * GIB
    if offload > groups["routed_experts"]:
        raise ValueError("Offload exceeds backbone expert payload")
    text_gpu = sum(v for k, v in groups.items() if k not in (
        "engram_tables", "vision", "draft"))
    # Separate weight and scale allocations on each TP rank; not a measured RSS.
    rounded = sum(tp * (1 << (math.ceil(size / tp) - 1).bit_length()) for size in tables)
    return {
        "text_weight_bytes_excluding_cpu_engram": text_gpu,
        "ideal_weight_gib_per_gpu_after_offload": (text_gpu - offload) / tp / GIB,
        "expert_offload_gib_total": offload_gib,
        "expert_offload_gib_per_rank": offload_gib / tp,
        "host_payload_gib_with_exact_engram": (groups["engram_tables"] + offload) / GIB,
        "engram_gib_if_each_tp_allocation_rounds_to_power_of_two": rounded / GIB,
        "qualification": "Estimates only: excludes replication, padding, allocator overhead, KV, workspaces and load peaks",
    }


def audit(root, tp=8, offload_gib=80):
    root = Path(root).resolve()
    config_raw = (root / "config.json").read_bytes()
    index_raw = (root / "model.safetensors.index.json").read_bytes()
    cfg, index = json.loads(config_raw), json.loads(index_raw)
    if cfg.get("model_type") != "deepseek_v41" or tp != 8:
        raise ValueError("This audit supports DeepSeek V4.1 at TP8 only")
    text = cfg["text_config"]
    for key in ("hidden_size", "moe_intermediate_size", "num_attention_heads"):
        if text[key] % tp:
            raise ValueError(f"Cannot evenly shard {key} at TP{tp}")
    quant = cfg["quantization_config"]
    global_quant = quant["global_quant_config"]
    if quant["quant_method"] != "quark":
        raise ValueError("Expected Quark checkpoint")
    for key in ("weight", "input_tensors"):
        q = global_quant[key]
        if (q["dtype"], q["group_size"], q["scale_format"]) != ("fp4", 32, "e8m0"):
            raise ValueError(f"Unexpected global {key} quantization")
    for name, q in quant["layer_quant_config"].items():
        if (q["weight"]["block_size"] != [32, 32]
                or q["weight"]["scale_type"] != "float8_e8m0fnu"
                or q["input_tensors"]["group_size"] != 32):
            raise ValueError(f"Unexpected attention quantization: {name}")
    owners = defaultdict(set)
    for name, shard in index["weight_map"].items():
        if Path(shard).name != shard:
            raise ValueError("Shard must be a filename within the checkpoint")
        owners[shard].add(name)
    tensors, groups, tables = {}, defaultdict(int), []
    file_bytes = 0
    for shard, names in sorted(owners.items()):
        path = root / shard
        header = read_headers(path)
        if set(header) != names:
            raise ValueError(f"Index/header mismatch: {shard}")
        file_bytes += path.stat().st_size
        tensors.update(header)
    check_scales(tensors)
    for name, tensor in tensors.items():
        size = tensor["data_offsets"][1] - tensor["data_offsets"][0]
        group = component(name)
        groups[group] += size
        if group == "engram_tables":
            tables.append(size)
    return {
        "model_type": cfg["model_type"], "tp": tp,
        "config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "index_sha256": hashlib.sha256(index_raw).hexdigest(),
        "shards": len(owners), "tensors": len(tensors),
        "payload_bytes": sum(groups.values()), "shard_file_bytes": file_bytes,
        "index_metadata_total_size": index.get("metadata", {}).get("total_size"),
        "component_bytes": dict(sorted(groups.items())),
        "component_gib": {k: v / GIB for k, v in sorted(groups.items())},
        "expert_intermediate_per_rank": text["moe_intermediate_size"] // tp,
        "attention_override_count": len(quant["layer_quant_config"]),
        "memory": memory_estimates(groups, tables, tp, offload_gib),
        "validation_scope": "Headers, offsets, index membership and scale geometry only; no payload checksums, numerical or GPU validation",
        "runtime_qualified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--expert-offload-gib", type=float, default=80,
                        help="Total across all eight ranks, NOT per GPU")
    args = parser.parse_args()
    print(json.dumps(audit(args.model, offload_gib=args.expert_offload_gib), indent=2))


if __name__ == "__main__":
    main()
