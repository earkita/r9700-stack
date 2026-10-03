"""CPU-only structural preflight for the local MiMo MOPD checkpoint."""

import argparse
import json
import math
from pathlib import Path
import struct


def headers(path):
    with path.open("rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        if length > 100_000_000:
            raise ValueError(f"Invalid safetensors header: {path.name}")
        tensors = json.loads(f.read(length))
    end = path.stat().st_size - length - 8
    for name, info in tensors.items():
        if name != "__metadata__":
            start, stop = info["data_offsets"]
            if not 0 <= start <= stop <= end:
                raise ValueError(f"Truncated tensor {path.name}:{name}")
            element_bytes = {"U8": 1, "F8_E4M3": 1, "BF16": 2, "F32": 4}.get(
                info["dtype"]
            )
            if (
                element_bytes
                and stop - start != math.prod(info["shape"]) * element_bytes
            ):
                raise ValueError(f"Invalid tensor byte count {path.name}:{name}")
    return tensors


def validate(root, tp=8):
    root = Path(root)
    cfg = json.loads((root / "config.json").read_text())
    expected = dict(
        model_type="mimo_v2",
        num_hidden_layers=48,
        hidden_size=4096,
        num_attention_heads=64,
        num_key_value_heads=4,
        head_dim=192,
        v_head_dim=128,
        swa_num_key_value_heads=8,
        n_routed_experts=256,
        num_experts_per_tok=8,
        moe_intermediate_size=2048,
        sliding_window=128,
        sliding_window_size=128,
        swa_num_attention_heads=64,
        swa_head_dim=192,
        swa_v_head_dim=128,
    )
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"Unsupported MiMo {key}: {cfg.get(key)!r}")
    if (
        tp != 8
        or cfg["hybrid_layer_pattern"].count(0) != 9
        or cfg["hybrid_layer_pattern"].count(1) != 39
    ):
        raise ValueError("Requires TP8 and 9 global / 39 sliding layers")
    quant = cfg["quantization_config"]
    if (
        quant["quant_method"] != "fp8"
        or quant["store_dtype"] != "mxfp4"
        or quant.get("weight_block_size") != [128, 128]
        or quant.get("mxfp4_block_size") != 32
    ):
        raise ValueError("Requires original FP8/MXFP4 checkpoint")
    index = json.loads((root / "model.safetensors.index.json").read_text())
    if index["metadata"]["tp_size"] != 4:
        raise ValueError("Expected TP4 checkpoint layout")
    tensors = {}
    for file in set(index["weight_map"].values()):
        shard = headers(root / file)
        for name, owner in index["weight_map"].items():
            if owner == file and name not in shard:
                raise ValueError(f"Missing indexed tensor {file}:{name}")
        tensors.update(shard)
    for layer, sliding in enumerate(cfg["hybrid_layer_pattern"]):
        prefix = f"model.layers.{layer}.self_attn.qkv_proj"
        rows = 14848 if sliding else 13568
        for suffix, shape, dtype in [
            ("weight", [rows, 4096], "F8_E4M3"),
            ("weight_scale_inv", [116 if sliding else 108, 32], "F32"),
        ]:
            t = tensors[prefix + "." + suffix]
            if t["shape"] != shape or t["dtype"] != dtype:
                raise ValueError(f"Unexpected fused QKV layout: {prefix}.{suffix}")
    for name, tensor in tensors.items():
        if ".mlp.experts." in name:
            down = ".down_proj." in name
            shape = (
                ([4096, 64] if down else [2048, 128])
                if name.endswith("weight_scale")
                else ([4096, 1024] if down else [2048, 2048])
            )
            if tensor["shape"] != shape or tensor["dtype"] != "U8":
                raise ValueError(f"Unexpected expert layout: {name}")
    draft = json.loads((root / "dflash/config.json").read_text())
    if (
        draft["architectures"] != ["DFlashDraftModel"]
        or draft["num_hidden_layers"] != 5
        or draft["hidden_size"] != 4096
        or draft["dflash_config"]["target_layer_ids"] != [0, 11, 23, 35, 47]
    ):
        raise ValueError("Incompatible DFlash drafter")
    for key, value in dict(
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=128,
        v_head_dim=128,
        intermediate_size=16384,
        sliding_window=1024,
        block_size=8,
        is_causal=False,
    ).items():
        if draft.get(key) != value:
            raise ValueError(f"Incompatible DFlash {key}: {draft.get(key)!r}")
    files = list((root / "dflash").glob("*.safetensors"))
    if not files:
        raise ValueError("Missing DFlash weights")
    draft_tensors = {}
    for file in files:
        draft_tensors.update(headers(file))
    shapes = {
        "fc.weight": [4096, 20480],
        "hidden_norm.weight": [4096],
        "norm.weight": [4096],
    }
    layer_shapes = {
        "input_layernorm.weight": [4096],
        "post_attention_layernorm.weight": [4096],
        "self_attn.q_proj.weight": [8192, 4096],
        "self_attn.k_proj.weight": [1024, 4096],
        "self_attn.v_proj.weight": [1024, 4096],
        "self_attn.o_proj.weight": [4096, 8192],
        "self_attn.q_norm.weight": [128],
        "self_attn.k_norm.weight": [128],
        "self_attn.attention_sink_bias": [64],
        "mlp.gate_proj.weight": [16384, 4096],
        "mlp.up_proj.weight": [16384, 4096],
        "mlp.down_proj.weight": [4096, 16384],
    }
    for layer in range(5):
        shapes.update(
            {f"layers.{layer}.{name}": shape for name, shape in layer_shapes.items()}
        )
    for name, shape in shapes.items():
        tensor = draft_tensors.get(name, {})
        if tensor.get("shape") != shape or tensor.get("dtype") != "BF16":
            raise ValueError(f"Incompatible or missing DFlash tensor: {name}")
    print(
        f"MiMo preflight PASS: {len(set(index['weight_map'].values()))} shards; TP4 -> TP8; 9 GA / 39 SWA"
    )
    return tensors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--tp", type=int, default=8)
    args = parser.parse_args()
    validate(args.model, args.tp)
