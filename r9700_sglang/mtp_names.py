# SPDX-License-Identifier: Apache-2.0
"""GLM NextN runtime names versus the original Quark checkpoint names.

Only names change. Quantization metadata and tensors remain untouched.
"""


def nextn_name_map(num_hidden_layers):
    if not isinstance(num_hidden_layers, int) or num_hidden_layers <= 0:
        raise ValueError("GLM NextN requires a positive target layer count")
    mappings = {"lm_head": "model.shared_head.head"}
    for root in ("model.layers", "model.language_model.layers"):
        base = f"{root}.{num_hidden_layers}"
        for name in ("eh_proj", "enorm", "hnorm", "shared_head"):
            mappings[f"{base}.{name}"] = f"model.{name}"
        mappings[base] = "model.decoder"
    return mappings


def checkpoint_name(runtime_name, num_hidden_layers):
    if runtime_name.startswith("model.decoder."):
        return (f"model.language_model.layers.{num_hidden_layers}."
                + runtime_name.removeprefix("model.decoder."))
    if runtime_name.startswith("model.layers."):
        return "model.language_model." + runtime_name.removeprefix("model.")
    return runtime_name


def nextn_expert_fp8_config(quant_config, num_hidden_layers):
    """Require uniform native block-FP8 metadata for every draft expert."""
    prefix = f"model.language_model.layers.{num_hidden_layers}.mlp.experts."
    configs = {k.removeprefix(prefix): v for k, v in
               quant_config.get("layer_quant_config", {}).items() if k.startswith(prefix)}
    ids = {int(k.split('.')[0]) for k in configs}
    if not ids or ids != set(range(max(ids) + 1)):
        raise ValueError("GLM NextN expert metadata is incomplete")
    required = {f"{i}.{p}" for i in ids for p in ("gate_proj", "up_proj", "down_proj")}
    if set(configs) != required:
        raise ValueError("GLM NextN needs gate/up/down metadata for every expert")
    first = next(iter(configs.values()))
    if not all(v == first for v in configs.values()):
        raise ValueError("GLM NextN expert formats are not uniform")
    weight, inputs = first.get("weight", {}), first.get("input_tensors", {})
    if not (weight.get("dtype") == "fp8_e4m3" and weight.get("qscheme") == "per_block"
            and weight.get("block_size") == [128, 128] and weight.get("is_dynamic") is False
            and inputs.get("dtype") == "fp8_e4m3" and inputs.get("qscheme") == "per_group"
            and inputs.get("group_size") == 128 and inputs.get("is_dynamic") is True):
        raise ValueError("GLM NextN requires native FP8 128x128 weights and dynamic group-128 activations")
    return first
