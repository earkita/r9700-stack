"""GLM5Next checkpoint inspection; execution belongs to upstream vLLM.

No torch/vLLM imports here: the same checks can run before allocating GPU memory.
Family detection deliberately does not use the repository name or local path.
This module does not register a Qwen implementation under a GLM architecture.
"""
from __future__ import annotations

from collections.abc import Mapping

ARCHITECTURES = frozenset({"Glm5NextForConditionalGeneration", "Glm5NextForCausalLM"})
MODEL_TYPES = frozenset({"glm5_next", "glm5_next_text"})


def is_glm5_next(config: Mapping) -> bool:
    text = config.get("text_config") or config
    return (bool(ARCHITECTURES.intersection(config.get("architectures") or []))
            or config.get("model_type") in MODEL_TYPES
            or text.get("model_type") in MODEL_TYPES)


def inspect_checkpoint(config: Mapping, tp: int = 8) -> dict:
    """Describe a baseline candidate, rejecting unsupported quantization metadata.

    This checks metadata, not weight bytes or backend availability. In particular,
    W4A4 must not be silently relabelled W4A8 just because both store MXFP4 weights.
    Per-layer overrides and exclusions remain owned by upstream QuarkConfig.
    """
    if not is_glm5_next(config):
        raise ValueError("Expected GLM5Next architecture/model_type (GLM-5.3-Flash)")
    if tp < 1:
        raise ValueError("TP must be positive")
    text = config.get("text_config") or config
    quant = config.get("quantization_config") or text.get("quantization_config") or {}
    if quant.get("quant_method") != "quark":
        raise ValueError("The baseline profile requires the original Quark checkpoint")
    global_quant = quant.get("global_quant_config") or {}
    for key, dynamic in (("weight", False), ("input_tensors", True)):
        spec = global_quant.get(key) or {}
        expected = {"dtype": "fp4", "group_size": 32, "qscheme": "per_group",
                    "ch_axis": -1, "scale_format": "e8m0", "is_dynamic": dynamic}
        for field, value in expected.items():
            if spec.get(field) != value:
                raise ValueError(f"Unsupported Quark {key}.{field}: {spec.get(field)!r}; expected {value!r}")
    export = quant.get("export") or {}
    if export.get("pack_method") != "reorder" or export.get("weight_format") != "real_quantized":
        raise ValueError("Expected Quark real_quantized/reorder weight export")
    for field in ("hidden_size", "moe_intermediate_size", "num_attention_heads"):
        value = text.get(field)
        if not isinstance(value, int) or value <= 0 or value % tp:
            raise ValueError(f"{field}={value!r} must be positive and divisible by TP={tp}")
    return {
        "family": "glm5_next", "quantization": "quark", "weight_format": "MXFP4/E8M0/group32",
        "activation_format": "dynamic MXFP4/E8M0/group32", "tp": tp,
        "hidden_size": text["hidden_size"], "experts": text.get("n_routed_experts"),
        "top_k": text.get("num_experts_per_tok"), "shared_experts": text.get("n_shared_experts"),
        "expert_intermediate_per_rank": text["moe_intermediate_size"] // tp,
        "router_output_dtype": text.get("moe_router_dtype"),
        "scoring_func": text.get("scoring_func"), "topk_method": text.get("topk_method"),
        "norm_topk_prob": text.get("norm_topk_prob"),
        "routed_scaling_factor": text.get("routed_scaling_factor"),
        "swiglu_limit": text.get("swiglu_limit"),
        "layer_types": sorted(set(text.get("layer_types", []))),
        "layer_quant_overrides": len(quant.get("layer_quant_config") or {}),
        "execution": "metadata check only; runtime kernel selection is validated separately",
    }


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="config.json, checkpoint directory, or Hugging Face ID (metadata only)")
    parser.add_argument("--tp", type=int, default=8)
    args = parser.parse_args()
    try:
        path = Path(args.config)
        if path.is_dir():
            path = path / "config.json"
        elif not path.exists() and not path.is_absolute():
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(args.config, "config.json"))
        result = inspect_checkpoint(json.loads(path.read_text()), args.tp)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"GLM preflight: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
