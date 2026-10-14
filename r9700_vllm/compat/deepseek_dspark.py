"""Pinned Quark configuration names for the built-in DeepSeek DSpark model.

The draft's ModuleList uses indices0..2 for checkpoint loading, but its layer
prefixes are40..42 for attention state. Quark rules must match those constructor
prefixes; the upstream custom weight loader still owns tensor-name mapping.
"""
import os


def check_speculation(config):
    """Reject unqualified draft settings during construction, before loading."""
    if config is None:
        return
    if getattr(config, 'method', None) != 'dspark':
        raise RuntimeError('DeepSeek packed draft supports only the opt-in DSpark path')
    if os.environ.get('R9K_DEEPSEEK_DSPARK', '0') != '1':
        raise RuntimeError('DeepSeek DSpark requires R9K_DEEPSEEK_DSPARK=1')
    if (config.num_speculative_tokens != 5
            or config.enable_adaptive_verification
            or config.draft_sample_method != 'probabilistic'):
        raise RuntimeError('DeepSeek DSpark qualification requires static 5 tokens and probabilistic draft sampling')


def quant_mapper():
    from vllm.model_executor.models.utils import WeightsMapper
    prefixes = {}
    for stage in range(3):
        for name in ('main_proj', 'main_norm', 'norm', 'confidence_head'):
            prefixes[f'mtp.{stage}.{name}'] = f'model.{name}'
        prefixes[f'mtp.{stage}.markov_head.embed'] = 'model.markov_head.markov_w1'
        prefixes[f'mtp.{stage}.markov_head.head'] = 'model.markov_head.markov_w2'
        prefixes[f'mtp.{stage}.'] = f'model.layers.{40+stage}.'
    return WeightsMapper(orig_to_new_prefix=prefixes)


def install():
    if os.environ.get('R9K_DEEPSEEK_DSPARK','0') != '1':
        return
    from vllm.models.deepseek_v41.amd.dspark import DSparkDeepseekV4ForCausalLM as Draft
    if getattr(Draft,'_r9700_quark_names',False):
        return
    if hasattr(Draft,'hf_to_vllm_mapper') or hasattr(Draft,'packed_modules_mapping'):
        raise RuntimeError('DeepSeek DSpark mapper interface changed; revalidate Quark rules')
    Draft.hf_to_vllm_mapper = quant_mapper()
    Draft.packed_modules_mapping = {
        'gate_up_proj': ['w1','w3'],
        'fused_wqa_wkv': ['wq_a','wkv'],
    }
    Draft._r9700_quark_names = True
