# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, pinned GLM FP8 reader integration without modifying vLLM files.

Only GLM TP8 rope-free E4M3 instances use the local reader. A private globals
binding on the metadata builder avoids changing the shared backend predicate.
Other model instances and BF16 calls retain upstream methods and arithmetic.
"""
from functools import wraps
from types import FunctionType


def eligible(model_config, cache_dtype, heads, head_size, kv_lora_rank):
    hf = getattr(model_config, 'hf_config', None)
    # The conditional-generation wrapper constructs its language model under
    # text_config; attention instances therefore see glm5_next_text.
    return (getattr(hf, 'model_type', None) in ('glm5_next', 'glm5_next_text')
            and cache_dtype in ('fp8', 'fp8_e4m3') and heads == 8
            and head_size == kv_lora_rank == 512)


def _private_build(original):
    upstream_select = original.__globals__['_use_rocm_sparse_triton']
    def select(**kwargs):
        if (kwargs['kv_cache_dtype'] in ('fp8', 'fp8_e4m3')
                and kwargs['head_size'] == kwargs['kv_lora_rank'] == 512
                and (kwargs['num_prefills'] > 0 or kwargs['num_decodes'] > 0)):
            return True
        return upstream_select(**kwargs)
    local = FunctionType(original.__code__, original.__globals__ | {
        '_use_rocm_sparse_triton': select}, original.__name__,
        original.__defaults__, original.__closure__)
    local.__kwdefaults__ = original.__kwdefaults__
    @wraps(original)
    def build(self, *args, **kwargs):
        dims = self.mla_dims
        use_local = eligible(self.model_config, self.kv_cache_dtype, self.num_heads,
                             dims.kv_lora_rank + dims.qk_rope_head_dim, dims.kv_lora_rank)
        return (local if use_local else original)(self, *args, **kwargs)
    return build


def forward_fp8(self, q, kv_cache, meta, layer):
    """Pinned upstream forward_mqa with BF16 Q and explicitly scaled E4M3 KV."""
    import torch
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends.mla import rocm_aiter_mla_sparse as backend
    from ..attn.glm_sparse_fp8 import rocm_sparse_attn_prefill
    helper = backend.AiterMLAHelper
    if isinstance(q, tuple):
        nope, rope = q
        if rope.shape[-1] != 0:
            raise ValueError('GLM FP8 adapter requires rope-free MLA')
        q = self.q_concat_buffer[:nope.shape[0]]
        q.copy_(nope)
    if q.dtype != torch.bfloat16 or q.shape[-1] != 512:
        raise ValueError('GLM FP8 adapter requires BF16 Q with latent dimension 512')
    topk = backend.fit_kpool_indices_to_aiter(
        self.topk_indices_buffer[:meta.num_actual_tokens], meta.topk_tokens)
    backend.triton_convert_req_index_to_global_index(
        meta.req_id_per_token, meta.block_table, topk,
        meta.paged_kv_indptr, meta.paged_kv_indices,
        BLOCK_SIZE=meta.block_size, NUM_TOPK_TOKENS=meta.topk_tokens)
    kv = kv_cache.view(current_platform.fp8_dtype())
    q = helper.get_mla_padded_q(self.num_heads, q)
    output = torch.empty((q.shape[0], q.shape[1], self.kv_lora_rank),
                         dtype=meta.attn_out_dtype, device=q.device)
    sinks = None
    if self.sinks is not None:
        sinks = helper.get_mla_padded_q(self.num_heads,
            self.sinks.reshape(1, self.num_heads, 1), q.shape[1]).reshape(-1)
    rocm_sparse_attn_prefill(q=q, kv=kv.view(-1, 1, q.shape[-1]), indices=None,
        scale=self.scale, head_dim=512, nope_head_dim=512, rope_head_dim=0,
        attn_sink=sinks, output=output, ragged_indices=meta.paged_kv_indices,
        ragged_indptr=meta.paged_kv_indptr, kv_scale=layer._k_scale)
    return helper.get_mla_unpadded_o(self.num_heads, output), None


def patch():
    import os
    enabled = os.environ.get('R9K_GLM_FP8_SPARSE', '0')
    if enabled == '0':
        return False
    if enabled != '1':
        raise ValueError('R9K_GLM_FP8_SPARSE must be 0 or 1')
    import hashlib
    from pathlib import Path
    import torch
    from vllm.platforms import current_platform
    from .gate import check, vllm_commit
    if (not current_platform.is_rocm() or not vllm_commit().startswith('e97573215')
            or torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(':')[0] != 'gfx1201'):
        raise RuntimeError('GLM FP8 adapter requires gfx1201 and audited vLLM e97573215')
    from vllm.v1.attention.backends.mla import rocm_aiter_mla_sparse as backend
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as ops
    # Reject already source-patched images too: do not stack two integrations.
    for module, expected in [(backend, '7444de2f0a2e85f74c5e06e6c2109b290936b32167594bb8a2583c72393e9e83'),
                             (ops, '40fccd0466ed53e26c618a5f85b0c120c5ea3ff83c0423027010857bea81c984')]:
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
            raise RuntimeError('GLM FP8 adapter requires unmodified pinned vLLM sources; revalidate this image')
    impl = backend.ROCMAiterMLASparseImpl
    if getattr(impl.forward_mqa, '_r9700_glm_fp8', False):
        return True
    original_init, original_forward = impl.__init__, impl.forward_mqa
    @wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        cfg = backend.get_current_vllm_config()
        self._r9700_glm_fp8_instance = eligible(cfg.model_config,
            self.kv_cache_dtype, self.num_heads, self.head_size, self.kv_lora_rank)
    @wraps(original_forward)
    def forward(self, q, kv_cache, meta, layer):
        if getattr(self, '_r9700_glm_fp8_instance', False):
            return forward_fp8(self, q, kv_cache, meta, layer)
        return original_forward(self, q, kv_cache, meta, layer)
    forward._r9700_glm_fp8 = True
    builder = backend.ROCMAiterMLASparseMetadataBuilder
    builder.build = _private_build(builder.build)
    impl.__init__, impl.forward_mqa = init, forward
    check('glm_sparse_fp8')
    from vllm.logger import init_logger
    init_logger('vllm.r9700_vllm').info('r9700: GLM FP8 sparse MLA plugin enabled; upstream files unchanged')
    return True
