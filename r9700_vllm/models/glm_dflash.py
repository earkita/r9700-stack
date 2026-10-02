"""GLM DFlash auxiliary outputs via layer hooks, preserving upstream forward.

Completed mHC state semantics follow vllm-project/vllm#55423 (zixi-qi,
Leoyzen, Shijin Zhang). No target state is replaced or modified in place.
"""
from types import FunctionType
import os

from vllm.model_executor.models.interfaces import EagleModelMixin, SupportsEagle3
from vllm.models.glm5next.common import model as glm


class GlmDFlashModel(glm.Glm5NextModel, EagleModelMixin):
    def _set_aux_hidden_state_layers(self, layers):
        if self.start_layer != 0 or self.end_layer != len(self.layers):
            raise NotImplementedError("GLM DFlash adapter is qualified only without pipeline parallelism")
        if any(i <= 0 or i > self.end_layer for i in layers):
            raise ValueError("GLM DFlash taps must select completed decoder layers")
        for handle in getattr(self, "_r9k_aux_hooks", ()):
            handle.remove()
        EagleModelMixin._set_aux_hidden_state_layers(self, layers)
        self._r9k_aux_hooks = [self.layers[i - 1].register_forward_hook(self._capture_aux)
                               for i in self.aux_hidden_state_layers]

    def _capture_aux(self, layer, inputs, output):
        hidden, residual, post, comb = output
        aux = hidden
        if post is not None:
            aux = glm.hc_contract(layer.hc_post(hidden, residual, post, comb), layer.n)
        if os.environ.get("R9K_GLM_DFLASH_AUDIT_DIR"):
            from ..compat.glm_dflash_audit import capture_aux
            capture_aux(self, layer, output, aux)
        if self.is_sequence_parallel:
            aux = glm.sp_all_gather(aux)[:self._r9k_full_tokens]
        self._r9k_aux.append(aux)

    def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds=None, **kwargs):
        if not self.aux_hidden_state_layers:
            return super().forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)
        self._r9k_aux = []
        self._r9k_full_tokens = positions.shape[0]
        result = super().forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)
        aux = self._r9k_aux
        self._r9k_aux = []
        if len(aux) != len(self.aux_hidden_state_layers):
            raise RuntimeError("GLM DFlash did not capture every requested auxiliary layer")
        return result, aux


# Change only the constructed core class in the inherited initializer. The
# upstream function's zero-argument super closure still names its real base.
_original_init = glm.Glm5NextForCausalLM.__init__
_init = FunctionType(_original_init.__code__,
                     _original_init.__globals__ | {"Glm5NextModel": GlmDFlashModel},
                     _original_init.__name__, _original_init.__defaults__, _original_init.__closure__)
_init.__kwdefaults__ = _original_init.__kwdefaults__


class R9kGlmDFlashForCausalLM(glm.Glm5NextForCausalLM, SupportsEagle3):
    __init__ = _init


@glm.MULTIMODAL_REGISTRY.register_processor(
    glm.Glm5NextMultiModalProcessor, info=glm.Glm5NextProcessingInfo,
    dummy_inputs=glm.Glm4vDummyInputsBuilder,
)
class R9kGlmDFlashForConditionalGeneration(glm.Glm5NextForConditionalGeneration, SupportsEagle3):
    pass
