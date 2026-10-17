"""CPU-only prototype schema and grammar contracts; no model or GPU work."""
import importlib.util
from pathlib import Path

import pytest

path=Path(__file__).parents[1]/"r9700_sglang/glm_json_tools.py"
spec=importlib.util.spec_from_file_location("glm_json_tools",path)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def test_wire_alias_and_actual_schema_are_preserved():
    class Format:
        def model_dump(self,**kwargs):
            assert kwargs=={"by_alias":True,"exclude_none":True}
            return {"type":"json_schema","json_schema":{"schema":{"type":"object",
                "properties":{"schema_":{"type":"string"}},"required":["schema_"],
                "additionalProperties":False}}}
    schema=mod.response_schema(Format())
    assert schema["properties"]=={"schema_":{"type":"string"}}
    assert schema["required"]==["schema_"]
    with pytest.raises(ValueError,match="schema is required"):
        mod.response_schema({"type":"json_schema","json_schema":{"schema_":{"type":"object"}}})


@pytest.mark.parametrize("schema",[
    {"type":"object","properties":{"a":{"type":"string","pattern":"x"}}},
    {"type":"integer","minimum":0},
    {"$ref":"https://example.invalid/schema.json"},
    {"type":"array","items":{"oneOf":[{"type":"string"},{"type":"integer"}]}},
    {"type":"object","unevaluatedProperties":False}])
def test_unqualified_assertions_never_degrade_to_any_object(schema):
    with pytest.raises(ValueError):
        mod.response_schema({"type":"json_schema","json_schema":{"schema":schema}})


def test_schema_annotation_and_property_names_are_not_validation_keywords():
    schema={"type":"object","title":"word","description":"title is not an assertion",
            "properties":{"minimum":{"type":["string","null"]}},
            "required":["minimum"],"additionalProperties":False}
    assert mod.response_schema({"type":"json_schema","json_schema":{"schema":schema}})==schema


@pytest.mark.parametrize("value",[None,{}, {"type":"text"},
    {"type":"json_schema","json_schema":{"schema":{"type":"nonexistent"}}},
    {"type":"json_schema","json_schema":{"schema":{"enum":[float('nan')]}}}])
def test_invalid_format_and_schema_fail_explicitly(value):
    with pytest.raises(Exception): mod.response_schema(value)


def test_instruction_contains_schema_not_an_expected_answer():
    instruction=mod.final_format_instruction({"type":"json_schema","json_schema":{"schema":{
        "type":"object","properties":{"word":{"type":"string"}},"required":["word"]}}})
    assert '"word"' in instruction and "results" in instruction
    assert "quokka" not in instruction


@pytest.mark.parametrize("prompt,thinking,expected",[
    ("<|assistant|><think>",False,True),
    ("<|assistant|><think></think>",False,False),
    ("<|assistant|>",False,False),
    ("<|assistant|><think>",True,False),
    ("<|assistant|><think>",None,False),
])
def test_disabled_thinking_closes_only_a_real_open_template_prefix(prompt,thinking,expected):
    assert mod.thinking_prefix_needs_close(prompt,thinking) is expected


def test_disabled_thinking_rejects_unknown_rendered_layout():
    with pytest.raises(ValueError,match="rendered text prompt"):
        mod.thinking_prefix_needs_close([1,2,3],False)


def test_text_renderer_uses_only_bounded_authoritative_token_suffix():
    class Tokenizer:
        def decode(self,ids,**kwargs):
            assert ids==list(range(8184,8200))
            assert kwargs=={"skip_special_tokens":False}
            return "<|assistant|><think>"
    assert mod.thinking_prefix_needs_close("",False,prompt_ids=list(range(8200)),tokenizer=Tokenizer())


@pytest.mark.parametrize("ids,tokenizer",[([],object()),([1],None),("unrendered",object())])
def test_invalid_rendered_ids_never_silently_use_an_empty_prompt(ids,tokenizer):
    with pytest.raises(ValueError,match="rendered token IDs"):
        mod.thinking_prefix_needs_close("",False,prompt_ids=ids,tokenizer=tokenizer)
