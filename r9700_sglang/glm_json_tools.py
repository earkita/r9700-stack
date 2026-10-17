# SPDX-License-Identifier: Apache-2.0
"""CPU-only GLM47 tool-or-JSON grammar prototype; not enabled in serving.

Compose the existing thinking/tool rules with XGrammar's JSON Schema frontend.
No string substitution or manual renaming of JSON grammar nonterminals. A
serving adapter must preserve this as full-assistant EBNF instead of replacing
it with a bare response_format constraint. Qualify that adapter separately.
"""
import json


# This initial candidate admits only features exercised by its grammar tests.
# Additional validation keywords must be qualified, never ignored or converted
# into an unconstrained object. Annotation values do not constrain generation.
_ANNOTATIONS = {"$schema", "$id", "$comment", "title", "description", "default",
                "examples", "deprecated", "readOnly", "writeOnly"}
_KEYWORDS = {"$defs", "definitions", "$ref", "type", "properties", "required",
             "additionalProperties", "items", "prefixItems", "enum", "const", "anyOf"}


def validate_schema_features(schema, path="schema"):
    """Fail explicitly for unqualified assertions, including nested schemas."""
    if isinstance(schema, bool):
        return
    if not isinstance(schema, dict):
        raise ValueError(path+" must be an object or boolean JSON Schema")
    unknown = set(schema)-_ANNOTATIONS-_KEYWORDS
    if unknown:
        raise ValueError(path+": unqualified JSON Schema keywords: "+", ".join(sorted(unknown)))
    ref = schema.get("$ref")
    if ref is not None and (not isinstance(ref,str) or not (ref=="#" or ref.startswith("#/"))):
        raise ValueError(path+": only local schema references are qualified")
    for key in ("$defs", "definitions", "properties"):
        if key in schema:
            if not isinstance(schema[key],dict):
                raise ValueError(path+"/"+key+" must be an object")
            for name,child in schema[key].items():
                validate_schema_features(child,path+"/"+key+"/"+name)
    for key in ("items", "additionalProperties"):
        if key in schema:
            validate_schema_features(schema[key],path+"/"+key)
    for key in ("anyOf", "prefixItems"):
        if key in schema:
            if not isinstance(schema[key],list):
                raise ValueError(path+"/"+key+" must be a list")
            for index,child in enumerate(schema[key]):
                validate_schema_features(child,path+"/"+key+"/"+str(index))


def response_schema(response_format):
    """Read the wire alias `schema`, with no schema_ or any-object fallback."""
    if hasattr(response_format,"model_dump"):
        response_format = response_format.model_dump(by_alias=True,exclude_none=True)
    if not isinstance(response_format,dict):
        raise ValueError("A JSON response_format is required")
    if response_format.get("type")=="json_object":
        schema = {"type":"object"}
    elif response_format.get("type")=="json_schema":
        wrapper = response_format.get("json_schema")
        if not isinstance(wrapper,dict) or "schema" not in wrapper:
            raise ValueError("json_schema.schema is required; do not serialize schema_ without aliases")
        schema = wrapper["schema"]
    else:
        raise ValueError("Only json_object and json_schema are qualified")
    # Both validation and the native compiler see the same finite JSON value.
    value = json.dumps(schema,allow_nan=False)
    validate_schema_features(schema)
    from jsonschema import Draft202012Validator
    Draft202012Validator.check_schema(schema)
    return json.loads(value)


def thinking_prefix_needs_close(prompt, enable_thinking, *, prompt_ids=None, tokenizer=None):
    """Use the rendered prefix, since GLM may ignore enable_thinking=False.

    The pinned GLM-5.3 template always ends in <think>. An explicitly disabled
    reasoning grammar must still emit </think> for the output channel parser.
    A template that already closed/omitted the prefix needs no extra marker.
    """
    if enable_thinking is not False:
        return False
    # The pinned text renderer deliberately leaves result.prompt empty. Its
    # token IDs are authoritative. Decode only a short suffix, not a 256K
    # conversation on every cached interaction.
    if prompt_ids is not None:
        if not isinstance(prompt_ids, (list,tuple)) or not prompt_ids or tokenizer is None:
            raise ValueError("Combined GLM tools+JSON requires rendered token IDs and tokenizer")
        prompt = tokenizer.decode(prompt_ids[-16:], skip_special_tokens=False)
    if not isinstance(prompt, str):
        raise ValueError("Combined GLM tools+JSON requires a rendered text prompt")
    return prompt.endswith("<think>")


def compose_grammar(required_tool_ebnf, schema, tool_choice="auto", *, close_thinking_prefix=False):
    """Preserve thinking, then choose required tool calls or exact schema JSON.

    `required_tool_ebnf` must come from generate_glm_grammar(required=True), with
    the actual request's thinking/template and parallel-tool settings. Forced
    tool choices retain only the tool branch. `none` retains only JSON. This
    is not a scheduler, speculative-state, tokenizer or full-model test.
    """
    if tool_choice not in ("auto","required","none"):
        raise ValueError("Resolve a named tool to its required grammar before composition")
    # Apply the same feature gate even when the caller already extracted schema.
    schema = response_schema({"type":"json_schema","json_schema":{"schema":schema}})
    import xgrammar as xgr
    if not all(hasattr(xgr.Grammar,name) for name in ("concat","union","from_ebnf","from_json_schema")):
        raise RuntimeError("The installed XGrammar lacks the required grammar composition API")
    thinking = (xgr.Grammar.from_ebnf('root ::= "</think>"') if close_thinking_prefix
                else xgr.Grammar.from_ebnf(required_tool_ebnf,root_rule_name="thinking_block"))
    tool = xgr.Grammar.from_ebnf(required_tool_ebnf,root_rule_name="tool_call_blocks")
    final = xgr.Grammar.from_json_schema(schema,strict_mode=True)
    branch = (xgr.Grammar.union(tool,final) if tool_choice=="auto"
              else tool if tool_choice=="required" else final)
    return str(xgr.Grammar.concat(thinking,branch))


def final_format_instruction(response_format):
    schema = response_schema(response_format)
    return ("You may use the provided tools when needed. After receiving their results, "
            "finish with a JSON value matching the following schema, without prose or "
            "code fences. Do not repeat a tool call if its result already supplies the "
            "information needed for the final answer. Return the requested answer "
            "data, never the schema definition itself. JSON Schema: "
            +json.dumps(schema,ensure_ascii=False,allow_nan=False))
