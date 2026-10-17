"""Retain SGLang streaming cache/speculation extensions for the agent client.

The completed-task runner and its token/latency arithmetic are reused unchanged.
This wrapper adds explicit response-extension flags, then reads their saved raw
SSE evidence after generation. It never substitutes a final answer or infers a
zero cache count from a missing field. No model launch or reset occurs here.
"""
import json
import math

from sglang_cache_evidence import native_cache_source
from sglang_glm_agent_task import request as base_request


def extension_evidence(records,usage):
    identities=set();extensions={}
    for record in records:
        text=record["line"]
        if not text.startswith("data:") or text[5:].strip()=="[DONE]":continue
        event=json.loads(text[5:])
        if event.get("id") is not None:identities.add(event["id"])
        for key,value in (event.get("sglext") or {}).items():
            if value is None:continue
            if key in extensions and extensions[key]!=value:
                raise ValueError("Conflicting final SGLang extension: "+key)
            extensions[key]=value
    if len(identities)!=1 or any(not isinstance(item,str) or not item for item in identities):
        raise ValueError("One actual backend request ID is required")
    cache=extensions.get("cached_tokens_details")
    count=(usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if cache is not None:
        storage=cache.get("storage")
        if (type(cache.get("device")) is not int or type(cache.get("host")) is not int
                or (storage is not None and (type(storage) is not int or storage!=0))):
            raise ValueError("Qualified device/host counters required; storage was not declared")
        total=cache["device"]+cache["host"]
        if count is not None and (type(count) is not int or count!=total):
            raise ValueError("Usage and native cache-source counters disagree")
        evidence=native_cache_source(total,{"device":cache["device"],"host":cache["host"]},allow_miss=True)
        evidence.update(actual_cached_tokens=total,provenance="native SGLang streaming extension")
    elif count is not None:
        if type(count) is not int or count!=0:
            raise ValueError("Nonzero or invalid cache count has no source evidence")
        evidence=native_cache_source(0,None,allow_miss=True)
        evidence.update(actual_cached_tokens=0,provenance="explicit zero usage counter")
    else:
        evidence=dict(actual_cached_tokens=None,cache_source="unknown",
            provenance="Zero was not emitted; exact request-ID server log is required")
    spec=extensions.get("spec_tokens_details")
    if not isinstance(spec,dict):raise ValueError("Declared speculative execution statistics are absent")
    for key in ("spec_verify_ct","spec_num_correct_drafts","spec_num_proposed_drafts"):
        if type(spec.get(key)) is not int or spec[key]<0:
            raise ValueError("Invalid speculative counter: "+key)
    if spec["spec_num_correct_drafts"]>spec["spec_num_proposed_drafts"]:
        raise ValueError("Accepted drafts exceed proposals")
    for key in ("spec_accept_rate","spec_accept_length"):
        value=spec.get(key)
        if type(value) not in (int,float) or not math.isfinite(value) or value<0:
            raise ValueError("Invalid speculative observation: "+key)
    if spec["spec_accept_rate"]>1:raise ValueError("Invalid speculative acceptance fraction")
    return dict(request_id=next(iter(identities)),cache=evidence,speculation=spec,
        scheduler_prefill_seconds=None,gpu_prefill_seconds=None,
        timing_note="Pinned streaming Chat Completions rejects return_meta_info; TTFT is retained separately, not called pure GPU prompt time")


def request(base,body,stem):
    payload=body|dict(return_cached_tokens_details=True,return_spec_tokens_details=True)
    result,metrics=base_request(base,payload,stem)
    with stem.with_suffix(".sse").open() as handle:
        evidence=extension_evidence((json.loads(line) for line in handle),result["usage"])
    # Keep the original response, counters and model timer exactly as measured.
    # Wrapper parsing time is included in the outer whole-task timer.
    return result,metrics|dict(backend_evidence=evidence)
