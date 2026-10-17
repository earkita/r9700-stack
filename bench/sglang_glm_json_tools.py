#!/usr/bin/env python3
"""Two-turn tools+response_format qualification; keep schema on BOTH turns.

The secret exists only in the virtual read_file result, never in the initial
prompt or schema. This tests the actual API combination from SGLang #42269,
not the final-only-schema workaround. No timing/quality comparison claim.
The frozen controller provides all-eight-GPU idle/loaded/runtime evidence.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import time
import urllib.request

from sglang_glm_multitool import post, strict_json


TOOL = {"type":"function","function":{"name":"read_file","description":"Read a virtual file.",
    "parameters":{"type":"object","properties":{"path":{"type":"string","enum":["note.txt"]}},
        "required":["path"],"additionalProperties":False}}}
SCHEMA = {"type":"object","properties":{"word":{"type":"string"}},
          "required":["word"],"additionalProperties":False}


def response_format(kind):
    if kind=="json_object":
        return {"type":"json_object"}
    if kind=="json_schema":
        return {"type":"json_schema","json_schema":{"name":"file_secret","strict":True,"schema":SCHEMA}}
    raise ValueError("Unqualified response_format")


def check_call(reply):
    calls=reply.get("tool_calls") or []
    if len(calls)!=1 or reply.get("finish_reason")!="tool_calls":
        raise ValueError("Expected one read_file call before any secret answer")
    call=calls[0]
    if (not isinstance(call.get("id"),str) or not call["id"]
            or call.get("name")!="read_file" or strict_json(call["arguments"])!={"path":"note.txt"}):
        raise ValueError("Wrong tool identity, name or arguments")
    return call


def attempt(plan,case,out,transport=post):
    # Exact same case secret/prompt/seed/schema in the control/candidate legs.
    common=dict(model=plan["model"],tools=[TOOL],tool_choice="auto",parallel_tool_calls=False,
        response_format=response_format(case["format"]),temperature=0.0,top_p=1.0,top_k=-1,
        max_tokens=512,seed=case["seed"],stream=case["stream"],cache_salt=plan["cache_salt"]+"-"+case["id"],
        chat_template_kwargs={"reasoning_effort":"high","enable_thinking":case["thinking"],"clear_thinking":False})
    if case["stream"]:common["stream_options"]={"include_usage":True}
    messages=[{"role":"user","content":"Use read_file to read note.txt. The secret word is available "
        'only in that tool result. Then return exactly {"word": "<the actual secret>"}, with no other text. '
        "Read the file once; after receiving its result, answer instead of calling it again."}]
    started=time.perf_counter();row=dict(case=case["id"],seed=case["seed"],format=case["format"],
        stream=case["stream"],thinking_requested=case["thinking"],status="incomplete",tool_calls=0)
    try:
        first=transport(plan["base"],common|dict(messages=messages),out/"first")
        row["first_finish"]=first.get("finish_reason")
        if row["first_finish"]=="length":
            row["error"]="Initial turn truncated";return row
        if row["first_finish"] not in ("stop","tool_calls"):
            raise RuntimeError("Initial runtime abort/missing finish reason")
        if row["first_finish"]=="stop":
            row.update(status="missing_tool_call",error="Model answered before receiving the secret");return row
        call=check_call(first);row["tool_calls"]=1
        assistant=dict(role="assistant",content=first["content"] or None,tool_calls=[dict(
            id=call["id"],type="function",function=dict(name=call["name"],arguments=call["arguments"]))])
        if first.get("reasoning"):assistant["reasoning_content"]=first["reasoning"]
        result=dict(role="tool",tool_call_id=call["id"],content=json.dumps({"word":case["secret"]}))
        # Preserve response_format and auto here: otherwise this would test only
        # the existing workaround and could falsely claim the API bug is fixed.
        final=transport(plan["base"],common|dict(messages=messages+[assistant,result]),out/"final")
        row["final_finish"]=final.get("finish_reason")
        if row["final_finish"]=="length":
            row["error"]="Final turn truncated";return row
        if row["final_finish"]=="tool_calls":
            row.update(status="looped_tool_call",error="Repeated tool call instead of final answer",final=final);return row
        if row["final_finish"]!="stop":raise RuntimeError("Final runtime abort/missing finish reason")
        if final.get("tool_calls"):raise ValueError("Tool data present in a normal final answer")
        value=strict_json(final["content"])
        row.update(status="pass" if value=={"word":case["secret"]} else "wrong_completed",final=final)
    except (ValueError,KeyError,TypeError) as error:
        row.update(status="contract_failure",error=repr(error))
    except Exception as error:
        row.update(status="error",error=repr(error))
    finally:
        row["wall_seconds"]=time.perf_counter()-started
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out",type=Path,required=True)
    out=parser.parse_args().out
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if (out/"attempts.jsonl").exists():raise RuntimeError("Refusing to overwrite attempts")
    if sha(out/"protocol.json")!=(out/"protocol.sha256").read_text().strip():raise RuntimeError("Protocol changed")
    plan=json.loads((out/"protocol.json").read_text())
    expected=set(itertools.product(("json_object","json_schema"),(False,True),(False,True),(1234,5678,9012)))
    actual={(case["format"],case["stream"],case["thinking"],case["seed"]) for case in plan["cases"]}
    if len(plan["cases"])!=24 or actual!=expected or len({case["id"] for case in plan["cases"]})!=24:
        raise RuntimeError("The whole frozen 24-case format/stream/thinking/seed cohort is required")
    for file,digest in plan["sources"].items():
        if sha(file)!=digest:raise RuntimeError("Frozen source changed: "+file)
    if plan["sources"].get("bench/sglang_glm_json_tools.py")!=sha(__file__):raise RuntimeError("Runner not frozen")
    image=subprocess.check_output(["docker","inspect",plan["container"],"--format","{{.Image}}"],text=True).strip()
    if image!=plan["image_id"]:raise RuntimeError("Wrong image")
    with urllib.request.urlopen(plan["base"]+"/get_server_info",timeout=30) as response:info=json.load(response)
    runtime=info.get("server_args",info)
    for key,value in plan["runtime"].items():
        if runtime.get(key)!=value:raise RuntimeError("Runtime mismatch: "+key)
    env=json.loads(subprocess.check_output(["docker","inspect",plan["container"],"--format","{{json .Config.Env}}"],text=True))
    env=dict(value.split("=",1) for value in env)
    if env.get("R9K_SGLANG_GLM_JSON_TOOLS","0")!=plan["feature_flag"]:raise RuntimeError("Wrong feature flag")
    (out/"server-info.json").write_text(json.dumps(info,indent=2)+"\n")
    rows=[]
    for case in plan["cases"]:
        folder=out/case["id"];folder.mkdir()
        row=attempt(plan,case,folder);rows.append(row)
        with (out/"attempts.jsonl").open("a") as file:file.write(json.dumps(row)+"\n");file.flush()
        print(json.dumps({key:row.get(key) for key in ("case","status","tool_calls","error")}),flush=True)
        if row["status"]=="error":raise RuntimeError("Runtime/transport error; stop further requests")
    counts={name:sum(row["status"]==name for row in rows) for name in (
        "pass","missing_tool_call","looped_tool_call","wrong_completed","incomplete","contract_failure","error")}
    record=dict(status="pass" if counts["pass"]==len(plan["cases"]) else "failed",attempts=len(rows),counts=counts,
        scope="Bounded tools+response_format API contract, not sampling-quality or throughput",
        schema_on_both_turns=True,tool_choice_auto_on_both_turns=True,
        hardware_evidence="Provided and checked by the frozen exclusive controller; this runner does not qualify hardware itself")
    (out/"status.json").write_text(json.dumps(record,indent=2)+"\n")
    if record["status"]!="pass":raise SystemExit(1)


if __name__=="__main__":main()
