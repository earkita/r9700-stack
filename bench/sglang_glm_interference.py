#!/usr/bin/env python3
"""C4 client primitives: three active decodes followed by one long prefill.

Not a launcher and not yet a qualified C4 benchmark. A frozen controller must
prepare exact tagged corpora, runtime/graph/memory gates, cache population,
all-eight-GPU PCIe evidence, repetitions and rollback before invoking this.
Generated-code evaluation runs only after every model future is terminal.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import threading
import time
import subprocess
import urllib.request

from sglang_glm_completed_code import evaluate, extract_code
from sglang_glm_prefill import save_evidence, watch_links
from sglang_cache_evidence import native_cache_source
from sglang_latency_report import latencies
from sglang_spec_decode import completion_observation, counter_metrics


class ArrivalGate:
    """Track actual starts, first-token callbacks and terminal requests."""
    def __init__(self, short_lanes, clock=time.monotonic):
        if len(short_lanes)!=3 or len(set(short_lanes))!=3:
            raise ValueError("Declare exactly three independent short lanes")
        self.short_lanes=tuple(short_lanes)
        self.clock=clock
        self.condition=threading.Condition()
        self.first_seen=set()
        self.finished=set()
        self.errors={}
        self.events=[]

    def event(self, kind, lane):
        with self.condition:
            if kind=="long_send":
                if self.first_seen!=set(self.short_lanes) or self.finished or self.errors:
                    raise RuntimeError("Cannot form cohort: a short decode is not active at long send")
            elif kind=="first":
                if lane not in self.short_lanes:
                    raise ValueError("Unknown short lane")
                self.first_seen.add(lane)
            elif kind=="finish":
                self.finished.add(lane)
            self.events.append(dict(kind=kind,lane=lane,monotonic=self.clock()))
            self.condition.notify_all()

    def failure(self,lane,error):
        with self.condition:
            self.errors[lane]=repr(error)
            self.condition.notify_all()

    def wait_for_first_tokens(self,timeout):
        deadline=self.clock()+timeout
        with self.condition:
            while self.first_seen!=set(self.short_lanes):
                if self.finished or self.errors:
                    raise RuntimeError("Short decode finished or failed before all three first tokens")
                remaining=deadline-self.clock()
                if remaining<=0:
                    raise TimeoutError("First-token barrier did not form the declared cohort")
                self.condition.wait(min(remaining,.5))
            if self.finished or self.errors:
                raise RuntimeError("Short decode finished before long request could start")


def run_cohort(bodies,out,transport,abort_owned,first_timeout=120):
    """Transport invokes on_send immediately before HTTP and on_first once.

    No retries or evaluator work while generation is active. On failure abort
    only the four explicitly owned request IDs. Never abort_all or touch another
    deployment. A completed early short response cannot silently pass as C4.
    """
    if len(bodies)!=4 or set(bodies)!={"short0","short1","short2","long"}:
        raise ValueError("Expected short0/1/2 and long")
    ids=[body.get("rid") for body in bodies.values()]
    if any(not isinstance(rid,str) or not rid for rid in ids) or len(set(ids))!=4:
        raise ValueError("Four distinct owned request IDs are required")
    gate=ArrivalGate(("short0","short1","short2"))
    results={};errors={}
    def lane(name):
        try:
            return transport(bodies[name],out/name,
                lambda:gate.event("long_send" if name=="long" else "send",name),
                (lambda:gate.event("first",name)) if name!="long" else (lambda:None))
        except Exception as error:
            gate.failure(name,error)
            raise
        finally:
            gate.event("finish",name)
    pool=ThreadPoolExecutor(max_workers=4)
    futures={}
    try:
        for name in gate.short_lanes:futures[name]=pool.submit(lane,name)
        gate.wait_for_first_tokens(first_timeout)
        futures["long"]=pool.submit(lane,"long")
        names={future:name for name,future in futures.items()}
        # Notice a failed long lane while the short lanes are still generating,
        # rather than blocking on short0 until its completion/read timeout.
        for future in as_completed(names):
            results[names[future]]=future.result()
    except BaseException as error:
        errors["cohort"]=repr(error)
        try:abort_owned(ids)
        except Exception as abort_error:errors["abort_owned"]=repr(abort_error)
    finally:
        # The exclusive controller must treat failed abort/runtime as a failure,
        # retain evidence and clean up its own deployment; not launch a new trial.
        pool.shutdown(wait=True)
        for name,future in futures.items():
            try:results[name]=future.result()
            except BaseException as error:errors[name]=repr(error)
    report=dict(status="error" if errors else "responses_saved",results=results,
                errors=errors,arrival_events=gate.events,
                active_short_at_long_send=any(e["kind"]=="long_send" for e in gate.events))
    save_evidence(out/"cohort-transport.json",report)
    return report


def stream_native(base,body,stem,on_send,on_first,timeout=600):
    """Record continuous native counters, local monotonic and host-wall axes."""
    if stem.with_suffix(".request.json").exists():
        raise RuntimeError("Refusing to overwrite a started lane")
    if body["sampling_params"].get("ignore_eos") is not False:
        raise ValueError("C4 correctness requires natural EOS")
    save_evidence(stem.with_suffix(".request.json"),body)
    save_evidence(stem.with_suffix(".attempt.json"),dict(status="started",rid=body["rid"]))
    request=urllib.request.Request(base+"/generate",json.dumps(body).encode(),{"Content-Type":"application/json"})
    events=[];wall_events=[];last=None;start=time.monotonic()
    try:
        on_send()
        with urllib.request.urlopen(request,timeout=timeout) as response,stem.with_suffix(".sse").open("w",buffering=1) as raw:
            for line in response:
                elapsed=time.monotonic()-start;wall=time.time();text=line.decode()
                raw.write(json.dumps(dict(seconds=elapsed,wall=wall,line=text))+"\n")
                if elapsed>timeout:raise TimeoutError("Native lane exceeded its total time budget")
                if not text.startswith("data:") or text[5:].strip()=="[DONE]":continue
                last=json.loads(text[5:]);count=last.get("meta_info",{}).get("completion_tokens",0)
                if type(count) is not int or count<0 or (events and count<events[-1][1]):
                    raise ValueError("Invalid/decreased cumulative completion count")
                if count and (not events or count>events[-1][1]):
                    events.append((elapsed,count));wall_events.append((wall,count))
                    if len(events)==1:on_first()
        if last is None:raise RuntimeError("No native response")
        save_evidence(stem.with_suffix(".response.json"),last)
        meta=last["meta_info"]
        if meta.get("id")!=body["rid"] or meta.get("prompt_tokens")!=len(body["input_ids"]):
            raise RuntimeError("Request ID or actual context count mismatch")
        completion=completion_observation(meta,body["sampling_params"]["max_new_tokens"])
        if not events or completion["actual_tokens"]!=events[-1][1]:
            raise RuntimeError("Native final usage does not match stream counters")
        # A very short completion is retained with unknown decode/ITL, not an
        # invented rate. The independent completed-code gate still applies.
        metrics=counter_metrics(events) if len(events)>=2 else None
        row=dict(meta_info=meta,completion=completion,metrics=metrics,
                 latency=latencies(events) if len(events)>=2 else None,
                 ttft_seconds=events[0][0],wall_events=wall_events,wall_seconds=time.monotonic()-start)
        save_evidence(stem.with_suffix(".attempt.json"),dict(status="response_saved",rid=body["rid"]))
        return row
    except BaseException as error:
        save_evidence(stem.with_suffix(".attempt.json"),dict(status="error",rid=body["rid"],error=repr(error)))
        raise


def scheduler_interference(results):
    """Host/API overlap and arrival gaps, never labelled GPU kernel overlap."""
    meta=results["long"]["meta_info"]
    begin,end=meta.get("forward_entry_time"),meta.get("prefill_finished_time")
    if (type(begin) not in (int,float) or type(end) not in (int,float)
            or not math.isfinite(begin) or not math.isfinite(end) or end<begin):
        raise ValueError("Long-request scheduler timestamps are absent or invalid")
    short={}
    for name in ("short0","short1","short2"):
        events=results[name]["wall_events"]
        if not events:raise ValueError("No actual short-lane token observations")
        if (any(type(ts) not in (int,float) or not math.isfinite(ts)
                or type(count) is not int or count<=0 for ts,count in events)
                or any(b[0]<a[0] or b[1]<=a[1] for a,b in zip(events,events[1:]))):
            raise ValueError("Short-lane wall clock or native counters are invalid")
        during=[(ts,count) for ts,count in events if begin<=ts<=end]
        gaps=[(b[0]-a[0])*1000 for a,b in zip(events,events[1:]) if a[0]<=end and b[0]>=begin]
        short[name]=dict(arrival_window_overlaps_prefill=events[0][0]<end and events[-1][0]>begin,
            updates_during_prefill=len(during),max_gap_touching_prefill_ms=max(gaps) if gaps else None,
            tokens_between_observed_updates_during_prefill=(during[-1][1]-during[0][1]) if len(during)>=2 else 0,
            whole_request_latency=results[name]["latency"])
    return dict(scheduler_prefill_wall_seconds=end-begin,short=short,
        note="Same-host scheduler wall timestamps and API token arrivals only. Zero progress can be scheduling starvation; arrival overlap does not prove simultaneous GPU kernels. Mixed plain tokens are not all represented by speculative verify counters.")


def evaluate_cohort(transport,cases,out,image,evaluator=evaluate):
    """Caller invokes this only after run_cohort has joined every future."""
    if transport["status"]!="responses_saved":
        raise RuntimeError("Cannot qualify an incomplete/failed transport cohort")
    lanes={"short0","short1","short2","long"}
    if set(transport["results"])!=lanes or set(cases)!=lanes:
        raise ValueError("All four declared lanes and independent task cases are required")
    tags=[cases[name]["session_tag"] for name in lanes]
    if any(not isinstance(tag,str) or not tag for tag in tags) or len(set(tags))!=4:
        raise ValueError("Four distinct session tags are required")
    rows={}
    for name,result in transport["results"].items():
        row=dict(status="incomplete")
        if result["completion"]["natural_eos"]:
            try:
                native=json.loads((out/name).with_suffix(".response.json").read_text())
                code=extract_code(native["text"])
                expected=cases[name]["session_tag"]
                # Assertion is independently added after generated source in the
                # isolated evaluator, never executed on the host filesystem.
                checked=code+"\nassert globals().get('SESSION_TAG') == "+repr(expected)+", 'wrong session tag'\n"
                execution=evaluator(checked,image)
                (out/name).with_suffix(".py").write_text(code+"\n")
                try:passed=execution["exit_code"]==0 and json.loads(execution["stdout"])=={"status":"pass","cases":169}
                except (ValueError,KeyError,TypeError):passed=False
                row.update(status="correct_completed" if passed else "wrong_completed",execution=execution)
            except (ValueError,SyntaxError) as error:row.update(status="wrong_completed",error=repr(error))
            except Exception as error:row.update(status="error",error=repr(error))
        rows[name]=row
    report=dict(status="pass" if len(rows)==4 and all(r["status"]=="correct_completed" for r in rows.values()) else "failed",lanes=rows,
                scope="Four independent tagged dependency implementations, not broad coding or agent quality")
    save_evidence(out/"cohort-quality.json",report)
    return report


def abort_requests(base,ids):
    """Pinned SGLang accepts one RID per abort, with abort_all explicitly false."""
    errors={}
    for rid in ids:
        try:
            request=urllib.request.Request(base+"/abort_request",
                json.dumps(dict(rid=rid,abort_all=False)).encode(),{"Content-Type":"application/json"})
            with urllib.request.urlopen(request,timeout=15) as response:response.read()
        except Exception as error:errors[rid]=repr(error)
    if errors:raise RuntimeError(f"Owned request aborts failed: {errors}")


def require_cache(row,expected):
    meta=row["meta_info"]
    if type(meta.get("cached_tokens")) is not int or meta["cached_tokens"]!=expected:
        raise RuntimeError("Actual cache hit differs from the frozen preparation")
    source=native_cache_source(expected,meta.get("cached_tokens_details"),allow_miss=expected==0)
    if source["cache_source"] not in ("device","none"):
        raise RuntimeError("Only GPU prefix reuse was declared")
    return source


def require_speculation(rows, algorithm):
    """Establish the declared speculative state on every C4 lane."""
    if algorithm not in ("EAGLE", None):
        raise ValueError(f"Unsupported C4 speculative algorithm: {algorithm!r}")
    for name, row in rows.items():
        count = row["meta_info"].get("spec_verify_ct", 0)
        if type(count) is not int or count < 0:
            raise RuntimeError(f"Invalid speculative verify count on {name}")
        if algorithm == "EAGLE" and count == 0:
            raise RuntimeError(f"Declared MTP execution not established on {name}")
        if algorithm is None and count != 0:
            raise RuntimeError(f"Unexpected MTP execution on target-only lane {name}")


def cohort_observations(result):
    """Aggregate on a shared interval; never add per-request token rates."""
    rows=result["results"]
    events=[row["wall_events"] for row in rows.values()]
    begin=min(event[0][0] for event in events);end=max(event[-1][0] for event in events)
    if end<=begin:raise ValueError("No shared arrival interval")
    tokens=sum(event[-1][1]-event[0][1] for event in events)
    return dict(scheduler_interference=scheduler_interference(rows),
        aggregate_post_first_burst_tokens=tokens,aggregate_arrival_seconds=end-begin,
        aggregate_post_first_burst_tps=tokens/(end-begin),
        per_request={name:dict(metrics=row["metrics"],latency=row["latency"],
            completion=row["completion"],wall_seconds=row["wall_seconds"],
            ttft_seconds=row["ttft_seconds"],
            speculation={key:value for key,value in row["meta_info"].items() if key.startswith("spec_")})
            for name,row in rows.items()},
        note="Custom native C4 client. Aggregate spans earliest first burst to latest final burst, subtracting every lane's first burst; not sum of request rates. Scheduler windows include waiting, not pure GPU time. This is a bounded completed-code cohort, not a multi-turn agent task.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    out=parser.parse_args().out
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if sha(out/"protocol.json")!=(out/"protocol.sha256").read_text().strip():
        raise RuntimeError("Frozen C4 protocol changed")
    plan=json.loads((out/"protocol.json").read_text())
    for path,digest in plan["sources"].items():
        if sha(path)!=digest:raise RuntimeError(f"Frozen C4 source changed: {path}")
    if plan["sources"].get(str(Path(__file__).relative_to(Path.cwd())))!=sha(__file__):
        raise RuntimeError("C4 measurement client was not frozen")
    boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if boot_id!=plan["boot_id"]:raise RuntimeError("Reboot requires a fresh protocol/hardware preparation")
    if (out/"attempts.jsonl").exists():raise RuntimeError("Refusing to overwrite a C4 series")
    if plan["excluded_cohorts"]!=1 or plan["measured_cohorts"]!=5 or len(plan["seeds"])!=6:
        raise ValueError("Declare one excluded full cohort and five measured cohorts")
    identity=json.loads(subprocess.check_output(["docker","inspect",plan["container"]],text=True))[0]
    if not identity["State"]["Running"] or identity["Image"]!=plan["image_id"]:
        raise RuntimeError("C4 runtime identity differs from protocol")
    env=dict(item.split("=",1) for item in identity["Config"]["Env"])
    if env.get("ROCP_TOOL_ATTACH")=="1" or env.get("R9K_SGLANG_MTP_TRACE_DIR"):
        raise RuntimeError("Profiling/snapshots cannot qualify C4 latency")
    for key,value in plan["environment"].items():
        if env.get(key)!=value:raise RuntimeError(f"C4 environment differs: {key}")
    def fetch(path,body=None):
        request=urllib.request.Request(plan["base"]+path,
            None if body is None else json.dumps(body).encode(),{"Content-Type":"application/json"})
        with urllib.request.urlopen(request,timeout=30) as response:return response.read()
    info=json.loads(fetch("/get_server_info"));args=info.get("server_args",info)
    for key,value in plan["runtime"].items():
        if args.get(key)!=value:raise RuntimeError(f"C4 server option differs: {key}")
    if args["max_running_requests"]!=4 or args["max_mamba_cache_size"]<20:
        raise RuntimeError("C4 active/KDA-state qualification bounds are absent")
    corpus_path=out/plan["corpus"]["file"]
    if sha(corpus_path)!=plan["corpus"]["sha256"]:raise RuntimeError("C4 corpus changed")
    cases=json.loads(corpus_path.read_text())["lanes"]
    if set(cases)!={"short0","short1","short2","long"}:raise ValueError("Four exact C4 cases required")
    tags=[case["session_tag"] for case in cases.values()]
    if any(not isinstance(tag,str) or not tag for tag in tags) or len(set(tags))!=4:
        raise ValueError("Four independent task session tags are required before GPU work")
    if type(plan["short_prefix_tokens"]) is not int or plan["short_prefix_tokens"]<=0:
        raise ValueError("Positive short-prefix preparation length is required")
    if any(cases[name]["input_tokens"]<=plan["short_prefix_tokens"] for name in ("short0","short1","short2")):
        raise ValueError("Short requests need a new tail beyond the prepared prefix")
    sampling=plan["sampling"]
    if sampling.get("ignore_eos") is not False or sampling["max_new_tokens"]!=4096:
        raise ValueError("The declared completed-code budget is natural EOS, at most 4096 tokens")
    if any(len(case["input_ids"])!=case["input_tokens"] for case in cases.values()):
        raise ValueError("Exact native input lengths differ from corpus")
    required=sum(case["input_tokens"]+sampling["max_new_tokens"] for case in cases.values())+plan["lookahead_reserve_tokens"]
    if args["max_total_num_tokens"]<required:
        raise RuntimeError(f"Measured GPU KV capacity {args['max_total_num_tokens']} is below declared cohort reserve {required}")
    # This pinned SGLang build reserves four framing tokens when admitting a
    # native input_ids request, even though prompt_tokens reports the supplied
    # ID count. The first C4 warmup was rejected at 131072+4+4096 > 135170.
    if any(case["input_tokens"]+sampling["max_new_tokens"]+4>args["context_length"] for case in cases.values()):
        raise RuntimeError("Full natural-EOS budget does not fit per-request context")
    spec=importlib.util.spec_from_file_location("c4_hardware",Path(__file__).parent/"baseline-earkita/hardware.py")
    hw=importlib.util.module_from_spec(spec);spec.loader.exec_module(hw)
    mapping=hw.hip_mapping(plan["container"])
    if mapping!=plan["mapping"]:raise RuntimeError("C4 HIP/worker/BDF mapping changed")
    idle=hw.capture(out/"pcie-before",mapping)
    save_evidence(out/"server-info.json",info)
    samples=[];observer_errors=[];done=threading.Event()
    watcher=threading.Thread(target=watch_links,args=(lambda:hw.links(mapping),
        out/"loaded-links.journal.jsonl",done,samples,observer_errors,boot_id));watcher.start()
    rows=[];failure=None
    try:
        for index,seed in enumerate(plan["seeds"]):
            row=dict(index=index,excluded_warmup=index==0,status="error")
            folder=out/f"cohort-{index}";folder.mkdir()
            try:
                if observer_errors:raise RuntimeError(f"PCIe observer failed: {observer_errors}")
                save_evidence(folder/"flush.json",dict(response=fetch("/flush_cache?timeout=5",{}).decode()))
                requests={name:dict(input_ids=case["input_ids"],stream=True,
                    sampling_params=sampling|dict(sampling_seed=seed+lane),
                    rid=f"{plan['run_id']}-{index}-{name}",cache_salt=f"{plan['cache_salt']}-{index}-{name}")
                    for lane,(name,case) in enumerate(cases.items())}
                for name in ("short0","short1","short2"):
                    population=requests[name]|dict(input_ids=cases[name]["input_ids"][:plan["short_prefix_tokens"]],
                        rid=requests[name]["rid"]+"-population",
                        sampling_params=requests[name]["sampling_params"]|dict(max_new_tokens=1))
                    populated=stream_native(plan["base"],population,folder/(name+"-population"),lambda:None,lambda:None)
                    require_cache(populated,0)
                    save_evidence(folder/(name+"-population.metrics.json"),populated)
                result=run_cohort(requests,folder,
                    lambda body,stem,on_send,on_first:stream_native(plan["base"],body,stem,on_send,on_first,timeout=plan["request_timeout_seconds"]),
                    lambda ids:abort_requests(plan["base"],ids),first_timeout=plan["first_token_timeout_seconds"])
                if result["status"]!="responses_saved":raise RuntimeError("C4 cohort formation/transport failed; no selected retry")
                row["cache"]={name:require_cache(value,0 if name=="long" else plan["short_prefix_tokens"])
                    for name,value in result["results"].items()}
                require_speculation(result["results"],plan["runtime"]["speculative_algorithm"])
                row["observations"]=cohort_observations(result)
                # All HTTP futures have joined before starting isolated CPU work.
                row["quality"]=evaluate_cohort(result,cases,folder,plan["image_id"])
                row["status"]="pass" if row["quality"]["status"]=="pass" else "quality_failed"
                if observer_errors:raise RuntimeError(f"PCIe observer failed: {observer_errors}")
            except BaseException as error:
                row.update(status="error",error=repr(error));raise
            finally:
                with (out/"attempts.jsonl").open("a") as handle:
                    handle.write(json.dumps(row)+"\n");handle.flush();os.fsync(handle.fileno())
                rows.append(row);print(json.dumps(dict(index=index,status=row["status"])),flush=True)
    except BaseException as error:failure=repr(error)
    finally:
        done.set();watcher.join()
        try:
            if observer_errors:raise RuntimeError(f"PCIe observer failed: {observer_errors}")
            hw.check_loaded(samples,idle,hw.expected_links(idle))
            (out/"pcie-status.txt").write_text("PASS all eight GPUs idle/loaded\n")
        except BaseException as error:failure=repr(error) if failure is None else failure+"; "+repr(error)
        save_evidence(out/"status.json",dict(status="error" if failure else "pass" if len(rows)==6 and all(r["status"]=="pass" for r in rows) else "quality_failed",
            error=failure,cohorts=len(rows),measured_cohorts=sum(not r["excluded_warmup"] for r in rows),
            scope="C4 tagged completed-code and scheduler-interference series; no C1 speed comparison or broad agent qualification"))
    if failure or len(rows)!=6 or any(row["status"]!="pass" for row in rows):raise SystemExit(1)


if __name__=="__main__":main()
