#!/usr/bin/env python3
"""Frozen C1 target/MTP comparison using native cumulative token counters.

Streaming bursts are not tokens. Subtract the actual token count in the first
update, not an assumed one-token SSE chunk. Both comparison legs must use
this client and the same frozen input IDs, sampling and cache preparation.
Never launches/restarts a model. Raw outputs stay in the supplied result dir.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics
import subprocess
import time
import urllib.request


def counter_metrics(events):
    if len(events) < 2 or events[-1][0] <= events[0][0]:
        raise ValueError("Need at least two distinct cumulative token updates")
    if any(b[1] <= a[1] for a,b in zip(events,events[1:])):
        raise ValueError("Generation counters must increase")
    return dict(first_update_tokens=events[0][1],last_update_tokens=events[-1][1],
                post_first_update_tokens=events[-1][1]-events[0][1],
                decode_seconds=events[-1][0]-events[0][0],
                decode_tps=(events[-1][1]-events[0][1])/(events[-1][0]-events[0][0]),
                ttft_seconds=events[0][0])


def completion_observation(meta, budget):
    """Natural EOS is valid; aborts, missing counters and retractions are not."""
    actual=meta.get('completion_tokens');finish=(meta.get('finish_reason') or {}).get('type')
    if type(actual) is not int or not 1 <= actual <= budget or finish not in ('stop','length'):
        raise ValueError('Invalid actual completion count, abort or finish reason')
    if type(meta.get('num_retractions')) is not int or meta['num_retractions'] != 0:
        raise ValueError('Retraction or missing runtime counter')
    if finish == 'length' and actual != budget:
        raise ValueError('Length termination before declared output budget')
    return dict(actual_tokens=actual,budget_tokens=budget,
                budget_truncated=finish=='length',natural_eos=finish=='stop')


def audit_native_markers(text, finish):
    """Keep a normal terminal stop marker, reject embedded role continuations."""
    found=[]
    for marker in ('<|user|>','<|endoftext|>','<|assistant|>'):
        positions=[m.start() for m in re.finditer(re.escape(marker),text)]
        found.extend(dict(marker=marker,position=pos) for pos in positions)
    for item in found:
        if (finish != 'stop' or item['marker']=='<|assistant|>' or len(found)!=1
                or text[item['position']:].strip()!=item['marker']):
            raise ValueError('Embedded role/stop marker; continuation is not representative')
    return found


def get(base,path,body=None):
    req=urllib.request.Request(base+path,None if body is None else json.dumps(body).encode(),
                               {'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=30) as r:return json.load(r)


def stream(base,body,path):
    path.with_suffix('.request.json').write_text(json.dumps(body)+'\n')
    request=urllib.request.Request(base+'/generate',json.dumps(body).encode(),{'Content-Type':'application/json'})
    start=time.perf_counter();events=[];last=None
    with path.with_suffix('.sse').open('w') as raw, urllib.request.urlopen(request,timeout=1800) as response:
        for line in response:
            elapsed=time.perf_counter()-start;text=line.decode()
            raw.write(json.dumps(dict(seconds=elapsed,line=text))+'\n')
            if not text.startswith('data:'):continue
            data=text[5:].strip()
            if data=='[DONE]':break
            last=json.loads(data);meta=last.get('meta_info',{});count=meta.get('completion_tokens',0)
            if count and (not events or count>events[-1][1]):events.append((elapsed,count))
    if last is None:raise RuntimeError('No native generation data')
    path.with_suffix('.response.json').write_text(json.dumps(last,ensure_ascii=False)+'\n')
    metrics=counter_metrics(events);metrics.update(wall_seconds=time.perf_counter()-start,
        prompt_tokens=last['meta_info']['prompt_tokens'],meta_info=last['meta_info'],events=events)
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    out=parser.parse_args().out;raw=(out/'protocol.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=(out/'protocol.sha256').read_text().strip():raise RuntimeError('Protocol changed')
    plan=json.loads(raw);base=plan['base'];name=plan['container']
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=plan['runner_sha256']:
        raise RuntimeError('Measurement client differs from frozen protocol')
    if (out/'requests.jsonl').exists():raise RuntimeError('Previous attempts must not be overwritten')
    corpus=out/plan['corpus']['file']
    if hashlib.sha256(corpus.read_bytes()).hexdigest()!=plan['corpus']['sha256']:raise RuntimeError('Corpus changed')
    case=json.loads(corpus.read_text());ids=case['input_ids']
    if len(ids)!=plan['corpus']['input_tokens']:raise RuntimeError('Input length changed')
    actual=subprocess.check_output(['docker','inspect',name,'--format','{{.Image}}'],text=True).strip()
    if actual!=plan['image_id']:raise RuntimeError('Image mismatch')
    env=json.loads(subprocess.check_output(['docker','inspect',name,'--format','{{json .Config.Env}}'],text=True))
    env=dict(item.split('=',1) for item in env)
    if env.get('R9K_SGLANG_MTP_TRACE_DIR') or env.get('ROCP_TOOL_ATTACH')=='1':
        raise RuntimeError('Diagnostic snapshots/profiling must be disabled for throughput')
    for key,value in plan['expected_environment'].items():
        if env.get(key)!=value:raise RuntimeError(f'Environment mismatch: {key}')
    info=get(base,'/get_server_info');(out/'server-info.json').write_text(json.dumps(info,indent=2))
    args=info.get('server_args',info)
    for key,value in plan['expected_server_args'].items():
        if args.get(key)!=value:raise RuntimeError(f'Runtime mismatch: {key}: {args.get(key)!r} != {value!r}')
    spec=importlib.util.spec_from_file_location('hardware',Path(__file__).parent/'baseline-earkita/hardware.py')
    hw=importlib.util.module_from_spec(spec);spec.loader.exec_module(hw)
    mapping=hw.hip_mapping(name);idle=hw.capture(out/'pcie-idle',mapping);expected=hw.expected_links(idle)
    def loaded():
        values=[]
        for i in range(3):time.sleep(1);values.append(hw.capture(out/f'pcie-loaded-{i}',mapping))
        return values
    excluded=plan.get('excluded_requests',1)
    if type(excluded) is not int or not 1 <= excluded <= 4:
        raise ValueError('Declare bounded excluded cold/population and warmup requests')
    rows=[]
    with ThreadPoolExecutor(max_workers=1) as pool:
        for index in range(excluded+5):
            start=datetime.datetime.now(datetime.timezone.utc).isoformat()
            future=pool.submit(loaded) if index==0 else None
            body=dict(input_ids=ids,sampling_params=plan['sampling_params'],cache_salt=plan['cache_salt'],stream=True)
            row=dict(index=index,excluded_population=index==0,excluded_warmup=index<excluded)
            try:
                metrics=stream(base,body,out/f'request-{index}');row['metrics']=metrics
                if plan.get('audit_native_eos'):
                    if (plan['sampling_params'].get('skip_special_tokens') is not False
                            or plan['sampling_params'].get('ignore_eos') is not False):
                        raise RuntimeError('Natural-EOS audit requires visible markers and EOS enabled')
                    generated=json.loads((out/f'request-{index}.response.json').read_text())['text']
                    row['generated_markers']=audit_native_markers(generated,metrics['meta_info'].get('finish_reason',{}).get('type'))
                if plan.get('forbid_generated_markers'):
                    generated=json.loads((out/f'request-{index}.response.json').read_text())['text']
                    if plan['sampling_params'].get('skip_special_tokens') is not False:
                        raise RuntimeError('EOS audit requires visible special tokens')
                    found=[m for m in plan['forbid_generated_markers'] if m in generated]
                    row['early_end_markers']=found
                    if found:raise RuntimeError(f'Forced continuation after an end marker is not representative: {found}')
                logs=subprocess.check_output(['docker','logs','--since',start,name],stderr=subprocess.STDOUT,text=True)
                (out/f'request-{index}.server.log').write_text(logs)
                hits=[int(n) for n in re.findall(r'#cached-token: (\d+)',logs)]
                row['cached_tokens_observed']=max(hits) if hits else None
                if plan.get('cache_proof')=='native_device':
                    cached=metrics['meta_info'].get('cached_tokens')
                    if type(cached) is not int:raise RuntimeError('Native cache count absent')
                    if cached:
                        detail=metrics['meta_info'].get('cached_tokens_details') or {}
                        if detail.get('device')!=cached or detail.get('host')!=0:
                            raise RuntimeError('Device cache source not established')
                    row['cached_tokens_observed']=cached
                if metrics['prompt_tokens']!=len(ids):raise RuntimeError('Actual prompt length mismatch')
                if plan.get('require_fixed_output',True):
                    if metrics['last_update_tokens']!=plan['sampling_params']['max_new_tokens']:
                        raise RuntimeError('Actual token budget mismatch')
                    if metrics['meta_info'].get('finish_reason',{}).get('type')!='length':raise RuntimeError('Unexpected finish reason')
                else:
                    row['completion']=completion_observation(metrics['meta_info'],plan['sampling_params']['max_new_tokens'])
                    if row['completion']['actual_tokens']!=metrics['last_update_tokens']:
                        raise RuntimeError('Final usage and cumulative stream counters disagree')
                if plan['expected_server_args'].get('speculative_algorithm'):
                    if metrics['meta_info'].get('spec_verify_ct',0)<=0:
                        raise RuntimeError('Declared speculation not established by native counters')
                    row['speculation']={k:metrics['meta_info'][k] for k in
                        ('spec_verify_ct','spec_num_correct_drafts','spec_num_proposed_drafts',
                         'spec_accept_rate','spec_accept_length') if k in metrics['meta_info']}
                if row['cached_tokens_observed'] is None or (index==0 and row['cached_tokens_observed']!=0) or (index>0 and row['cached_tokens_observed']<=0):
                    raise RuntimeError('Cold/warm prefix state not established')
                if future:
                    hw.check_loaded(future.result(),idle,expected);(out/'pcie-status.txt').write_text('PASS all eight GPUs idle/loaded\n')
                row['status']='pass'
            except BaseException as error:
                row.update(status='error',error=repr(error));raise
            finally:
                with (out/'requests.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
            rows.append(row)
    summary={'scope':plan['scope'],'samples':5,'excluded_requests':excluded,'measured':{}}
    for key in ('decode_tps','ttft_seconds','wall_seconds'):
        values=[r['metrics'][key] for r in rows[excluded:]]
        summary['measured'][key]=dict(median=statistics.median(values),minimum=min(values),maximum=max(values),values=values)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
