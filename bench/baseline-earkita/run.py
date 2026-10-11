#!/usr/bin/env python3
"""Frozen GLM cache/decode matrix. Never starts, stops or replaces a service."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import uuid
from urllib.parse import urlparse
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))
from glm_quality import stream, niah_scores
import hardware


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def http(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    with urlopen(Request(url, data=data, headers={'Content-Type': 'application/json'}), timeout=30) as r:
        return r.read().decode()


def metric(raw, name, *, unobserved_zero=False):
    values = [float(line.rsplit(' ', 1)[1]) for line in raw.splitlines()
              if re.match(re.escape(name) + r'(?:\{| )', line)]
    if not values and f'# TYPE {name} ' not in raw and not unobserved_zero:
        raise RuntimeError('Required metric missing: ' + name)
    return sum(values)


V = 'vllm:'
L = 'lmcache_mp_'
COUNTERS = {
    'gpu_hit': V+'prefix_cache_hits_total',
    'external_hit': V+'external_prefix_cache_hits_total',
    'queries': V+'prefix_cache_queries_total',
    'success': V+'request_success_total',
    'preempt': V+'num_preemptions_total',
    'drafts': V+'spec_decode_num_drafts_total',
    'draft_tokens': V+'spec_decode_num_draft_tokens_total',
    'accepted': V+'spec_decode_num_accepted_tokens_total',
    'l1_hit': L+'lookup_hit_l1_tokens_total',
    'l2_hit': L+'lookup_hit_l2_tokens_total',
    'l2_load': L+'l2_prefetch_load_completed_chunks_total',
    'stores_sent': L+'l2_store_submitted_requests_total',
    'stores_done': L+'l2_store_completed_requests_total',
    'stored_objects': L+'l2_store_completed_objects_chunks_total',
    'gpu_stores_sent': L+'num_submitted_stores_total',
    'gpu_stores_done': L+'num_finished_stores_total',
}

# LMCache 0.5.5 uses OTel event counters: even TYPE is absent until the first
# add(), including add(0). Gauges/vLLM metrics never get this exception.
LAZY_COUNTERS = {name for name in COUNTERS.values() if name.startswith(L)}
SEEN_COUNTERS = set()


def snapshot(p, out=None):
    vr = http(p['base']+'/metrics')
    lr = http(p['cache_metrics']+'/metrics')
    if out:
        out.with_suffix('.vllm.prom').write_text(vr)
        out.with_suffix('.lmcache.prom').write_text(lr)
    result = {}
    for key,name in COUNTERS.items():
        raw = vr if name.startswith(V) else lr
        identity = (p['cache_metrics'],name)
        unobserved = (p.get('versions',{}).get('lmcache')=='0.5.5'
                      and name in LAZY_COUNTERS and identity not in SEEN_COUNTERS)
        result[key] = metric(raw,name,unobserved_zero=unobserved)
        if f'# TYPE {name} ' in raw:
            SEEN_COUNTERS.add(identity)
    result.update(running=metric(vr, V+'num_requests_running'),
                  waiting=metric(vr, V+'num_requests_waiting'),
                  l1_bytes=metric(lr, L+'l1_memory_usage_bytes'))
    for line in vr.splitlines():
        if line.startswith(V+'spec_decode_num_accepted_tokens_per_pos_total{'):
            pos = re.search(r'position="(\d+)"', line)
            if pos:
                key = 'accepted_position_'+pos.group(1)
                result[key] = result.get(key, 0) + float(line.rsplit(' ', 1)[1])
    return result


def idle(p):
    deadline = time.monotonic() + p['settle_timeout_seconds']
    while True:
        s = snapshot(p)
        if s['running'] or s['waiting']:
            raise RuntimeError('Server is busy: use an exclusive benchmark session')
        if s['stores_sent'] == s['stores_done'] and s['gpu_stores_sent'] == s['gpu_stores_done']:
            return s
        if time.monotonic() >= deadline:
            raise RuntimeError('L2 stores did not drain before cache manipulation')
        time.sleep(1)


def reset_gpu(p):
    # Never reset running requests or external cache. A 404 is a hard failure.
    deadline = time.monotonic() + p['settle_timeout_seconds']
    while True:
        idle(p)
        value = json.loads(http(p['base']+'/reset_prefix_cache?reset_running_requests=false&reset_external=false', {}))
        if value.get('success') is True:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError('GPU APC reset did not succeed')
        time.sleep(1)


def clear_l1(p):
    idle(p)
    code = '''import importlib.metadata as m
assert m.version("lmcache")=="0.5.5"
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
c=RequestClientFactory.create(%r)
try:
 c.clear().result(timeout=30)
finally:
 c.close()
''' % ('tcp://127.0.0.1:'+str(p['cache_port']))
    subprocess.run(['docker', 'exec', p['cache_container'], 'python3', '-c', code],
                   check=True, timeout=45, stdout=subprocess.DEVNULL)
    deadline = time.monotonic()+p['settle_timeout_seconds']
    while snapshot(p)['l1_bytes'] != 0:
        if time.monotonic() >= deadline:
            raise RuntimeError('L1 clear failed; refusing to label this disk-warm')
        time.sleep(1)


def cache_proof(state, before, after, length, count, chunk):
    d = {k: after.get(k, 0)-before.get(k, 0) for k in after if k not in ('running','waiting','l1_bytes')}
    if any(v < 0 for v in d.values()):
        raise RuntimeError('Counters reset during batch; service restart or concurrent reset')
    if d['preempt'] or d['gpu_hit']:
        raise RuntimeError('Preemption or GPU APC hit invalidates cache-tier measurement')
    if d['success'] != count:
        raise RuntimeError('Missing completion or foreign request detected')
    expected = (length // chunk) * chunk * count
    if state == 'cold' and (d['gpu_stores_done'] <= 0 or d['stores_done'] <= 0 or d['stored_objects'] <= 0):
        # Completed L2 requests include failed tasks; successful objects do not.
        raise RuntimeError('Cold cache population not proven by GPU/L2 successful-store counters')
    if state == 'cold':
        ok = d['external_hit'] == d['l1_hit'] == d['l2_hit'] == d['l2_load'] == 0
    elif state == 'ram':
        ok = d['external_hit'] == d['l1_hit'] == expected and d['l2_hit'] == d['l2_load'] == 0
    else:
        ok = before['l1_bytes'] == 0 and d['l1_hit'] == 0 and d['external_hit'] == d['l2_hit'] == expected and d['l2_load'] > 0
    if not ok:
        raise RuntimeError(f'Cache source not proven for {state}: expected {expected} tokens, deltas={d}')
    return d


def container_identity(name):
    # Never write Docker's full Env: it can contain credentials.
    c = json.loads(hardware.command('docker', 'inspect', name))[0]
    env = dict(x.split('=', 1) for x in c['Config']['Env'])
    allowed = ('HIP_VISIBLE_DEVICES', 'VLLM_SERVER_DEV_MODE', 'VLLM_PLUGINS', 'HSA_ENABLE_IPC_MODE_LEGACY')
    args = list(c['Config']['Cmd'] or [])
    if any('api-key' in x.lower() or 'token=' in x.lower() for x in args):
        raise RuntimeError('Credential-bearing launch args: use the documented isolated local benchmark deployment')
    return dict(id=c['Id'], image=c['Image'], args=args,
                env={k:v for k,v in env.items() if k in allowed or k.startswith('R9K_')},
                mounts=sorted((dict(source=m['Source'], destination=m['Destination']) for m in c['Mounts']),
                              key=lambda m:(m['destination'],m['source'])),
                running=c['State']['Running'])


def flag(args, name):
    if name not in args:
        raise RuntimeError('Missing launch flag '+name)
    return args[args.index(name)+1]


def sources():
    paths = [*HERE.glob('*.py'), HERE/'glm.json', HERE/'setup.sh',
             HERE.parent/'glm_quality.py', HERE.parent/'api_smoke.py', HERE.parent/'glm_niah_corpus.py']
    paths += list((ROOT/'r9700_vllm').rglob('*.py'))
    paths += list((ROOT/'r9700_vllm/kernels').glob('*.so'))
    paths += [ROOT/'serve/serve.sh', ROOT/'serve/glm-kv-cache.sh',
              ROOT/'serve/baseline-earkita/glm/glm-5.3-flash-c4-lmcache.sh']
    return {str(p.relative_to(ROOT)):digest(p) for p in paths}


def host_memory():
    raw = Path('/proc/meminfo').read_text()
    values = {line.split(':')[0]:int(line.split()[1])*1024 for line in raw.splitlines()}
    return dict(available=values['MemAvailable'],total=values['MemTotal'],
                swap_used=values['SwapTotal']-values['SwapFree'])


def prepare(a):
    p = json.loads((HERE/'glm.json').read_text())
    a.out = a.out.resolve()
    a.out.relative_to(ROOT/'bench/results')
    a.out.mkdir(parents=True, exist_ok=False)
    p.update(base=a.base.rstrip('/'), cache_metrics=a.cache_metrics.rstrip('/'),
             cache_port=a.cache_port, model_container=a.model_container, cache_container=a.cache_container,
             model='glm-5.3-flash', created=time.time(), run_id=uuid.uuid4().hex,
             git_head=hardware.command('git','-C',str(ROOT),'rev-parse','HEAD').strip(), sources=sources())
    for url in (p['base'],p['cache_metrics']):
        if urlparse(url).hostname not in ('localhost','127.0.0.1'):
            raise RuntimeError('Use loopback benchmark endpoints')
    p['model_identity'] = container_identity(a.model_container)
    p['cache_identity'] = container_identity(a.cache_container)
    mi, ci = p['model_identity'], p['cache_identity']
    if not mi['running'] or not ci['running'] or mi['image'] != ci['image']:
        raise RuntimeError('Model and cache must be running the same qualified image')
    if mi['env'].get('VLLM_SERVER_DEV_MODE') != '1':
        raise RuntimeError('Benchmark requires VLLM_SERVER_DEV_MODE=1 for confirmed APC resets; no services changed')
    for k,v in {'--host':'127.0.0.1','--tensor-parallel-size':'8','--max-num-seqs':'4','--kv-cache-dtype':'fp8_e4m3'}.items():
        if flag(mi['args'], k) != v:
            raise RuntimeError(f'This standard requires {k} {v}')
    if '--enable-expert-parallel' in mi['args'] or '--enforce-eager' in mi['args']:
        raise RuntimeError('This standard requires TP8/no EP and graph execution')
    for k in ('--pipeline-parallel-size','--data-parallel-size'):
        if k in mi['args'] and flag(mi['args'],k) != '1':
            raise RuntimeError('HIP-to-worker mapping requires PP1/DP1')
    spec = json.loads(flag(mi['args'], '--speculative-config'))
    if spec.get('method') != 'dflash' or spec.get('num_speculative_tokens') != 4:
        raise RuntimeError('This standard requires DFlash K4')
    if spec.get('kv_cache_dtype') != 'fp8_e4m3' or spec.get('draft_tensor_parallel_size') != 8:
        raise RuntimeError('This standard requires FP8 draft KV and draft TP8')
    if '--no-enable-prefix-caching' in mi['args'] or '--kv-transfer-config' not in mi['args']:
        raise RuntimeError('APC and LMCache must be enabled')
    if int(flag(ci['args'], '--port')) != p['cache_port']:
        raise RuntimeError('Cache control port mismatch')
    if int(flag(ci['args'], '--prometheus-port')) != urlparse(p['cache_metrics']).port:
        raise RuntimeError('Cache metrics port mismatch')
    if int(flag(mi['args'], '--port')) != urlparse(p['base']).port:
        raise RuntimeError('Model API port mismatch')
    p['configured_l1_gib'] = float(flag(ci['args'], '--l1-size-gb'))
    p['host_memory_at_prepare'] = host_memory()
    if int(flag(ci['args'], '--chunk-size')) != p['chunk_tokens']:
        raise RuntimeError('Cache chunk geometry differs from protocol')
    adapter = json.loads(flag(ci['args'], '--l2-adapter'))
    p['configured_l2_gib'] = adapter.get('max_capacity_gb', 0)
    p['l2_eviction'] = adapter.get('eviction')
    mount = next(m for m in ci['mounts'] if m['destination']=='/kv-cache')
    if adapter['type'] != 'fs_native' or adapter.get('use_odirect') is not False:
        raise RuntimeError('Expected buffered filesystem L2')
    p['disk_path'] = str(Path(mount['source'])/Path(adapter['base_path']).relative_to('/kv-cache'))
    p['versions'] = json.loads(hardware.command('docker','exec',a.model_container,'python3','-c',
        'import importlib.metadata as m,json; print(json.dumps({k:m.version(k) for k in ("vllm","lmcache","torch","triton")}))'))
    if 'e97573215' not in p['versions']['vllm'] or p['versions']['lmcache'] != '0.5.5':
        raise RuntimeError('This versioned standard requires vLLM e97573215 and LMCache 0.5.5')
    p['mapping'] = hardware.hip_mapping(a.model_container)
    p['idle_links'] = hardware.capture(a.out/'hardware-before',p['mapping'])
    p['pcie_expected'] = hardware.expected_links(p['idle_links'])
    if a.pcie_policy:
        policy = json.loads(a.pcie_policy.read_text())
        if not policy.get('reason') or set(policy['links']) != set(p['pcie_expected']):
            raise RuntimeError('PCIe policy needs a reason and every endpoint/upstream bridge')
        p['pcie_policy'] = policy
        p['pcie_expected'] = policy['links']
    p['corpora'] = {}
    model_path = flag(mi['args'], '--model') if '--model' in mi['args'] else next(
        (x for x in mi['args'] if x.startswith('/models/')), None)
    if not model_path:
        raise RuntimeError('Cannot determine tokenizer path')
    for effort in p['efforts']:
        # Model's repo mount can be read-only. Build in a temporary container
        # directory and return JSON, rather than writing through that mount.
        argv = ['glm_niah_corpus.py','--model',model_path,'--effort',effort,'--records','4096',
                '--lengths',*[str(n) for n in p['lengths']]]
        code = '''import contextlib,io,json,runpy,sys,tempfile,pathlib,hashlib
with tempfile.TemporaryDirectory() as t:
 out=pathlib.Path(t)/"corpus"
 sys.argv=%r+["--out",str(out)]
 with contextlib.redirect_stdout(io.StringIO()):
  runpy.run_path("/opt/r9700/bench/glm_niah_corpus.py",run_name="__main__")
 root=pathlib.Path(%r)
 hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in root.iterdir() if f.is_file() and ("token" in f.name or "config" in f.name or "template" in f.name or f.name.endswith(".index.json"))}
 print(json.dumps(dict(corpus=json.loads((out/"corpus.json").read_text()),manifest=json.loads((out/"manifest.json").read_text()),checkpoint_metadata=hashes)))
''' % (argv, model_path)
        print('Preparing exact token corpus:',effort,flush=True)
        text = subprocess.check_output(['docker','exec',a.model_container,'python3','-c',code],text=True,timeout=1800)
        corpus = json.loads(text)
        file = a.out/f'corpus-{effort}.json'
        save(file,corpus)
        p['corpora'][effort] = dict(file=file.name,sha256=digest(file))
    snapshot(p,a.out/'metrics-prepared')
    save(a.out/'protocol.json',p)
    (a.out/'protocol.sha256').write_text(digest(a.out/'protocol.json')+'\n')
    print('Protocol frozen. No inference or cache reset performed:',a.out)


def batch(p, out, cases, salts, effort, length, quality):
    barrier = threading.Barrier(len(cases))
    stop = threading.Event()
    samples, errors = [], []
    def observe():
        with (out/'telemetry.jsonl').open('w') as f:
            while not stop.is_set():
                try:
                    s = dict(time=time.monotonic(),links=hardware.links(p['mapping']),metrics=snapshot(p),memory=host_memory(),
                             gpu=json.loads(hardware.command('amd-smi','metric','--usage','--power','--temperature','--json')))
                    samples.append(s)
                    f.write(json.dumps(s)+'\n'); f.flush()
                except Exception as e:
                    errors.append(str(e))
                    f.write(json.dumps({'error':str(e)})+'\n'); f.flush()
                stop.wait(p['telemetry_interval_seconds'])
    def request(i):
        case = cases[i]
        body = dict(model=p['model'],messages=case['messages'],cache_salt=salts[i],
                    temperature=p['temperature'],top_p=p['top_p'],top_k=p['top_k'],seed=p['seed'],
                    max_tokens=p['quality_output_tokens'] if quality else p['output_tokens'],
                    ignore_eos=not quality,chat_template_kwargs={'reasoning_effort':effort},
                    presence_penalty=0.,frequency_penalty=0.,repetition_penalty=1.,min_p=0.,
                    stream=True,stream_options={'include_usage':True})
        save(out/f'request-{i}.json',body)
        barrier.wait(timeout=30)
        row = dict(case=case['id'],start=time.monotonic())
        try:
            if quality:
                response,timing = stream(p['base'],body,p['request_timeout_seconds'])
                row.update(response=response,timing=timing,niah=niah_scores(response,case),
                           foreign_key=any(c['expected'] in response['content'] for c in cases if c is not case))
            else:
                from betterbench.client import stream_chat_sync
                extra = {k:v for k,v in body.items() if k not in
                         ('model','messages','max_tokens','temperature','top_p','top_k','seed','stream','stream_options')}
                row['result'] = stream_chat_sync(p['base']+'/v1',p['model'],case['messages'],
                    max_tokens=p['output_tokens'],temperature=p['temperature'],top_p=p['top_p'],
                    top_k=p['top_k'],seed=p['seed'],extra_body=extra,timeout=p['request_timeout_seconds'],
                    category='earkita',prompt_id=case['id']).as_dict()
        except Exception as e:
            row['error'] = f'{type(e).__name__}: {e}'
            if quality:
                # A transport exception alone does not prove a backend fault.
                row['niah'] = niah_scores(None,case,backend_runtime_error='UNKNOWN')
        row['end'] = time.monotonic()
        save(out/f'response-{i}.json',row)
        return row
    observer = threading.Thread(target=observe)
    observer.start()
    try:
        with ThreadPoolExecutor(max_workers=len(cases)) as pool:
            rows = list(pool.map(request,range(len(cases))))
    finally:
        stop.set(); observer.join()
    return rows, samples, errors


def summarize(rows, length):
    duration = max(r['end'] for r in rows)-min(r['start'] for r in rows)
    starts = [r['start']+r['result']['ttft_ms']/1000 for r in rows]
    ends = [s+sum(r['result']['update_gaps_ms'])/1000 for s,r in zip(starts,rows)]
    return dict(batch_wall_s=duration,
                aggregate_e2e_output_tps=sum(r['result']['completion_tokens'] for r in rows)/duration,
                all_clients_decode_overlap_s=max(0,min(ends)-max(starts)),
                per_request=[dict(case=r['case'],decode_tps=r['result']['decode_tps'],
                    ttft_ms=r['result']['ttft_ms'],ttfa_ms=r['result']['ttfa_ms'],
                    effective_prompt_tps=length/(r['result']['ttft_ms']/1000)) for r in rows])


def qualify_decode_overlap(summary, count, state, allow_serial_cold_prefill=False):
    """Cold-prefill diagnostics may finish short outputs before peers prefill.

    Preserve the strict baseline by default. An explicit frozen diagnostic
    exception validates cache/TTFT only, never concurrent decode; warm tiers
    must still demonstrate overlapping decode for every client.
    """
    qualified = count == 1 or summary['all_clients_decode_overlap_s'] > 0
    if not qualified and not (state == 'cold' and allow_serial_cold_prefill):
        raise RuntimeError('No simultaneous decode observed for all clients')
    return qualified


def report(out, batches, status):
    groups = defaultdict(list)
    quality = defaultdict(Counter)
    for b in batches:
        if b.get('valid') and b['phase']=='measured':
            groups[f"{b['length']}/C{b['concurrency']}/{b['effort']}/{b['cache']}"] += [b]
        if b['phase']=='quality':
            for r in b.get('requests',[]):
                for k,v in r.get('niah',{}).items():
                    if isinstance(v,str):
                        quality[k][v] += 1
    summary = {}
    for key,group in groups.items():
        stats = {}
        for field in ('decode_tps','ttft_ms','ttfa_ms','effective_prompt_tps'):
            values = [r[field] for b in group for r in b['summary']['per_request'] if r[field] is not None]
            stats[field] = dict(median=statistics.median(values),min=min(values),max=max(values),n=len(values)) if values else None
        summary[key] = dict(batches=len(group),per_request=stats,
            decode_concurrency_qualified=all(b.get('decode_concurrency_qualified',True) for b in group),
            aggregate_e2e_output_tps=[b['summary']['aggregate_e2e_output_tps'] for b in group],
            acceptance=[b.get('acceptance') for b in group],
            overlap_s=[b['summary']['all_clients_decode_overlap_s'] for b in group])
    save(out/'report.json',dict(status=status,groups=summary,niah=quality))
    lines = ['# Earkita benchmark', '', 'Status: '+status,
             '', 'Warm effective prompt rate includes cache restore; it is not computed prefill speed.',
             'Disk means filesystem L2, potentially served by the OS page cache.', '',
             '| Input / C / effort / cache | Batches | Decode median per request tok/s | TTFT median ms | Concurrent decode qualified |',
             '|---|---:|---:|---:|---|']
    for key,g in summary.items():
        s = g['per_request']
        lines.append(f"| {key} | {g['batches']} | {s['decode_tps']['median']:.2f} | {s['ttft_ms']['median']:.1f} | {g['decode_concurrency_qualified']} |")
    lines += ['', 'NIAH dimensions (including failed attempts):', '', '```json',json.dumps(quality,indent=2),'```',
              '', 'Raw attempts, warmups, per-request rates, overlap and acceptance are retained in JSON.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def run(a):
    out = a.out.resolve()
    p = json.loads((out/'protocol.json').read_text())
    if digest(out/'protocol.json') != (out/'protocol.sha256').read_text().strip() or p['sources'] != sources():
        raise RuntimeError('Protocol or benchmark code changed; prepare a fresh run')
    if importlib.metadata.version('betterbench') != '0.6.0':
        raise RuntimeError('Install pinned client with setup.sh')
    provenance = json.loads(importlib.metadata.distribution('betterbench').read_text('direct_url.json') or '{}')
    if provenance.get('vcs_info',{}).get('commit_id') != p['betterbench_commit']:
        raise RuntimeError('Client Git revision differs from frozen protocol; use setup.sh')
    lock = (ROOT/'.runtime').joinpath('earkita-benchmark.lock')
    lock.parent.mkdir(exist_ok=True)
    with lock.open('w') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out/'status.json').exists():
            raise RuntimeError('Attempt already started: preserve it and prepare a fresh run')
        save(out/'status.json',dict(status='running',started=time.time()))
        batches = []
        try:
            for role in ('model','cache'):
                if container_identity(p[role+'_container']) != p[role+'_identity']:
                    raise RuntimeError(f'{role} container/configuration changed')
            if hardware.hip_mapping(p['model_container']) != p['mapping']:
                raise RuntimeError('HIP device mapping changed')
            hardware.capture(out/'hardware-run',p['mapping'])
            if (out/'hardware-run/power-limits.json').read_text() != (out/'hardware-before/power-limits.json').read_text():
                raise RuntimeError('GPU power/capability settings changed')
            corpora = {}
            for effort, c in p['corpora'].items():
                if digest(out/c['file']) != c['sha256']:
                    raise RuntimeError('Frozen corpus changed')
                corpora[effort] = json.loads((out/c['file']).read_text())['corpus']
                for case in corpora[effort]:
                    length = int(case['id'].split('_')[1])
                    tokenized = json.loads(http(p['base']+'/tokenize',dict(model=p['model'],messages=case['messages'],
                        add_generation_prompt=True,chat_template_kwargs={'reasoning_effort':effort})))
                    if tokenized['count'] != length:
                        raise RuntimeError('Server tokenization differs from frozen corpus')
            for length in p['lengths']:
                for effort in p['efforts']:
                    for count in p['concurrency']:
                        cases = [c for c in corpora[effort] if c['id'].startswith(f'niah_{length}_')][:count]
                        phases = ['warmup']*p['warmup_batches']+['measured']*p['measured_batches']+['quality']
                        for repetition,phase in enumerate(phases):
                            salts = [hashlib.sha256(f"{p['run_id']}:{length}:{effort}:{count}:{repetition}:{i}".encode()).hexdigest()
                                     for i in range(len(cases))]
                            for state in p['cache_states']:
                                if shutil.disk_usage(p['disk_path']).free < p['minimum_disk_free_gib']*2**30:
                                    raise RuntimeError('Insufficient free cache disk space; no files deleted')
                                target = out/f'batch-{len(batches):04d}'
                                target.mkdir()
                                b = dict(length=length,effort=effort,concurrency=count,cache=state,
                                         repetition=repetition,phase=phase,valid=False)
                                batches.append(b)
                                save(target/'result.json',b)
                                save(out/'status.json',dict(status='running',batch=target.name,**b))
                                print(target.name,length,'C'+str(count),effort,state,phase,flush=True)
                                save(target/'pcie-idle.json',hardware.links(p['mapping']))
                                power = hardware.command('amd-smi','static','--asic','--limit','--json')
                                (target/'power-limits.json').write_text(power)
                                if power != (out/'hardware-before/power-limits.json').read_text():
                                    raise RuntimeError('Power/capability settings changed between cache variants')
                                idle(p); reset_gpu(p)
                                if state in ('cold','disk'):
                                    clear_l1(p)
                                before = snapshot(p,target/'before')
                                rows,samples,errors = batch(p,target,cases,salts,effort,length,phase=='quality')
                                b['requests'] = rows
                                save(target/'result.json',b)
                                idle(p)
                                deadline = time.monotonic()+p['settle_timeout_seconds']
                                while True:
                                    after = snapshot(p,target/'after')
                                    if after['success']-before['success'] >= count or time.monotonic() >= deadline:
                                        break
                                    time.sleep(1)
                                b.update(before=before,after=after)
                                save(target/'result.json',b)
                                if errors:
                                    raise RuntimeError('Telemetry failed: '+str(errors))
                                hardware.check_loaded([s['links'] for s in samples],p['idle_links'],p['pcie_expected'])
                                d = cache_proof(state,before,after,length,count,p['chunk_tokens'])
                                b['cache_proof'] = d
                                b['acceptance'] = dict(ratio=d['accepted']/d['draft_tokens'] if d['draft_tokens'] else None,
                                    per_position={k:dict(accepted=v,per_draft=v/d['drafts'] if d['drafts'] else None)
                                                  for k,v in d.items() if k.startswith('accepted_position_')},drafts=d['drafts'])
                                if phase=='quality':
                                    if any(r.get('error') or r.get('foreign_key') or r['niah']['generation_completed']!='PASS'
                                           or r['niah']['semantic_needle_retrieval']!='PASS' for r in rows):
                                        raise RuntimeError('Quality retrieval/completion/runtime gate failed; see separate NIAH fields')
                                else:
                                    if any(r.get('error') or not r['result']['ok'] or r['result']['prompt_tokens']!=length
                                           or r['result']['completion_tokens']!=p['output_tokens'] or not r['result']['decode_tps'] for r in rows):
                                        raise RuntimeError('Incomplete/failed performance request')
                                    b['summary'] = summarize(rows,length)
                                    b['decode_concurrency_qualified'] = qualify_decode_overlap(
                                        b['summary'], count, state,
                                        p.get('allow_serial_cold_prefill_diagnostic',False))
                                b['valid'] = True
                                save(target/'result.json',b)
                                report(out,batches,'running')
            save(out/'status.json',dict(status='complete',finished=time.time()))
            report(out,batches,'complete')
        except BaseException as e:
            save(out/'status.json',dict(status='failed',error=f'{type(e).__name__}: {e}',finished=time.time()))
            report(out,batches,'failed')
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action',required=True)
    sub.add_parser('plan')
    prep = sub.add_parser('prepare')
    prep.add_argument('--out',type=Path,required=True)
    prep.add_argument('--model-container',required=True)
    prep.add_argument('--cache-container',required=True)
    prep.add_argument('--base',default='http://127.0.0.1:8080')
    prep.add_argument('--cache-metrics',default='http://127.0.0.1:9091')
    prep.add_argument('--cache-port',type=int,default=5555)
    prep.add_argument('--pcie-policy',type=Path,help='Explicit expected links + reason for wired limits below device capabilities')
    execute = sub.add_parser('run')
    execute.add_argument('--out',type=Path,required=True)
    a = parser.parse_args()
    if a.action=='plan':
        p = json.loads((HERE/'glm.json').read_text())
        groups = len(p['lengths'])*len(p['efforts'])*len(p['cache_states'])
        perf = groups*(p['warmup_batches']+p['measured_batches'])
        print(json.dumps(dict(protocol=p,performance_batches=perf*len(p['concurrency']),
            performance_requests=perf*sum(p['concurrency']),quality_batches=groups*len(p['concurrency']),
            quality_requests=groups*sum(p['concurrency']),starts_services=False),indent=2))
    elif a.action=='prepare':
        prepare(a)
    else:
        run(a)


if __name__=='__main__':
    main()
