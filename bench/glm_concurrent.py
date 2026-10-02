#!/usr/bin/env python3
"""Concurrent GLM NIAH/BetterBench gate using a frozen, exactly tokenized corpus.

Quality uses natural EOS; speed uses fixed output lengths. Reports C4 decode
intersection independently of submitted clients and scheduler running count.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import re
from pathlib import Path
import subprocess
import threading
import time
import uuid

from glm_quality import metrics, read_url, stream, niah_scores


def snapshot(base):
    result = metrics(base)
    with read_url(base, '/metrics') as response:
        raw = response.read().decode()
    for name in ('num_preemptions_total', 'spec_decode_num_drafts_total',
                 'spec_decode_num_draft_tokens_total', 'spec_decode_num_accepted_tokens_total'):
        values = [float(line.rsplit(' ', 1)[1]) for line in raw.splitlines()
                  if line.startswith('vllm:' + name + '{')]
        if not values:
            raise RuntimeError('Missing metric: ' + name)
        result[name] = sum(values)
    for line in raw.splitlines():
        if line.startswith('vllm:spec_decode_num_accepted_tokens_per_pos_total{'):
            position = re.search(r'position="([0-9]+)"', line)
            if position:
                key = 'accepted_position_' + position.group(1)
                result[key] = result.get(key, 0.) + float(line.rsplit(' ', 1)[1])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', default='http://localhost:8080')
    p.add_argument('--model', default='glm-5.3-flash')
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--prompt-tokens', type=int, required=True)
    p.add_argument('--output-tokens', type=int, default=1024)
    p.add_argument('--concurrency', type=int, default=4)
    p.add_argument('--rounds', type=int, default=1)
    p.add_argument('--quality', action='store_true')
    p.add_argument('--reuse-prefix', action='store_true')
    p.add_argument('--prefix-salts-from', type=Path,
                   help='Reuse corresponding cases from an earlier results.json')
    p.add_argument('--timeout', type=int, default=1800)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    cases = [c for c in json.loads(a.corpus.read_text())
             if c['id'].startswith(f'niah_{a.prompt_tokens}_')][:a.concurrency]
    if len(cases) != a.concurrency:
        raise ValueError('Need one distinct frozen case per client')
    sampling = dict(temperature=1., top_p=.95, top_k=-1, seed=1234,
                    chat_template_kwargs={'reasoning_effort':'high'})
    for c in cases:
        with read_url(a.base, '/tokenize', dict(model=a.model, messages=c['messages'],
                    add_generation_prompt=True, chat_template_kwargs=sampling['chat_template_kwargs'])) as r:
            assert json.load(r)['count'] == a.prompt_tokens
    salts = [uuid.uuid4().hex for _ in cases]
    if a.prefix_salts_from:
        if not a.reuse_prefix:
            p.error('--prefix-salts-from requires --reuse-prefix')
        previous = json.loads(a.prefix_salts_from.read_text())[-1]['requests']
        by_case = {r['case']: r['request_extra']['cache_salt'] for r in previous}
        salts = [by_case[c['id']] for c in cases]
    rounds = []
    for repetition in range(a.rounds):
        before = snapshot(a.base)
        assert before['num_requests_running'] == before['num_requests_waiting'] == 0
        barrier = threading.Barrier(a.concurrency)
        stop = threading.Event()
        samples = []
        def observe():
            with (a.out/f'telemetry-{repetition}.jsonl').open('w') as f:
                tick = 0
                while not stop.is_set():
                    try:
                        row = dict(time=time.monotonic(), metrics=snapshot(a.base))
                        if tick % 5 == 0:
                            row['memory'] = json.loads(subprocess.check_output(
                                ['amd-smi','metric','--mem-usage','--json'], text=True))
                        samples.append(row)
                        f.write(json.dumps(row)+'\n'); f.flush()
                        tick += 1
                    except Exception as e:
                        f.write(json.dumps({'error':str(e)})+'\n'); f.flush()
                    stop.wait(.2)
        def request(i):
            c = cases[i]
            extra = dict(cache_salt=salts[i] if a.reuse_prefix else uuid.uuid4().hex,
                         chat_template_kwargs=sampling['chat_template_kwargs'],
                         min_p=0., presence_penalty=0., frequency_penalty=0., repetition_penalty=1.)
            barrier.wait()
            start = time.monotonic()
            try:
                if a.quality:
                    body = dict(model=a.model, messages=c['messages'], max_tokens=a.output_tokens,
                                ignore_eos=False, stream=True, stream_options={'include_usage':True},
                                **sampling) | extra
                    result, timing = stream(a.base, body, a.timeout)
                    row = dict(response=result, timing=timing, niah=niah_scores(result,c),
                               foreign_key=any(z['expected'] in result['content'] for z in cases if z is not c))
                else:
                    from betterbench.client import stream_chat_sync
                    result = stream_chat_sync(a.base+'/v1', a.model, c['messages'],
                                max_tokens=a.output_tokens, temperature=1., top_p=.95, top_k=-1, seed=1234,
                                category='glm-concurrent', prompt_id=c['id'], timeout=a.timeout,
                                extra_body=extra|{'ignore_eos':True})
                    row = dict(result=result.as_dict())
                return dict(case=c['id'], start=start, end=time.monotonic(), request_extra=extra, **row)
            except Exception as e:
                return dict(case=c['id'],start=start,end=time.monotonic(),error=f'{type(e).__name__}: {e}')
        observer = threading.Thread(target=observe)
        observer.start()
        try:
            with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
                rows = list(pool.map(request, range(a.concurrency)))
        finally:
            stop.set(); observer.join()
        # Prometheus publishes some counters asynchronously; wait for the batch.
        for _ in range(100):
            after = snapshot(a.base)
            if after['prefix_cache_queries_total']-before['prefix_cache_queries_total'] >= a.concurrency*a.prompt_tokens:
                break
            time.sleep(.1)
        duration = max(r['end'] for r in rows)-min(r['start'] for r in rows)
        summary = dict(round=repetition, quality=a.quality, sampling=sampling,
                       prompt_tokens=a.prompt_tokens, output_budget=a.output_tokens,
                       concurrency=a.concurrency, batch_elapsed_s=duration,
                       peak_running=max((s['metrics']['num_requests_running'] for s in samples), default=0),
                       metrics_before=before, metrics_after=after, requests=rows)
        if not a.quality and all('result' in r and r['result']['ok'] for r in rows):
            starts = [r['start']+r['result']['ttft_ms']/1000 for r in rows]
            ends = [s+sum(r['result']['update_gaps_ms'])/1000 for s,r in zip(starts,rows)]
            summary['all_decode_overlap_s'] = max(0., min(ends)-max(starts))
            summary['per_request_decode_tps'] = [r['result']['decode_tps'] for r in rows]
            summary['aggregate_output_tps_including_prefill'] = sum(r['result']['completion_tokens'] for r in rows)/duration
        rounds.append(summary)
        (a.out/'results.json').write_text(json.dumps(rounds,indent=2)+'\n')
        print(json.dumps({k:v for k,v in summary.items() if k not in ('requests','metrics_before','metrics_after')}),flush=True)
        assert all('error' not in r for r in rows), 'Request errors; see results.json'
        assert after['num_preemptions_total'] == before['num_preemptions_total'], 'Preemption; capacity gate failed'
        if a.quality:
            assert all(r['niah']['semantic_needle_retrieval']=='PASS'
                       and r['niah']['generation_completed']=='PASS' and not r['foreign_key'] for r in rows), 'Quality gate failed'
        else:
            assert all(r['result']['ok'] and r['result']['prompt_tokens']==a.prompt_tokens
                       and r['result']['completion_tokens']==a.output_tokens
                       and r['result']['finish_reason']=='length' for r in rows)
            assert summary['all_decode_overlap_s'] > 0, 'No all-client decode overlap'


if __name__ == '__main__':
    main()
