#!/usr/bin/env python3
"""Bounded diagnostic calibration, never a throughput benchmark.

Start the DeepSeek profile with R9K_DEEPSEEK_EXPERT_PROFILE=<container path>.
--snapshot names the same file on the host. Input is a frozen JSON list of
OpenAI chat payloads. Each run resets warmup counters and retains all responses.
Run corpora separately; keep held-out prompts out of calibration.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import time
import urllib.request
import uuid


def request(base, path, payload=None, timeout=600):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base.rstrip('/')+path,body,{'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as response:
        return response.read().decode()


def require_idle(base):
    metrics = request(base,'/metrics',timeout=10)
    for line in metrics.splitlines():
        if line.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{')) and float(line.rsplit(' ',1)[1]):
            raise RuntimeError('Calibration requires an idle, exclusively used backend')


def control(snapshot, action):
    command={'action':action,'id':uuid.uuid4().hex}
    path=Path(str(snapshot)+'.control');temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(command));temp.replace(path)
    end=time.monotonic()+30
    while time.monotonic()<end:
        if snapshot.exists():
            data=json.loads(snapshot.read_text())
            if data.get('command')==command:return data
        time.sleep(.2)
    raise TimeoutError('No acknowledgement from routing recorder')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8080')
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--prompts',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--concurrency',type=int,choices=(1,2,4),default=4)
    args=parser.parse_args()
    payloads=json.loads(args.prompts.read_text())
    if not isinstance(payloads,list) or not 1<=len(payloads)<=32:
        raise ValueError('Provide a bounded list of 1..32 frozen payloads')
    for payload in payloads:
        if not 1<=payload.get('max_tokens',0)<=256 or payload.get('stream',False):
            raise ValueError('Calibration requires non-streaming requests with max_tokens<=256')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'protocol.json').write_text(json.dumps({'requests':payloads,'concurrency':args.concurrency,
        'kind':'diagnostic routing calibration; timing invalid'},indent=2))
    require_idle(args.url)
    control(args.snapshot,'reset')
    def run(item):
        index,payload=item;path=args.output/f'request-{index:03d}.json'
        result={'request':payload,'status':'started'}
        path.write_text(json.dumps(result,indent=2))
        try:
            result.update(status='completed',response=json.loads(request(args.url,'/v1/chat/completions',payload)))
        except Exception as exc:
            result.update(status='failed',error=repr(exc))
        path.write_text(json.dumps(result,indent=2))
        return result['status']=='completed'
    with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
        passed=list(pool.map(run,enumerate(payloads)))
    require_idle(args.url)
    data=control(args.snapshot,'snapshot')
    data['calibration_requests']=len(payloads);data['all_requests_completed']=all(passed)
    (args.output/'routing.json').write_text(json.dumps(data,indent=2))
    if not all(passed):raise RuntimeError('Failed calibration requests; profile must not be promoted')
    print(f'Saved routing counters for {len(data["rows"])} layers; no performance claim')


if __name__=='__main__':main()
