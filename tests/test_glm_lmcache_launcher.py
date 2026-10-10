"""Combined launcher lifecycle, using a fake Docker executable; no GPU access."""
import json
import os
from pathlib import Path
import shlex
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / 'serve/baseline-earkita/glm-5.3-flash-c4-lmcache.sh'


def test_dryrun_configures_both_services(tmp_path):
    out = subprocess.check_output([
        'bash', str(PROFILE), 'DRYRUN=1', 'IMG=test-image',
        'KV_CACHE_PORT=5556', 'KV_CACHE_RAM_GB=2',
        f'KV_CACHE_DIR={tmp_path}/cache',
    ], text=True)
    cache, model = [shlex.split(line) for line in out.splitlines()]
    assert 'test-image' in cache and 'test-image' in model
    assert cache[cache.index('--port')+1] == '5556'
    assert cache[cache.index('--l1-size-gb')+1] == '2'
    config = json.loads(model[model.index('--kv-transfer-config')+1])
    assert config['kv_connector_extra_config']['lmcache.mp.port'] == 5556
    assert model[model.index('--max-num-seqs')+1] == '4'
    assert not (tmp_path/'cache').exists()


@pytest.mark.parametrize('scenario', ['success', 'model_exists', 'cache_exists', 'cache_failed', 'model_failed'])
def test_lifecycle_only_cleans_up_its_own_cache(tmp_path, scenario):
    fake = tmp_path/'docker'
    fake.write_text('''#!/usr/bin/env python3
import json,os,sys
args=sys.argv[1:]; scenario=os.environ['SCENARIO']
with open(os.environ['CALL_LOG'],'a') as f: f.write(json.dumps(args)+'\\n')
if args[:2]==['container','inspect']: sys.exit(0 if scenario=='model_exists' else 1)
if args[0]=='inspect': print('false' if scenario=='cache_failed' else 'true')
elif args[0]=='logs': print('LMCache ZMQ cache server is running on tcp://127.0.0.1:5555')
elif args[0]=='run' and '-d' in args:
 name=args[args.index('--name')+1]
 if name=='glm53-kv-cache-c4':
  if scenario=='cache_exists': sys.exit(125)
  print('new-cache-id')
 else:
  if scenario=='model_failed': sys.exit(125)
  print('new-model-id')
''')
    fake.chmod(0o755)
    log = tmp_path/'calls.jsonl'
    env = dict(os.environ, PATH=str(tmp_path)+os.pathsep+os.environ['PATH'],
               SCENARIO=scenario, CALL_LOG=str(log))
    result = subprocess.run([
        'bash', str(PROFILE), 'DRYRUN=0', 'DOCKER_SUDO=',
        f'KV_CACHE_DIR={tmp_path}/cache', f'VLLM_CACHE_DIR={tmp_path}/vllm',
        f'TRITON_CACHE_DIR={tmp_path}/triton',
    ], env=env, capture_output=True, text=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    stops = [c for c in calls if c[0]=='stop']
    assert (result.returncode == 0) == (scenario == 'success'), result.stderr
    if scenario in ('cache_failed', 'model_failed'):
        assert stops == [['stop','--time','5','new-cache-id']]
    else:
        assert stops == []
    if scenario == 'model_exists':
        assert len(calls) == 1
    if scenario == 'cache_failed':
        assert len([c for c in calls if c[0]=='run']) == 1
    if scenario == 'success':
        runs = [c for c in calls if c[0]=='run' and '-d' in c]
        assert [c[c.index('--name')+1] for c in runs] == [
            'glm53-kv-cache-c4', 'glm53-flash-c4-lmcache']
