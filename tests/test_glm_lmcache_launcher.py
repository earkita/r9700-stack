"""Combined launcher lifecycle, using a fake Docker executable; no GPU access."""
import json
import os
from pathlib import Path
import shlex
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / 'serve/baseline-earkita/glm/glm-5.3-flash-c4-lmcache.sh'


def test_dryrun_configures_both_services(tmp_path):
    out = subprocess.check_output([
        'bash', str(PROFILE), 'DRYRUN=1', 'IMG=test-image',
        'KV_CACHE_PORT=5556', 'KV_CACHE_RAM_GB=128', 'KV_CACHE_DISK_GB=512',
        f'KV_CACHE_DIR={tmp_path}/cache',
    ], text=True)
    cache, model = [shlex.split(line) for line in out.splitlines()]
    assert 'test-image' in cache and 'test-image' in model
    assert cache[cache.index('--port')+1] == '5556'
    assert cache[cache.index('--l1-size-gb')+1] == '128'
    assert '--no-l1-use-lazy' in cache
    assert 'HSA_USERPTR_FOR_PAGED_MEM=0' in cache
    disk = json.loads(cache[cache.index('--l2-adapter')+1])
    assert disk['max_capacity_gb'] == 512
    assert disk['eviction'] == dict(eviction_policy='LRU', trigger_watermark=.75, eviction_ratio=.2)
    config = json.loads(model[model.index('--kv-transfer-config')+1])
    assert config['kv_connector_extra_config']['lmcache.mp.port'] == 5556
    assert model[model.index('--max-num-seqs')+1] == '4'
    assert not (tmp_path/'cache').exists()


def test_native_pinned_allocator_disables_pageable_shm():
    out = subprocess.check_output([
        'bash', str(PROFILE), 'DRYRUN=1', 'KV_CACHE_ALLOCATOR=pinned',
    ], text=True)
    cache = shlex.split(out.splitlines()[0])
    assert '--no-l1-use-lazy' in cache
    assert '--l1-use-lazy' not in cache
    assert cache[cache.index('--shm-name')+1] == ''


def test_gtt_policy_is_scoped_to_cache_container():
    out = subprocess.check_output([
        'bash', str(PROFILE), 'DRYRUN=1', 'KV_CACHE_ALLOCATOR=pinned-gtt',
    ], text=True)
    cache, model = [shlex.split(line) for line in out.splitlines()]
    assert 'HSA_USERPTR_FOR_PAGED_MEM=0' in cache
    assert '--no-l1-use-lazy' in cache
    assert cache[cache.index('--shm-name')+1] == ''
    assert 'HSA_USERPTR_FOR_PAGED_MEM=0' not in model


@pytest.mark.parametrize('setting', ['KV_CACHE_ALLOCATOR=invalid', 'KV_CACHE_START_TIMEOUT=0'])
def test_invalid_allocator_or_timeout_fails_before_launch(setting):
    result = subprocess.run(['bash', str(PROFILE), 'DRYRUN=1', setting],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert not result.stdout


@pytest.mark.parametrize('setting', ['KV_CACHE_DISK_GB=-1', 'KV_CACHE_DISK_GB=nan',
                                    'KV_CACHE_DISK_WATERMARK=1', 'KV_CACHE_DISK_EVICTION_RATIO=0'])
def test_invalid_disk_budget_fails_before_docker(setting):
    result = subprocess.run(['bash', str(ROOT/'serve/glm-kv-cache.sh'), 'DRYRUN=1', setting],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert not result.stdout


def test_lru_refuses_unaccounted_existing_files(tmp_path):
    directory = tmp_path/'existing'
    directory.mkdir()
    data = directory/'keep.data'
    data.write_bytes(b'untouched')
    result = subprocess.run(['bash', str(ROOT/'serve/glm-kv-cache.sh'),
                             f'KV_CACHE_DIR={tmp_path}', 'KV_CACHE_NAMESPACE=existing',
                             'KV_CACHE_DISK_GB=512', 'DOCKER_SUDO='], capture_output=True, text=True)
    assert result.returncode == 2
    assert 'fresh empty namespace' in result.stderr
    assert data.read_bytes() == b'untouched'


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
