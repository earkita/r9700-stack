from types import SimpleNamespace as NS
import pytest
from r9700_vllm.moe.deepseek_expert_profile import phase_ranges


def test_calibration_identity_tracks_dense_storage_policy(tmp_path, monkeypatch):
    from r9700_vllm.moe import deepseek_expert_profile as profile
    identity = profile.identity
    monkeypatch.setattr(profile, 'version', lambda _: '0.31.1rc1.dev376+g5555a2d8f')
    for name in ('config.json', 'model.safetensors.index.json'):
        (tmp_path/name).write_text('{}')
    cfg = NS(model_config=NS(model=str(tmp_path), enforce_eager=True),
             scheduler_config=NS(max_num_seqs=4, max_num_batched_tokens=512),
             speculative_config=None)
    monkeypatch.setenv('VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD', '1')
    before = identity(cfg)
    monkeypatch.setenv('VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD', '0')
    after = identity(cfg)
    assert before != after
    assert before['checkpoint'] == after['checkpoint']


def test_calibration_rejects_other_runtime_and_graph_configuration(tmp_path, monkeypatch):
    import json
    from r9700_vllm.moe import deepseek_expert_profile as profile
    from r9700_vllm.moe.deepseek_placement import read_profiles
    for name in ('config.json', 'model.safetensors.index.json'):
        (tmp_path/name).write_text('{}')
    cfg = NS(model_config=NS(model=str(tmp_path), enforce_eager=False),
             scheduler_config=NS(max_num_seqs=4, max_num_batched_tokens=512),
             compilation_config=NS(cudagraph_mode='FULL_DECODE_ONLY', cudagraph_capture_sizes=[1]),
             speculative_config=None)
    monkeypatch.setattr(profile, 'version', lambda _: '0.31.1rc1.dev376+g5555a2d8f')
    expected = profile.identity(cfg)
    assert expected['vllm'].endswith('5555a2d8f')
    rows = [{'layer': i, 'decode': {'calls': 1, 'slots': [1]*384, 'calls_hit': [1]*384}}
            for i in range(40)]
    path = tmp_path/'routing.json'
    path.write_text(json.dumps(dict(schema=1, identity=expected, rows=rows,
                                    all_requests_completed=True)))
    read_profiles([path], expected)
    cfg.compilation_config.cudagraph_capture_sizes = [1, 2, 4]
    with pytest.raises(ValueError, match='identity'):
        read_profiles([path], profile.identity(cfg))
    cfg.compilation_config.cudagraph_capture_sizes = [1]
    monkeypatch.setattr(profile, 'version', lambda _: '0.31.0+g18f8f960')
    with pytest.raises(ValueError, match='identity'):
        read_profiles([path], profile.identity(cfg))


def test_mixed_batch_and_padding():
    meta={'a':NS(num_decode_tokens=3,num_prefill_tokens=128),'b':NS(num_decode_tokens=3,num_prefill_tokens=128)}
    assert phase_ranges(meta,144)=={'all':(0,131),'decode':(0,3),'prefill':(3,131)}
    assert phase_ranges(None,512) is None


def test_ambiguous_metadata_rejected():
    with pytest.raises(RuntimeError,match='classify'):
        phase_ranges({'a':NS()},512)
    with pytest.raises(RuntimeError,match='mismatch'):
        phase_ranges({'a':NS(num_decode_tokens=4,num_prefill_tokens=0)},3)


def test_padding_does_not_become_a_real_expert():
    torch=pytest.importorskip('torch')
    from r9700_vllm.moe.deepseek_expert_profile import count_routes
    ids=torch.tensor([[1,3,5],[1,8,9],[2,4,6]],dtype=torch.int32)
    counts,calls=count_routes(ids,torch.tensor([True,False,True]))
    assert counts.sum()==6 and counts[1]==1 and counts[8]==0 and calls==1
    counts,calls=count_routes(ids,torch.tensor([False,False,False]))
    assert counts.sum()==0 and calls==0


def test_control_thread_can_reset_inference_tensors(monkeypatch, tmp_path):
    import json
    import threading
    torch = pytest.importorskip('torch')
    from r9700_vllm.moe import deepseek_expert_profile as profile
    recorder = profile.Recorder.__new__(profile.Recorder)
    recorder.output = tmp_path / 'routes.json'
    recorder.identity = {}
    recorder.device = 0
    recorder.lock = threading.RLock()
    recorder.epoch = 0
    recorder.command = None
    with torch.inference_mode():
        recorder.counters = {0: {p: torch.ones(2,385,dtype=torch.int64)
                                for p in ('all','decode','prefill')}}
    command = {'action': 'reset', 'id': 'test-reset'}
    (tmp_path / 'routes.json.control').write_text(json.dumps(command))
    sleeps = iter([None])
    monkeypatch.setattr(profile.time, 'sleep', lambda _: next(sleeps))
    monkeypatch.setattr(torch.cuda, 'set_device', lambda _: None)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    with pytest.raises(StopIteration):
        recorder.control()
    result = json.loads(recorder.output.read_text())
    assert result['command'] == command and result['epoch'] == 1
    assert result['rows'][0]['decode']['calls'] == 0
    assert not any(result['rows'][0]['decode']['slots'])
