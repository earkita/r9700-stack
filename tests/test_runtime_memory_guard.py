"""The host guard must survive a slow Docker daemon under memory pressure."""
import importlib.util
from pathlib import Path
import subprocess
from unittest.mock import Mock


spec = importlib.util.spec_from_file_location(
    'memory_guard', Path(__file__).resolve().parents[1] / 'serve/runtime_memory_guard.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def test_inspect_timeout_does_not_skip_next_host_check(monkeypatch, tmp_path):
    info = {'Id': 'immutable-id', 'HostConfig': {'OomScoreAdj': 500}}
    inspect = Mock(side_effect=[info, subprocess.TimeoutExpired('docker', 5)])
    snapshot = Mock(side_effect=[{'host': {'MemAvailable': 8 * 2**30}},
                                 {'host': {'MemAvailable': 2 * 2**30}}])
    stop = Mock()
    monkeypatch.setattr(guard, 'inspect_container', inspect)
    monkeypatch.setattr(guard, 'memory_snapshot', snapshot)
    monkeypatch.setattr(guard, 'stop_container', stop)
    monkeypatch.setattr(guard.time, 'sleep', lambda _: None)
    monkeypatch.setattr('sys.argv', ['guard', 'name', '--log', str(tmp_path / 'guard.jsonl')])
    guard.main()
    assert snapshot.call_count == 2
    assert all(c.kwargs == {'include_gpu': False} for c in snapshot.call_args_list)
    assert stop.call_args.args[:2] == ('immutable-id', 2)
    assert 'inspect_timeout' in (tmp_path / 'guard.jsonl').read_text()


def test_kill_timeout_still_checks_container_exit(monkeypatch):
    run = Mock(side_effect=subprocess.TimeoutExpired('docker', 5))
    monkeypatch.setattr(guard.subprocess, 'run', run)
    monkeypatch.setattr(guard, 'inspect_container', lambda _: {
        'State': {'Running': False, 'ExitCode': 143}})
    events = []
    guard.stop_container('immutable-id', 2, events.append)
    assert any(e['action'] == 'kill_timeout' for e in events)
    assert events[-1] == {'action': 'stopped', 'exit_code': 143}
