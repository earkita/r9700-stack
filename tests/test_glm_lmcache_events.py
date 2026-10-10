"""Completion messages must never precede the GPU writes they acknowledge."""
from unittest.mock import Mock
import pytest

from r9700_vllm.compat.glm_lmcache_events import SynchronizedEvents


def test_acknowledge_only_completed_writes():
    backend = SynchronizedEvents(Mock())
    event = Mock()
    event.synchronize.side_effect = RuntimeError('GPU failure')
    with pytest.raises(RuntimeError, match='GPU failure'):
        backend.export_event(event, 'cuda:3')
    event.synchronize.side_effect = None
    handle = backend.export_event(event, 'cuda:3')
    assert event.synchronize.call_count == 2
    remote = backend.import_event(handle, 'cuda:3')
    assert backend.query_event(remote)
    backend.wait_event(remote, object())
    backend.synchronize_event(remote, 'cuda:3')


def test_mixed_protocol_is_rejected():
    with pytest.raises(RuntimeError, match='protocol mismatch'):
        SynchronizedEvents(Mock()).import_event(b'foreign-ipc-event', 'cuda:0')
