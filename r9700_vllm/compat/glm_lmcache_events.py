"""Conservative ROCm completion protocol for the opt-in GLM cache service.

HIP memory IPC is retained. Event IPC is replaced by an acknowledgement sent
only after the local event completes. Both endpoints must install this backend;
foreign handles are rejected. This sacrifices transfer/compute overlap, not
ordering, and does not change the model's normal (non-offload) execution.
"""
from importlib.metadata import version


class CompletedEvent:
    def wait(self, stream=None):
        pass

    def query(self):
        return True

    def synchronize(self):
        pass


class SynchronizedEvents:
    device_type = 'cuda'
    HANDLE = b'r9700-glm-completed-v1'

    def __init__(self, cuda):
        self.cuda = cuda

    def check_event_support(self, device):
        if not callable(getattr(self.cuda, 'Event', None)):
            raise RuntimeError('GLM cache requires local GPU events')

    def create_event(self, device):
        with self.cuda.device(device):
            return self.cuda.Event(interprocess=False)

    def record_event(self, event, stream):
        event.record(stream)

    def export_event(self, event, device):
        # Never acknowledge pending GPU writes or signal success after an error.
        event.synchronize()
        return self.HANDLE

    def import_event(self, handle, device):
        if handle != self.HANDLE:
            raise RuntimeError('GLM cache completion protocol mismatch; update both endpoints')
        return CompletedEvent()

    def wait_event(self, event, stream):
        event.wait(stream)

    def query_event(self, event):
        return event.query()

    def synchronize_event(self, event, device):
        event.synchronize()


def install():
    import torch
    from lmcache.v1.platform import get_device_spec

    if not torch.version.hip or version('lmcache') != '0.5.5':
        raise RuntimeError('GLM cache completion adapter requires ROCm / LMCache 0.5.5')
    spec = get_device_spec('cuda')
    if not hasattr(spec, '_event_backend_cache'):
        raise RuntimeError('Unsupported LMCache event backend interface')
    if not isinstance(spec._event_backend_cache, SynchronizedEvents):
        spec._event_backend_cache = SynchronizedEvents(torch.cuda)
