"""Conservative ROCm completion events for the GLM LMCache MP candidate.

Memory IPC remains unchanged. Each producer synchronizes its *local* event
before returning an acknowledgement over the request channel. A consumer may
then treat that acknowledgement as a completed event. Both processes must use
this backend; a foreign event handle fails closed. This deliberately gives up
copy/compute overlap to establish ordering on the pinned ROCm stack.
"""


class CompletedEvent:
    def wait(self, stream=None):
        pass

    def query(self):
        return True

    def synchronize(self):
        pass


class SynchronizedEvents:
    device_type = "cuda"
    HANDLE = b"r9700-sglang-glm-completed-v1"

    def __init__(self, cuda):
        self.cuda = cuda

    def check_event_support(self, device):
        if not callable(getattr(self.cuda, "Event", None)):
            raise RuntimeError("GLM LMCache requires local GPU events")

    def create_event(self, device):
        with self.cuda.device(device):
            return self.cuda.Event(interprocess=False)

    def record_event(self, event, stream):
        event.record(stream)

    def export_event(self, event, device):
        # Never acknowledge pending writes or mask a failed GPU operation.
        event.synchronize()
        return self.HANDLE

    def import_event(self, handle, device):
        if handle != self.HANDLE:
            raise RuntimeError(
                "GLM LMCache completion protocol mismatch; update both endpoints"
            )
        return CompletedEvent()

    def wait_event(self, event, stream):
        event.wait(stream)

    def query_event(self, event):
        return event.query()

    def synchronize_event(self, event, device):
        event.synchronize()
