# SPDX-License-Identifier: Apache-2.0
"""Opt-in profiler child-start workaround; no model/attention/quantization changes.

ROCm's native HIP initialization sets HSA_TOOLS_ROCPROFILER_V1_TOOLS=0 in the
parent's libc environment. Passing that value into a new worker can suppress
its GPU profiler initialization. Python os.environ does not see native setenv.
Only a diagnostic launcher may enable this, around scheduler proc.start().
"""
from contextlib import contextmanager
import ctypes
import os

HSA_KEY = b'HSA_TOOLS_ROCPROFILER_V1_TOOLS'
REGISTER_KEY = b'ROCPROFILER_REGISTER_LIBRARY'


@contextmanager
def profile_child_environment():
    mode = os.environ.get('R9K_SGLANG_PROFILE_CHILD_ENV', '0')
    if mode not in ('0', 'hsa', 'both'):
        raise ValueError('R9K_SGLANG_PROFILE_CHILD_ENV must be 0, hsa or both')
    if mode == '0':
        yield
        return
    libc = ctypes.CDLL(None, use_errno=True)
    libc.getenv.argtypes = [ctypes.c_char_p];libc.getenv.restype = ctypes.c_char_p
    libc.unsetenv.argtypes = [ctypes.c_char_p];libc.unsetenv.restype = ctypes.c_int
    libc.setenv.argtypes = [ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int];libc.setenv.restype = ctypes.c_int
    keys = [HSA_KEY] + ([REGISTER_KEY] if mode == 'both' else [])
    saved = []
    try:
        for key in keys:
            value = libc.getenv(key)
            if value is None or (key == HSA_KEY and value != b'0'):
                continue
            saved.append((key, value))
            if libc.unsetenv(key):raise OSError(ctypes.get_errno(), 'Cannot clear profiler child environment')
        yield
    finally:
        for key,value in reversed(saved):
            if libc.setenv(key,value,1):raise OSError(ctypes.get_errno(), 'Cannot restore parent profiler environment')
