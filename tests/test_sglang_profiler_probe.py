"""Observe native profiler env changes without importing or using a GPU."""
import ctypes
import importlib.util
import os
from pathlib import Path

spec=importlib.util.spec_from_file_location('probe',Path(__file__).parents[1]/'tuning/sglang_profiler_probe.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def test_native_setenv_is_visible_but_authentication_environment_is_not():
    # A stale Python mapping hid native changes in the initial probe. The test
    # reproduces that discrepancy and verifies the diagnostic's output filter.
    public=b'ROCPROFILER_R9700_TEST_ENV';private=b'R9700_TEST_AUTH_TOKEN'
    libc=ctypes.CDLL(None)
    libc.setenv.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int]
    libc.unsetenv.argtypes=[ctypes.c_char_p]
    assert public.decode() not in os.environ and private.decode() not in os.environ
    try:
        assert libc.setenv(public,b'native-only',1)==0
        assert libc.setenv(private,b'never-export-this',1)==0
        assert public.decode() not in os.environ
        observed=probe.profiler_environment()
        assert observed[public.decode()]=='native-only'
        assert private.decode() not in observed
    finally:
        libc.unsetenv(public);libc.unsetenv(private)
