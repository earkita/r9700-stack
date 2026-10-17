"""Native child-env cleanup is scoped, opt-in and exception safe; CPU only."""
import ctypes
import pytest

from r9700_sglang.profile_worker_env import HSA_KEY, REGISTER_KEY, profile_child_environment


@pytest.fixture
def native():
    lib=ctypes.CDLL(None)
    lib.getenv.argtypes=[ctypes.c_char_p];lib.getenv.restype=ctypes.c_char_p
    lib.setenv.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int]
    lib.unsetenv.argtypes=[ctypes.c_char_p]
    saved={k:lib.getenv(k) for k in [HSA_KEY,REGISTER_KEY]}
    yield lib
    for k,v in saved.items():
        lib.unsetenv(k) if v is None else lib.setenv(k,v,1)


def test_default_leaves_native_environment_untouched(native,monkeypatch):
    monkeypatch.delenv('R9K_SGLANG_PROFILE_CHILD_ENV',raising=False)
    native.setenv(HSA_KEY,b'0',1)
    with profile_child_environment():assert native.getenv(HSA_KEY)==b'0'


def test_hsa_cleanup_restores_parent_after_child_start_exception(native,monkeypatch):
    monkeypatch.setenv('R9K_SGLANG_PROFILE_CHILD_ENV','hsa')
    native.setenv(HSA_KEY,b'0',1);native.setenv(REGISTER_KEY,b'/sdk/library.so',1)
    with pytest.raises(RuntimeError):
        with profile_child_environment():
            assert native.getenv(HSA_KEY) is None
            assert native.getenv(REGISTER_KEY)==b'/sdk/library.so'
            raise RuntimeError('simulated proc.start failure')
    assert native.getenv(HSA_KEY)==b'0'
    assert native.getenv(REGISTER_KEY)==b'/sdk/library.so'


def test_both_mode_nested_context_and_absent_values(native,monkeypatch):
    monkeypatch.setenv('R9K_SGLANG_PROFILE_CHILD_ENV','both')
    native.setenv(HSA_KEY,b'0',1);native.setenv(REGISTER_KEY,b'/sdk/library.so',1)
    with profile_child_environment():
        with profile_child_environment():
            assert native.getenv(HSA_KEY) is native.getenv(REGISTER_KEY) is None
        assert native.getenv(HSA_KEY) is native.getenv(REGISTER_KEY) is None
    assert native.getenv(HSA_KEY)==b'0' and native.getenv(REGISTER_KEY)==b'/sdk/library.so'


def test_explicit_nonzero_tool_setting_is_preserved(native,monkeypatch):
    monkeypatch.setenv('R9K_SGLANG_PROFILE_CHILD_ENV','hsa');native.setenv(HSA_KEY,b'1',1)
    with profile_child_environment():assert native.getenv(HSA_KEY)==b'1'


def test_unknown_mode_fails_before_start(monkeypatch):
    monkeypatch.setenv('R9K_SGLANG_PROFILE_CHILD_ENV','automatic')
    with pytest.raises(ValueError):
        with profile_child_environment():pytest.fail('unqualified start')
