"""CPU memory-bound planning; these checks do not qualify GPU arithmetic."""
import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location("indexer_prefill",Path(__file__).parents[1]/"r9700_sglang/indexer_prefill.py")
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def test_disabled_has_no_torch_import_or_metadata_requirement(monkeypatch):
    monkeypatch.delenv("R9K_SGLANG_INDEXER_PREFILL_ROWS",raising=False)
    assert mod.apply(None,None,None,None,None) is None


def test_256k_context_and_8192_queries_cannot_allocate_two_gib():
    chunks=mod.row_slices(8192,65536,8192)
    assert [(x.start,x.stop) for x in chunks]==[(0,8191),(8191,8192)]
    assert all((x.stop-x.start)*65536*4 < 2**31 for x in chunks)


def test_key_stride_alignment_and_nondivisible_tail_cover_every_query_once():
    chunks=mod.row_slices(2051,257,1024,byte_limit=512*4*700)
    assert [(x.start,x.stop) for x in chunks]==[(0,700),(700,1400),(1400,2051)]
    assert sum(x.stop-x.start for x in chunks)==2051


def test_no_row_fits_or_invalid_explicit_configuration_is_rejected(monkeypatch):
    with pytest.raises(ValueError,match="One aligned"):
        mod.row_slices(1,65536,1024,byte_limit=65536*4-1)
    monkeypatch.setenv("R9K_SGLANG_INDEXER_PREFILL_ROWS","auto")
    with pytest.raises(ValueError,match="0, 512"):
        mod.apply(None,None,None,None,None)
