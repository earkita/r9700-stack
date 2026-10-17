"""Run in the pinned SGLang/LMCache image; no GPU or server is required."""

from threading import Lock
from unittest.mock import Mock

import pytest


native = pytest.importorskip(
    "lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter"
)


@pytest.mark.parametrize("protected_by", ["locked", "lookup", "store"])
def test_l2_delete_keeps_in_use_key(protected_by):
    adapter = object.__new__(native.NativeConnectorL2Adapter)
    adapter._has_delete = True
    adapter._lock = Lock()
    adapter._locked_keys = {"key": 1} if protected_by == "locked" else {}
    adapter._pending_ops = (
        {1: (adapter._OP_LOOKUP, 9, 1, ["key"])}
        if protected_by == "lookup"
        else {}
    )
    adapter._pending_store_sizes = (
        {2: (["key"], [4096])} if protected_by == "store" else {}
    )
    adapter._client = Mock()

    adapter.delete(["key"])

    adapter._client.submit_batch_delete.assert_not_called()
