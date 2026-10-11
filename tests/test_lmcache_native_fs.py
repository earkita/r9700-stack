"""CPU-only integration test; run inside the pinned LMCache image.

python3 -m unittest discover -s tests -p test_lmcache_native_fs.py -v
No GPU devices or model weights are required.
"""
import select
import tempfile
import time
import unittest
from pathlib import Path

try:
    from lmcache.lmcache_fs import LMCacheFSClient
    from lmcache.v1.distributed.api import ObjectKey
    from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import _object_key_to_filename
    from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import _object_key_to_string
except ImportError:
    LMCacheFSClient = None


@unittest.skipIf(LMCacheFSClient is None, 'requires the pinned LMCache native image')
class NativeFSContract(unittest.TestCase):
    def until(self, predicate):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            result = predicate()
            if result is not None and result is not False:
                return result
            time.sleep(.01)
        self.fail('Adapter completion timed out')

    def test_native_lru_preserves_recent_and_locked_keys(self):
        from types import SimpleNamespace
        from lmcache.v1.distributed.config import EvictionConfig
        from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import NativeConnectorL2Adapter
        from lmcache.v1.distributed.storage_controllers.eviction_controller import L2AdapterEvictionState, L2EvictionController
        with tempfile.TemporaryDirectory() as directory:
            adapter = NativeConnectorL2Adapter(LMCacheFSClient(directory, 1, '', False, 0),
                                              max_capacity_gb=4*4096/2**30)
            state = L2AdapterEvictionState(0, adapter, EvictionConfig('LRU', .7, .34))
            controller = L2EvictionController([state])
            keys = [ObjectKey(bytes([i])*32, 'model', 0, 0, 'session') for i in range(3)]
            obj = SimpleNamespace(byte_array=memoryview(bytearray(4096)), get_size=lambda:4096)
            try:
                for key in keys:
                    task = adapter.submit_store_task([key], [obj])
                    self.until(lambda: adapter.pop_completed_store_tasks().get(task))
                self.until(lambda: state.eviction_policy.get_num_tracked_keys() == 3)
                task = adapter.submit_lookup_and_lock_task([keys[0]], {})
                self.until(lambda: adapter.query_lookup_and_lock_result(task))
                # Direct deletion must respect the lookup lock.
                adapter.delete([keys[0]])
                self.assertTrue((Path(directory)/_object_key_to_filename(keys[0])).exists())
                task = adapter.submit_load_task([keys[0]], [obj])
                self.until(lambda: adapter.query_load_result(task))
                adapter.submit_unlock([keys[0]])
                self.until(lambda: state.eviction_policy.get_eviction_candidates(1) == [keys[1]])
                controller._check_and_evict_global(state)
                self.assertFalse((Path(directory)/_object_key_to_filename(keys[1])).exists())
                self.assertTrue((Path(directory)/_object_key_to_filename(keys[0])).exists())
                self.assertTrue((Path(directory)/_object_key_to_filename(keys[2])).exists())
                self.assertEqual(adapter.get_usage().total_bytes_used, 8192)
            finally:
                adapter.close()

    def wait(self, client, task):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            for result in client.drain_completions():
                if result[0] == task:
                    self.assertTrue(result[1], result[2])
                    return result
            select.select([client.event_fd()], [], [], .05)
        self.fail('Native FS completion timed out')

    def test_salted_groups_ranks_roundtrip_and_reopen(self):
        keys = [ObjectKey(chunk_hash=bytes.fromhex('01'*32), model_name='test/model',
                          kv_rank=rank, object_group_id=group, cache_salt=salt)
                for rank in (0, 7) for group in (0, 1, 2)
                for salt in ('', 'session-a', 'session-b')]
        strings = [_object_key_to_string(key) for key in keys]
        buffers = [memoryview(bytearray([i+1]*4096)) for i in range(len(keys))]
        with tempfile.TemporaryDirectory() as directory:
            client = LMCacheFSClient(directory, 2, '', False, 0)
            try:
                self.wait(client, client.submit_batch_set(strings, buffers))
                self.assertEqual({p.name for p in Path(directory).glob('*.data')},
                                 {_object_key_to_filename(k) for k in keys})
            finally:
                client.close()
            client = LMCacheFSClient(directory, 2, '', False, 0)
            try:
                exists = self.wait(client, client.submit_batch_exists(strings))
                self.assertEqual(list(exists[3]), [True]*len(keys))
                loaded = [memoryview(bytearray(4096)) for _ in keys]
                self.wait(client, client.submit_batch_get(strings, loaded))
                self.assertEqual([bytes(b) for b in loaded], [bytes(b) for b in buffers])
                self.wait(client, client.submit_batch_delete([strings[0]]))
                exists = self.wait(client, client.submit_batch_exists(strings))
                self.assertEqual(list(exists[3]), [False]+[True]*(len(keys)-1))
            finally:
                client.close()

    def test_legacy_filenames_unchanged(self):
        keys = ['model@00000000@abcd', 'model@00000000@abcd@salt']
        with tempfile.TemporaryDirectory() as directory:
            client = LMCacheFSClient(directory, 1, '', False, 0)
            try:
                buffers = [memoryview(bytearray(4096)) for _ in keys]
                self.wait(client, client.submit_batch_set(keys, buffers))
                self.assertEqual({p.name for p in Path(directory).glob('*.data')},
                                 {'model@0x00000000@abcd.data', 'model@0x00000000@abcd@salt.data'})
            finally:
                client.close()
