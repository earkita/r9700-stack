"""Bounded checkpoint-copy correctness; GPU checks require explicit opt-in."""
import mmap
import os
from pathlib import Path
import tempfile
import unittest

import torch
from r9700_vllm.compat.deepseek_load_buffer import BufferedLoadCopies, copy_tiles


class LoadBufferCPU(unittest.TestCase):
    def test_tiles_cover_strided_destination_and_broadcast(self):
        for source in (torch.arange(35).reshape(5, 7).T,
                       torch.arange(5).reshape(1, 5)):
            output = torch.full((14, 10), -1, dtype=torch.int64)
            destination = output[::2, ::2]
            visits = 0
            for dst, src in copy_tiles(destination, source, 3):
                self.assertLessEqual(src.numel(), 3)
                dst.copy_(src)
                visits += dst.numel()
            self.assertEqual(visits, destination.numel())
            torch.testing.assert_close(destination, source.expand(7, 5))
            self.assertTrue((output[1::2] == -1).all())
            self.assertTrue((output[:, 1::2] == -1).all())

    def test_cpu_copies_do_not_allocate_buffer_and_scope_exits_on_error(self):
        mode = BufferedLoadCopies(64)
        with self.assertRaisesRegex(ValueError, 'sentinel'):
            with mode:
                src = torch.arange(21)
                dst = torch.empty_like(src)
                dst.copy_(src)
                torch.testing.assert_close(dst, src)
                self.assertIsNone(mode.buffer)
                raise ValueError('sentinel')
        self.assertIsNone(mode.buffer)
        self.assertEqual(mode.copies, 0)
        torch.testing.assert_close(torch.ones(2).clone(), torch.ones(2))

    def test_staging_bytes_are_exact_and_allocation_is_reused(self):
        mode = BufferedLoadCopies(64)
        for dtype in (torch.uint8, torch.bfloat16, torch.float32,
                      torch.float8_e4m3fn, torch.float8_e8m0fnu):
            # Raw-byte fixtures include scale codes and special FP8 encodings.
            raw = torch.arange(240, dtype=torch.uint8)
            source = raw.view(dtype)
            target = torch.empty_like(source)
            mode._copy(target, source)
            pointer = mode.buffer.data_ptr()
            mode._copy(target, source)
            self.assertEqual(pointer, mode.buffer.data_ptr())
            self.assertEqual(mode.buffer.numel(), 64)
            self.assertLessEqual(mode.peak_tile_bytes, 64)
            self.assertTrue(torch.equal(target.view(torch.uint8), raw))


@unittest.skipUnless(os.environ.get('R9700_DEEPSEEK_GPU_TEST') == '1',
                     'Explicit gfx1201 GPU test opt-in required')
class LoadBufferGPU(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.version.hip)
        self.assertIn('gfx1201', torch.cuda.get_device_properties(0).gcnArchName)

    def test_mmap_copies_tp_slices_cast_broadcast_and_to(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.bin'
            path.write_bytes(bytes(range(256)) * 128)
            with path.open('rb') as file:
                mapping = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_COPY)
                source = torch.frombuffer(mapping, dtype=torch.uint8).reshape(128, 256)
                for dtype in (torch.uint8, torch.bfloat16, torch.float8_e4m3fn,
                              torch.float8_e8m0fnu):
                    typed = source.view(dtype)
                    for rank in range(8):
                        shard = typed.chunk(8, dim=1)[rank]  # non-contiguous w2
                        holder = torch.empty((shard.shape[0]*2, shard.shape[1]*2),
                                             dtype=dtype, device='cuda')
                        destination = holder[::2, ::2]
                        with BufferedLoadCopies(256) as mode:
                            destination.copy_(shard, non_blocking=True)
                            copied = shard.to('cuda', non_blocking=True)
                        self.assertEqual(mode.copies, 2)
                        self.assertGreater(mode.tiles, 2)
                        self.assertLessEqual(mode.peak_tile_bytes, 256)
                        self.assertIsNone(mode.buffer)
                        expected = shard.contiguous().view(torch.uint8)
                        for actual in (destination, copied):
                            self.assertTrue(torch.equal(actual.cpu().contiguous().view(
                                torch.uint8), expected))
                del shard, typed, source
                mapping.close()
        src = torch.linspace(-10, 10, 17, dtype=torch.bfloat16)
        dst = torch.empty(5, 17, device='cuda', dtype=torch.float32)
        with BufferedLoadCopies(64) as mode:
            dst.copy_(src)
            cast = src.to(device='cuda', dtype=torch.float32)
        torch.testing.assert_close(dst.cpu(), src.float().expand(5, 17), rtol=0, atol=0)
        torch.testing.assert_close(cast.cpu(), src.float(), rtol=0, atol=0)

    def test_exception_releases_allocated_buffer(self):
        mode = BufferedLoadCopies(64)
        with self.assertRaisesRegex(ValueError, 'sentinel'):
            with mode:
                torch.ones(32).to('cuda')
                self.assertIsNotNone(mode.buffer)
                raise ValueError('sentinel')
        self.assertIsNone(mode.buffer)
        before = mode.copies
        torch.ones(32).to('cuda')
        self.assertEqual(mode.copies, before)


if __name__ == '__main__':
    unittest.main()
