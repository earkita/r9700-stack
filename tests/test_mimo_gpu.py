"""MiMo TP8 QKV, MXFP4 and BF16 DiffKV numerical gates (one free gfx1201)."""

import math
import unittest
import torch
from vllm.model_executor.models.mimo_v2 import _shard_fp8_qkv_proj
from r9700_vllm.attn.mimo_diffkv_ops import unified_attention_diffkv


class MiMoNumerics(unittest.TestCase):
    def test_qkv_tp4_to_tp8(self):
        torch.manual_seed(991)
        for kv in (4, 8):
            qheads, hd, vd, kdim = 64, 192, 128, 128
            chunks = []
            for c in range(4):
                pieces = []
                for n, dim, offset in [
                    (16, hd, c * 16),
                    (kv // 4, hd, 64 + c * (kv // 4)),
                    (kv // 4, vd, 80 + c * (kv // 4)),
                ]:
                    pieces.append(
                        torch.cat(
                            [
                                torch.randn(1, kdim).expand(dim, kdim).clone()
                                for h in range(n)
                            ]
                        )
                    )
                chunks.append(torch.cat(pieces))
            w = torch.cat(chunks).to(torch.float8_e4m3fn)
            scale_rows = math.ceil(chunks[0].shape[0] / 128)
            s = 0.5 + 1.5 * torch.rand((4 * scale_rows, 1))
            # Independent per-checkpoint-chunk dequantization catches scale
            # indexing errors at the unaligned 3392-row global boundaries.
            reference_chunks = [
                chunk.to(torch.float8_e4m3fn).float()
                * scales.repeat_interleave(128, 0)[: chunk.shape[0]]
                for chunk, scales in zip(chunks, s.chunk(4, dim=0))
            ]
            for rank in range(8):
                rw, rs = _shard_fp8_qkv_proj(w, s, qheads, kv, hd, vd, rank, 8, 4)
                deq = rw.float() * rs.repeat_interleave(128, 0)[
                    : rw.shape[0]
                ].repeat_interleave(128, 1)
                q = []
                ks = []
                vs = []
                for chunk in reference_chunks:
                    q.append(chunk[: 16 * hd])
                    ks.append(chunk[16 * hd : 16 * hd + (kv // 4) * hd])
                    vs.append(chunk[16 * hd + (kv // 4) * hd :])
                kh = rank if kv == 8 else rank // 2
                expected = torch.cat(
                    [
                        torch.cat(q)[rank * 8 * hd : (rank + 1) * 8 * hd],
                        torch.cat(ks)[kh * hd : (kh + 1) * hd],
                        torch.cat(vs)[kh * vd : (kh + 1) * vd],
                    ]
                )
                error = (deq - expected).norm() / expected.norm()
                self.assertLess(error.item(), 0.04, (kv, rank, error.item()))

    def test_diffkv(self):
        torch.manual_seed(321)
        for lens, qlens in [
            ([129], [1]),
            ([257, 143], [1, 1]),
            ([263], [8]),
            ([263, 139], [8, 8]),
            ([263, 139], [8, 3]),
            ([96, 139], [64, 8]),
        ]:
            for window in [-1, 127]:
                for sink in [False, True]:
                    self._case(lens, qlens, window, sink)

    def _case(self, lens, qlens, window, sink):
        n = len(lens)
        block = 16
        per = math.ceil(max(lens) / block)
        heads = 8
        cache = (
            torch.randn((n * per, block, 1, 320), device="cuda", dtype=torch.bfloat16)
            * 0.2
        )
        q = (
            torch.randn((sum(qlens), heads, 192), device="cuda", dtype=torch.bfloat16)
            * 0.2
        )
        out = torch.empty((sum(qlens), heads, 128), device="cuda", dtype=torch.bfloat16)
        starts = torch.tensor(
            [0] + list(torch.tensor(qlens).cumsum(0).tolist()),
            device="cuda",
            dtype=torch.int32,
        )
        seq = torch.tensor(lens, device="cuda", dtype=torch.int32)
        table = torch.arange(n * per, device="cuda", dtype=torch.int32).view(n, per)
        sinks = torch.randn(heads, device="cuda") if sink else None
        rows = max(32, sum(qlens))
        segments = 64
        scratch = torch.empty(
            (rows, heads, segments, 128), device="cuda", dtype=torch.float32
        )
        mx = torch.empty((rows, heads, segments), device="cuda")
        ex = torch.empty_like(mx)
        unified_attention_diffkv(
            q=q,
            k=cache[..., :192],
            v=cache[..., 192:],
            out=out,
            cu_seqlens_q=starts,
            seqused_k=seq,
            softmax_scale=192**-0.5,
            causal=True,
            window_size=(window, 0),
            block_table=table,
            softcap=0,
            sinks=sinks,
            max_seqlen_q=max(qlens),
            seq_threshold_3D=rows,
            num_par_softmax_segments=segments,
            softmax_segm_output=scratch,
            softmax_segm_max=mx,
            softmax_segm_expsum=ex,
        )
        refs = []
        offset = 0
        for i, (length, queries) in enumerate(zip(lens, qlens)):
            kv = cache[i * per : (i + 1) * per].reshape(-1, 1, 320)[:length, 0].float()
            for j in range(queries):
                pos = length - queries + j
                lo = max(0, pos - window) if window >= 0 else 0
                k = kv[lo : pos + 1, :192]
                v = kv[lo : pos + 1, 192:]
                scores = q[offset + j].float() @ k.T / math.sqrt(192)
                if sink:
                    probs = torch.softmax(torch.cat([scores, sinks[:, None]], 1), 1)[
                        :, :-1
                    ]
                else:
                    probs = torch.softmax(scores, 1)
                refs.append(probs @ v)
            offset += queries
        torch.testing.assert_close(out.float(), torch.stack(refs), atol=2e-3, rtol=2e-2)

    def test_moe_mimo_shapes(self):
        from test_moe_mxfp4 import run_case

        for rows in (1, 2, 8, 16, 65):
            self.assertTrue(run_case(8, rows, 8, 512, 4096, 4096, 256, seed=123)[0])


if __name__ == "__main__":
    unittest.main()
