"""GPU regression gates for DFlash state completion and rejection rollback."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")


def test_future_pools_are_excluded_from_earlier_verify_rows():
    from r9700_vllm.attn.glm_tail import expand_pools_and_append_tail
    ids = torch.tensor([[0, 1, 2, -1]] * 4, device="cuda", dtype=torch.int32)
    lengths = torch.tensor([7, 8, 9, 12], device="cuda", dtype=torch.int32)
    out = expand_pools_and_append_tail(ids, lengths, 4)
    expected = []
    for length in lengths.tolist():
        full = length // 4
        row = [p * 4 + k if 0 <= p < full else -1 for p in (0, 1, 2, -1) for k in range(4)]
        row += [full * 4 + k if k < length % 4 else -1 for k in range(3)]
        expected.append(row)
    torch.testing.assert_close(out, torch.tensor(expected, device="cuda", dtype=torch.int32))


@pytest.mark.parametrize("spec", [1, 4, 7])
def test_tail_rejection_matches_committed_sequence(spec):
    from vllm.models.glm5next.amd.ops import kpool_compress as old
    from r9700_vllm.attn import glm_tail as new
    torch.manual_seed(490 + spec)
    pool, dim, block = 4, 128, 1
    ring = ((pool + spec + pool - 1) // pool) * pool
    keys = torch.randn(24, dim, device="cuda", dtype=torch.bfloat16)
    scores = torch.randn_like(keys)
    ape = torch.randn(pool, dim, device="cuda")

    def buffers(capacity):
        return (torch.zeros(2, 64, dim + 4, device="cuda", dtype=torch.uint8),
                torch.zeros(3, 2, capacity, dim, device="cuda", dtype=torch.bfloat16))

    def seed(ops, tail, capacity):
        pos = torch.arange(4, 7, device="cuda", dtype=torch.int32)
        slots = block * capacity + pos % capacity
        ops.kpool_seed_tail_cache(tail, keys[4:7], scores[4:7], slots, pool)

    def update(ops, buf, tail, capacity, positions, k, s):
        pos = torch.tensor([positions], device="cuda", dtype=torch.int32)
        slots = torch.where(pos % pool == pool - 1, pos // pool, -1)
        ts = block * capacity + pos % capacity
        ops.kpool_decode_update_and_maybe_write_cache_batched(
            buf, tail, ts, k[None].contiguous(), s[None].contiguous(), ape, slots, pos, pool)

    ref, ref_tail = buffers(pool)
    seed(old, ref_tail, pool)
    update(old, ref, ref_tail, pool, [7], keys[7:8], scores[7:8])
    # A correct bonus at 6, then rejected proposals starting with completion 7.
    positions = list(range(6, 6 + spec + 1))
    draft_k, draft_s = keys[positions].clone(), scores[positions].clone()
    draft_k[1:] *= -3
    draft_s[1:] *= -2
    for ops, capacity in [(old, pool), (new, ring)]:
        buf, tail = buffers(capacity)
        seed(ops, tail, capacity)
        update(ops, buf, tail, capacity, positions, draft_k, draft_s)
        update(ops, buf, tail, capacity, [7], keys[7:8], scores[7:8])
        # Compare the entire first 16-token tile (including scales). Only pool
        # 1 has completed in these cases, except K7 which also writes pool 2.
        # Keys occupy a 16x16 tiled cache: select pool 1 without assuming
        # token-major storage. The scale sits after all page key bytes.
        offsets = torch.arange(dim, device="cuda")
        key_offsets = (offsets // 16) * 256 + 16 + offsets % 16
        indices = torch.cat([key_offsets, torch.arange(64 * dim + 4, 64 * dim + 8, device="cuda")])
        actual = buf[0].flatten()[indices]
        expected = ref[0].flatten()[indices]
        if ops is old and spec >= 4:
            assert not torch.equal(actual, expected), "regression must reproduce on upstream"
        else:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [1, 2, 5, 8, 257])
def test_aux_gpu_mhc_reference_and_input_immutability(rows):
    from vllm.model_executor.layers.mhc import MHCPostOp, hc_contract
    torch.manual_seed(720 + rows)
    n, hidden = 4, 4096
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(rows, n, hidden, device="cuda", dtype=torch.bfloat16)
    post = torch.rand(rows, n, 1, device="cuda")
    comb = torch.rand(rows, n, n, device="cuda").softmax(-1)
    originals = [t.clone() for t in (x, residual, post, comb)]
    from vllm.config import VllmConfig, set_current_vllm_config
    with set_current_vllm_config(VllmConfig()):
        actual = hc_contract(MHCPostOp()(x, residual, post, comb), n)
    # Independent FP64 expression, with the BF16 materialization boundary
    # present in the target's completed mHC state.
    ref = (torch.einsum("sij,sih->sjh", comb.double(), residual.double())
           + post.double() * x.double().unsqueeze(1)).bfloat16().mean(1)
    torch.testing.assert_close(actual, ref, atol=.015625, rtol=.008)
    for t, original in zip((x, residual, post, comb), originals):
        torch.testing.assert_close(t, original, rtol=0, atol=0)
