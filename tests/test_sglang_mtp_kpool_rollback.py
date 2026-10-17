# SPDX-License-Identifier: Apache-2.0
"""KPool speculative ring writes: reject suffixes, then overwrite and continue.

Check state against sequential one-token writes for every accepted prefix at
MTP depths 1..5. Tentative future cache slots may differ; compare only the
committed compressed pools and the live incomplete tail. No throughput test.
"""
import argparse
import json
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3], choices=range(1, 6))
    args = parser.parse_args()
    import torch
    from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
        INDEX_HEAD_DIM, kpool_write_tail_and_maybe_compress,
    )
    from sglang.srt.layers.attention.dsa.kpool_plan import (
        _alloc_kpool_write_plan_buffers, update_kpool_write_plan,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    torch.manual_seed(20261015)
    dim = INDEX_HEAD_DIM
    ape = torch.randn(4, dim, device='cuda', dtype=torch.float32)
    results = []
    for depth in args.depths:
        n = depth + 1
        pool = SimpleNamespace(page_size=64, index_head_dim=dim,
                               slots_per_page=64, index_kpool=4,
                               tail_extra_slots=n, quant_block_size=128)
        ring = 4+n

        def state():
            return [torch.zeros(1, 64*dim+64*4, dtype=torch.uint8, device='cuda'),
                    torch.zeros(1, ring, dim, dtype=torch.bfloat16, device='cuda'),
                    torch.zeros(1, ring, dim, dtype=torch.bfloat16, device='cuda')]

        def write(dst, start, keys, scores):
            count = keys.shape[0]
            plan = _alloc_kpool_write_plan_buffers(
                max_bs=1, num_draft_tokens=count, pool_size=4,
                device=torch.device('cuda'), is_verify=True,
            )
            update_kpool_write_plan(
                SimpleNamespace(kpool_write_plan=plan),
                write_start=torch.tensor([start], dtype=torch.int32, device='cuda'),
                req_pool_indices=torch.tensor([0], dtype=torch.int64, device='cuda'),
                real_page_table=torch.zeros(count, 4, dtype=torch.int32, device='cuda'),
                pool_size=4, real_page_size=64, num_draft_tokens=count,
                forward_mode=ForwardMode.TARGET_VERIFY, slots_per_page=64,
            )
            kpool_write_tail_and_maybe_compress(
                pool=pool, buf=dst[0], key=keys, score=scores, tail_k=dst[1],
                tail_score=dst[2], ape=ape, req_pool_indices=plan.req,
                write_start=plan.write_start, tail_logical_start=plan.tail_logical_start,
                write_loc=plan.write_loc,
                out_cache_loc=torch.arange(start+1, start+count+1, dtype=torch.int64, device='cuda'),
                num_draft_tokens=count, round_scale=False,
            )

        def sequential(dst, start, keys, scores):
            for i in range(len(keys)):
                write(dst, start+i, keys[i:i+1], scores[i:i+1])

        def check(candidate, reference, length):
            closed = length//4
            torch.testing.assert_close(candidate[0][:, :closed*dim], reference[0][:, :closed*dim], atol=0, rtol=0)
            torch.testing.assert_close(candidate[0][:, 64*dim:64*dim+closed*4], reference[0][:, 64*dim:64*dim+closed*4], atol=0, rtol=0)
            live = torch.arange(length//4*4, length, device='cuda') % ring
            for j in (1, 2):
                torch.testing.assert_close(candidate[j][:, live], reference[j][:, live], atol=0, rtol=0)

        cases = 0
        for prefix in (0, 1, 3, 4, 7, 10):
            keys = torch.randn(prefix+2*n, dim, device='cuda', dtype=torch.bfloat16)
            scores = torch.randn_like(keys)
            initial = state();sequential(initial, 0, keys[:prefix], scores[:prefix])
            for accepted in range(1, n+1):
                for accepted_next in range(1, n+1):
                    actual = [x.clone() for x in initial]
                    ref = [x.clone() for x in initial]
                    # Verify writes all proposals; the rejected suffix remains
                    # tentative. Commit by logical length, as in the runtime.
                    write(actual, prefix, keys[prefix:prefix+n], scores[prefix:prefix+n])
                    sequential(ref, prefix, keys[prefix:prefix+accepted], scores[prefix:prefix+accepted])
                    check(actual, ref, prefix+accepted)
                    # New branch has different keys/scores from the rejected
                    # proposals. It must rebuild future pools from live history.
                    new_keys = keys[prefix+n:prefix+2*n]
                    new_scores = scores[prefix+n:prefix+2*n]
                    write(actual, prefix+accepted, new_keys, new_scores)
                    sequential(ref, prefix+accepted, new_keys[:accepted_next], new_scores[:accepted_next])
                    check(actual, ref, prefix+accepted+accepted_next)
                    cases += 1
        results.append(dict(depth=depth, accepted_prefix_pairs=cases,
                            committed_cache_exact=True, live_tail_exact=True,
                            continuation_exact=True))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
