# SPDX-License-Identifier: Apache-2.0
"""Exclusive GPU probe: C4 KPool rollback across physical pages and request slots.

Compare batched tentative writes/commit/changed continuation with sequential
accepted-token writes. Distinct nonidentity physical pages and noncontiguous
request slots catch cross-request contamination; tentative future bytes are
not committed state. No model throughput or answer-quality qualification.
"""
import argparse
import json
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--depths', type=int, nargs='+', default=[3, 4, 5], choices=range(1, 8))
    parser.add_argument('--batches', type=int, nargs='+', default=[4], choices=[1, 2, 4])
    args = parser.parse_args()
    import torch
    from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
        INDEX_HEAD_DIM, kpool_write_tail_and_maybe_compress,
    )
    from sglang.srt.layers.attention.dsa.kpool_plan import (
        _alloc_kpool_write_plan_buffers, update_kpool_write_plan,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    assert torch.version.hip and torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] == 'gfx1201'
    torch.manual_seed(20261016)
    dim = INDEX_HEAD_DIM
    ape = torch.randn(4, dim, device='cuda', dtype=torch.float32)
    results = []
    for batch in args.batches:
        # Reverse/permuted page IDs, reserving physical page0 as a sentinel.
        page_ids = [[2*batch-2*r, 2*r+1] for r in range(batch)]
        slots = [2*r+1 for r in range(batch)]
        for depth in args.depths:
            n = depth+1
            ring = 4+n
            pool = SimpleNamespace(page_size=64, index_head_dim=dim,
                                   slots_per_page=64, index_kpool=4,
                                   tail_extra_slots=n, quant_block_size=128)

            def state():
                return [torch.zeros(2*batch+1, 64*dim+64*4, device='cuda', dtype=torch.uint8),
                        torch.zeros(2*batch+1, ring, dim, device='cuda', dtype=torch.bfloat16),
                        torch.zeros(2*batch+1, ring, dim, device='cuda', dtype=torch.bfloat16)]

            def write(dst, reqs, starts, keys, scores, mask_last=False):
                count = keys.shape[1]
                bs = len(reqs)
                plan = _alloc_kpool_write_plan_buffers(
                    max_bs=bs, num_draft_tokens=count, pool_size=4,
                    device=torch.device('cuda'), is_verify=True)
                pages = torch.tensor([[page_ids[r][0]]*4+[page_ids[r][1]]*4
                                      for r in reqs for _ in range(count)],
                                     device='cuda', dtype=torch.int32)
                update_kpool_write_plan(
                    SimpleNamespace(kpool_write_plan=plan),
                    write_start=torch.tensor(starts, device='cuda', dtype=torch.int32),
                    req_pool_indices=torch.tensor([slots[r] for r in reqs], device='cuda', dtype=torch.int64),
                    real_page_table=pages, pool_size=4, real_page_size=64,
                    num_draft_tokens=count, forward_mode=ForwardMode.TARGET_VERIFY,
                    slots_per_page=64)
                out_locs = torch.ones(bs*count, device='cuda', dtype=torch.int64)
                if mask_last:
                    out_locs[-count:] = 0
                kpool_write_tail_and_maybe_compress(
                    pool=pool, buf=dst[0], key=keys.reshape(-1,dim),
                    score=scores.reshape(-1,dim), tail_k=dst[1], tail_score=dst[2],
                    ape=ape, req_pool_indices=plan.req, write_start=plan.write_start,
                    tail_logical_start=plan.tail_logical_start, write_loc=plan.write_loc,
                    out_cache_loc=out_locs, num_draft_tokens=count, round_scale=False)

            def sequential(dst, starts, keys, scores, counts):
                for i in range(max(counts)):
                    active = [r for r in range(batch) if i < counts[r]]
                    write(dst, active, [starts[r]+i for r in active],
                          keys[active,i:i+1].contiguous(), scores[active,i:i+1].contiguous())

            def check(actual, reference, lengths):
                for r, length in enumerate(lengths):
                    logical = torch.arange(length//4, device='cuda')
                    physical = torch.tensor(page_ids[r], device='cuda')[logical//64]
                    offset = logical%64
                    key_bytes = offset[:,None]*dim+torch.arange(dim,device='cuda')[None,:]
                    scale_bytes = 64*dim+offset[:,None]*4+torch.arange(4,device='cuda')[None,:]
                    for col in (key_bytes, scale_bytes):
                        torch.testing.assert_close(actual[0][physical[:,None],col],
                                                   reference[0][physical[:,None],col],atol=0,rtol=0)
                    live = torch.arange(length//4*4,length,device='cuda')%ring
                    for j in (1,2):
                        torch.testing.assert_close(actual[j][slots[r],live],reference[j][slots[r],live],atol=0,rtol=0)
                # No request or physical mapping ever owns slot/page zero.
                for tensor in actual:
                    assert not bool(tensor[0].any()), 'Reserved sentinel was overwritten'

            cases = 0
            for prefixes in ([0,1,3,4], [61,63,64,65], [252,253,254,255]):
                prefixes = prefixes[:batch]
                maximum = max(prefixes)
                history = torch.randn(batch,maximum,dim,device='cuda',dtype=torch.bfloat16)
                history_scores = torch.randn_like(history)
                keys = torch.randn(batch,2*n,dim,device='cuda',dtype=torch.bfloat16)
                scores = torch.randn_like(keys)
                initial = state()
                sequential(initial,[0]*batch,history,history_scores,prefixes)
                for accepted in range(1,n+1):
                    counts = [1+(accepted-1+r)%n for r in range(batch)]
                    for accepted_next in range(1,n+1):
                        next_counts = [1+(accepted_next-1+2*r)%n for r in range(batch)]
                        actual = [x.clone() for x in initial]
                        reference = [x.clone() for x in initial]
                        write(actual,list(range(batch)),prefixes,keys[:,:n].contiguous(),scores[:,:n].contiguous())
                        sequential(reference,prefixes,keys[:,:n],scores[:,:n],counts)
                        lengths = [p+c for p,c in zip(prefixes,counts)]
                        check(actual,reference,lengths)
                        write(actual,list(range(batch)),lengths,keys[:,n:].contiguous(),scores[:,n:].contiguous())
                        sequential(reference,lengths,keys[:,n:],scores[:,n:],next_counts)
                        check(actual,reference,[p+c for p,c in zip(lengths,next_counts)])
                        cases += 1
                # A masked request must preserve its entire old ring and pages,
                # even while the other requests execute their normal writes.
                masked = [x.clone() for x in initial]
                write(masked,list(range(batch)),prefixes,keys[:,:n].contiguous(),scores[:,:n].contiguous(),mask_last=True)
                last = batch-1
                for j in (1,2):assert torch.equal(masked[j][slots[last]],initial[j][slots[last]])
                for page in page_ids[last]:assert torch.equal(masked[0][page],initial[0][page])
            results.append(dict(batch=batch,depth=depth,forced_prefix_pairs=cases,
                                distinct_request_slots=slots,physical_pages=page_ids,
                                committed_pools_exact=True,live_tails_exact=True,
                                changed_continuation_exact=True,masked_request_unchanged=True))
    print(json.dumps(results,indent=2),flush=True)


if __name__ == '__main__':
    main()
