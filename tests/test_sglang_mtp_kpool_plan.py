# SPDX-License-Identifier: Apache-2.0
"""Exercise the public KPool plan wrapper on gfx1201, including graph replay.

The low-level kernel alone misses the HIP early-return regression. Test MTP
1..7 and C1/C2/C4, distinct slots, history/page boundaries, then mutate inputs
in-place across replay. No model weights or performance timing are involved.
"""
import argparse
import json
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3], choices=range(1, 8))
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2], choices=[1, 2, 4])
    args = parser.parse_args()
    import torch
    from sglang.srt.layers.attention.dsa.kpool_plan import (
        _alloc_kpool_write_plan_buffers, update_kpool_write_plan,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    assert torch.version.hip and torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] == 'gfx1201'
    results = []
    for batch in args.batches:
        for depth in args.depths:
            n = depth + 1
            plan = _alloc_kpool_write_plan_buffers(
                max_bs=batch, num_draft_tokens=n, pool_size=4,
                device=torch.device('cuda'), is_verify=True,
            )
            meta = SimpleNamespace(kpool_write_plan=plan)
            starts = torch.zeros(batch, dtype=torch.int32, device='cuda')
            reqs = torch.zeros(batch, dtype=torch.int64, device='cuda')
            # Deliberately non-identity physical pages, one expanded row/query.
            pages = (torch.arange(batch*n*20, device='cuda', dtype=torch.int32)
                     .reshape(batch*n, 20) + 7).contiguous()

            def update():
                update_kpool_write_plan(
                    meta, write_start=starts, req_pool_indices=reqs,
                    real_page_table=pages, pool_size=4, real_page_size=64,
                    num_draft_tokens=n, forward_mode=ForwardMode.TARGET_VERIFY,
                    slots_per_page=64,
                )

            def set_inputs(start):
                starts.copy_(torch.tensor([start + 61*i for i in range(batch)], device='cuda', dtype=torch.int32))
                reqs.copy_(torch.tensor([7 + start % 3 + 4*i for i in range(batch)], device='cuda', dtype=torch.int64))

            def check():
                expected_seq = starts[:, None] + torch.arange(1, n+1, device='cuda')[None, :]
                torch.testing.assert_close(plan.req, reqs, atol=0, rtol=0)
                torch.testing.assert_close(plan.write_start, starts, atol=0, rtol=0)
                torch.testing.assert_close(plan.tail_logical_start, starts//4*4, atol=0, rtol=0)
                torch.testing.assert_close(plan.seqlens_per_q, expected_seq.flatten().int(), atol=0, rtol=0)
                torch.testing.assert_close(plan.pool_seqlens_per_q, (expected_seq//4).flatten().int(), atol=0, rtol=0)
                pool_ids = starts[:, None]//4 + torch.arange(plan.write_loc.shape[1], device='cuda')[None, :]
                columns = pool_ids//64*4
                expected_loc = pages[torch.arange(batch, device='cuda')[:, None]*n, columns].long()*64 + pool_ids%64
                torch.testing.assert_close(plan.write_loc, expected_loc, atol=0, rtol=0)

            for start in (0, 3, 63, 186, 255, 1023):
                set_inputs(start); update(); check()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                update()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                update()
            for start in (186, 3, 255, 1023, 0):
                set_inputs(start); graph.replay(); check()
            results.append(dict(batch=batch, depth=depth, eager_cases=6,
                                graph_replays=5, metadata_exact=True))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
