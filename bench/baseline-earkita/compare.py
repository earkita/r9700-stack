#!/usr/bin/env python3
"""Compare completed focused K4/MTP3 runs without claiming task-level speedup."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


MATCHED_FIELDS = (
    'template_file', 'name', 'client', 'betterbench_commit', 'lengths',
    'concurrency', 'efforts', 'cache_states', 'warmup_batches',
    'measured_batches', 'output_tokens', 'quality_output_tokens',
    'quality_gate', 'temperature', 'top_p', 'top_k', 'seed', 'chunk_tokens',
    'required_max_model_len', 'required_kv_cache_bytes',
    'required_l1_gib', 'required_l2_gib', 'frozen_corpus_sha256',
    'allow_serial_cold_prefill_diagnostic', 'configured_l1_gib',
    'configured_l2_gib', 'l2_eviction', 'sources', 'versions', 'mapping',
    'pcie_expected', 'git_head',
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_path(identity):
    args = identity['args']
    if '--model' in args:
        return args[args.index('--model') + 1]
    return next((x for x in args if x.startswith('/models/')), None)


def model_mount(identity):
    return next((m['source'] for m in identity['mounts']
                 if m['destination'] == '/models'), None)


def power_limits(directory):
    raw = json.loads((directory/'hardware-before/power-limits.json').read_text())
    return {row['gpu']: row['limit']['ppt0']['socket_power_limit']['value']
            for row in raw['gpu_data']}


def hardware_signature(protocol):
    idle = protocol['idle_links']
    return dict(paths=idle['paths'], devices={key: {
        field: data[field] for field in ('max_link_speed', 'max_link_width', 'numa_node')}
        for key, data in idle['devices'].items()})


def read_leg(directory, profile):
    directory = directory.resolve()
    protocol_file = directory/'protocol.json'
    if digest(protocol_file) != (directory/'protocol.sha256').read_text().strip():
        raise ValueError(f'{profile}: protocol hash mismatch')
    protocol = json.loads(protocol_file.read_text())
    if protocol.get('template_file') != 'glm-mtp3-k4.json' or protocol.get('speculative_profile') != profile:
        raise ValueError(f'{profile}: wrong focused template or speculative profile')
    if protocol.get('quality_gate') != 'strict_exact_completed':
        raise ValueError(f'{profile}: exact completed-answer gate was not used')
    if json.loads((directory/'status.json').read_text()).get('status') != 'complete':
        raise ValueError(f'{profile}: run did not complete')
    report = json.loads((directory/'report.json').read_text())
    if report.get('status') != 'complete':
        raise ValueError(f'{profile}: report is not complete')
    for effort, corpus in protocol['corpora'].items():
        if digest(directory/corpus['file']) != corpus['sha256']:
            raise ValueError(f'{profile}: {effort} corpus changed')
        if corpus.get('source_sha256') != protocol['frozen_corpus_sha256']:
            raise ValueError(f'{profile}: {effort} corpus source differs')
    expected_groups = {f'{length}/C{count}/{effort}/{state}'
                       for length in protocol['lengths'] for count in protocol['concurrency']
                       for effort in protocol['efforts'] for state in protocol['cache_states']}
    if set(report['groups']) != expected_groups:
        raise ValueError(f'{profile}: missing or extra result group')
    for key, group in report['groups'].items():
        count = int(key.split('/')[1][1:])
        state = key.rsplit('/', 1)[1]
        if group['batches'] != protocol['measured_batches']:
            raise ValueError(f'{profile}: wrong repeat count for {key}')
        for metric in ('decode_tps', 'ttft_ms', 'effective_prompt_tps'):
            datum = group['per_request'].get(metric)
            if (not datum or datum['n'] != count * group['batches']
                    or not math.isfinite(datum['median']) or datum['median'] <= 0):
                raise ValueError(f'{profile}: missing per-request {metric} for {key}')
        aggregate = group.get('aggregate_e2e_output_tps', [])
        if len(aggregate) != group['batches'] or any(
                not math.isfinite(value) or value <= 0 for value in aggregate):
            raise ValueError(f'{profile}: aggregate output rate missing for {key}')
        if len(group.get('overlap_s', [])) != group['batches']:
            raise ValueError(f'{profile}: decode overlap evidence missing for {key}')
        if count > 1 and state != 'cold' and not group['decode_concurrency_qualified']:
            raise ValueError(f'{profile}: no simultaneous warm decode for {key}')
        if len(group.get('acceptance', [])) != group['batches'] or any(
                not a or a['ratio'] is None or not math.isfinite(a['ratio'])
                or not 0 <= a['ratio'] <= 1 or a['drafts'] <= 0
                for a in group['acceptance']):
            raise ValueError(f'{profile}: speculative acceptance missing for {key}')
    expected_quality = (sum(protocol['concurrency']) * len(protocol['lengths'])
                        * len(protocol['efforts']) * len(protocol['cache_states']))
    for dimension, passing in (('strict_exact_match', 'PASS'),
                               ('output_format_compliance', 'PASS'),
                               ('generation_completed', 'PASS'),
                               ('backend_runtime_error', 'NO')):
        values = report['niah'].get(dimension, {})
        if values.get(passing) != expected_quality or sum(values.values()) != expected_quality:
            raise ValueError(f'{profile}: {dimension} did not pass all {expected_quality} requests')
    return directory, protocol, report


def compare(baseline_dir, candidate_dir, image_diff_reason=None):
    bd, bp, br = read_leg(baseline_dir, 'dflash-k4')
    cd, cp, cr = read_leg(candidate_dir, 'mtp3')
    mismatches = [field for field in MATCHED_FIELDS if bp.get(field) != cp.get(field)]
    if mismatches:
        raise ValueError('Comparison protocol/source differs: ' + ', '.join(mismatches))
    if (model_path(bp['model_identity']) != model_path(cp['model_identity'])
            or model_mount(bp['model_identity']) != model_mount(cp['model_identity'])):
        raise ValueError('Checkpoint mount/path differs')
    image_diff = (bp['model_identity']['image'] != cp['model_identity']['image']
                  or bp['cache_identity']['image'] != cp['cache_identity']['image'])
    if image_diff and not image_diff_reason:
        raise ValueError('Image differs; an explicit diagnostic image-diff reason is required')
    if image_diff_reason and not image_diff:
        raise ValueError('Image-diff reason supplied, but images match')
    if hardware_signature(bp) != hardware_signature(cp) or power_limits(bd) != power_limits(cd):
        raise ValueError('PCIe topology/capability or GPU power limits differ')
    rows = {}
    for key in sorted(br['groups']):
        old, new = br['groups'][key], cr['groups'][key]
        def value(group, field):
            return group['per_request'][field]['median']
        base_decode, mtp_decode = value(old, 'decode_tps'), value(new, 'decode_tps')
        base_ttft, mtp_ttft = value(old, 'ttft_ms'), value(new, 'ttft_ms')
        qualified = (old['decode_concurrency_qualified']
                     and new['decode_concurrency_qualified'])
        rows[key] = dict(
            k4_decode_tps=base_decode, mtp3_decode_tps=mtp_decode,
            decode_change_percent=100 * (mtp_decode / base_decode - 1) if qualified else None,
            k4_ttft_ms=base_ttft, mtp3_ttft_ms=mtp_ttft,
            ttft_reduction_percent=100 * (1 - mtp_ttft / base_ttft),
            k4_effective_prompt_tps=value(old, 'effective_prompt_tps'),
            mtp3_effective_prompt_tps=value(new, 'effective_prompt_tps'),
            k4_aggregate_e2e_output_tps=old['aggregate_e2e_output_tps'],
            mtp3_aggregate_e2e_output_tps=new['aggregate_e2e_output_tps'],
            k4_acceptance_median=statistics.median(a['ratio'] for a in old['acceptance']),
            mtp3_acceptance_median=statistics.median(a['ratio'] for a in new['acceptance']),
            k4_overlap_s=old['overlap_s'], mtp3_overlap_s=new['overlap_s'],
            concurrent_decode_qualified=qualified,
        )
    return dict(status=('matched_protocol_image_variant_diagnostic' if image_diff
                        else 'matched_core_diagnostic'),
                baseline=str(bd), candidate=str(cd),
                protocol_template='glm-mtp3-k4.json', results=rows,
                image_variation=(dict(reason=image_diff_reason,
                    k4_model=bp['model_identity']['image'], k4_cache=bp['cache_identity']['image'],
                    mtp3_model=cp['model_identity']['image'], mtp3_cache=cp['cache_identity']['image'])
                    if image_diff else None),
                end_to_end_qualified=False,
                missing_for_promotion=['matched completed multi-turn coding/tool task',
                                       'review of individual outputs and runtime stability'],
                note=('TTFT includes prefill/cache restore; fixed-output decode is not full task latency. '
                      'With different images, observed differences apply to the deployed configurations '
                      'and cannot be attributed to the drafter alone.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--image-diff-reason', help='Explicitly permit a diagnostic comparison of different images')
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate, args.image_diff_reason)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(args.out)


if __name__ == '__main__':
    main()
