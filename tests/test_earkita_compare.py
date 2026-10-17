"""Offline checks of the focused K4/MTP3 comparison gate; no GPU or server."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'earkita_compare', HERE/'bench/baseline-earkita/compare.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def write(directory, name, value):
    path = directory/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')
    return path


def leg(directory, profile):
    directory.mkdir()
    corpus = write(directory, 'corpus.json', {'corpus': [{'id': 'same'}]})
    source_hash = module.digest(corpus)
    identity = dict(image='same-image', args=['--model', '/models/GLM'],
                    mounts=[dict(source='/same/checkpoint', destination='/models')])
    protocol = dict(template_file='glm-mtp3-k4.json', speculative_profile=profile,
                    quality_gate='strict_exact_completed', lengths=[130816],
                    concurrency=[1, 4], efforts=['high'], cache_states=['cold', 'ram', 'disk'],
                    measured_batches=3, frozen_corpus_sha256=source_hash,
                    corpora={'high': dict(file='corpus.json', sha256=source_hash,
                                         source_sha256=source_hash)},
                    model_identity=identity, cache_identity=identity,
                    idle_links=dict(paths={'0000:e3:00.0': ['0000:e3:00.0']},
                                    devices={'0000:e3:00.0': dict(max_link_speed='32 GT/s',
                                        max_link_width='16', numa_node='0')}),
                    sources={'same.py': 'hash'}, versions={'vllm': 'same'},
                    mapping=[dict(hip=0, worker_rank=0, bdf='0000:e3:00.0')])
    write(directory, 'protocol.json', protocol)
    (directory/'protocol.sha256').write_text(module.digest(directory/'protocol.json'))
    write(directory, 'status.json', {'status': 'complete'})
    write(directory, 'hardware-before/power-limits.json', {'gpu_data': [
        dict(gpu=0, limit={'ppt0': {'socket_power_limit': {'value': 210}}})]})
    groups = {}
    for count in (1, 4):
        for state in ('cold', 'ram', 'disk'):
            key = f'130816/C{count}/high/{state}'
            groups[key] = dict(batches=3, per_request={metric: dict(
                median=50. if metric == 'decode_tps' else 500., n=count*3)
                for metric in ('decode_tps', 'ttft_ms', 'effective_prompt_tps')},
                aggregate_e2e_output_tps=[10., 11., 9.],
                decode_concurrency_qualified=not (count == 4 and state == 'cold'),
                acceptance=[dict(ratio=.5, drafts=100)]*3,
                overlap_s=[0., 0., 0.] if count == 4 and state == 'cold' else [1., 1., 1.])
    niah = {name: {value: 15} for name, value in (
        ('strict_exact_match', 'PASS'), ('output_format_compliance', 'PASS'),
        ('generation_completed', 'PASS'), ('backend_runtime_error', 'NO'))}
    write(directory, 'report.json', dict(status='complete', groups=groups, niah=niah))
    return protocol


def rewrite_protocol(directory, protocol):
    write(directory, 'protocol.json', protocol)
    (directory/'protocol.sha256').write_text(module.digest(directory/'protocol.json'))


class ComparisonGate(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.k4, self.mtp = root/'k4', root/'mtp3'
        self.k4_protocol = leg(self.k4, 'dflash-k4')
        self.mtp_protocol = leg(self.mtp, 'mtp3')

    def test_matched_diagnostic_keeps_cold_c4_decode_unqualified(self):
        result = module.compare(self.k4, self.mtp)
        self.assertEqual(result['status'], 'matched_core_diagnostic')
        self.assertFalse(result['end_to_end_qualified'])
        self.assertIsNone(result['results']['130816/C4/high/cold']['decode_change_percent'])
        self.assertEqual(result['results']['130816/C1/high/ram']['decode_change_percent'], 0)
        self.assertEqual(result['results']['130816/C4/high/ram']['k4_aggregate_e2e_output_tps'],
                         [10., 11., 9.])

    def test_different_images_require_explicit_diagnostic_scope(self):
        self.mtp_protocol['model_identity'] = dict(
            self.mtp_protocol['model_identity'], image='mtp-patched-image')
        self.mtp_protocol['cache_identity'] = dict(
            self.mtp_protocol['cache_identity'], image='mtp-patched-image')
        rewrite_protocol(self.mtp, self.mtp_protocol)
        with self.assertRaisesRegex(ValueError, 'Image differs'):
            module.compare(self.k4, self.mtp)
        result = module.compare(self.k4, self.mtp, 'LMCache relocation fix for MTP3')
        self.assertEqual(result['status'], 'matched_protocol_image_variant_diagnostic')
        self.assertFalse(result['end_to_end_qualified'])
        self.assertEqual(result['image_variation']['reason'], 'LMCache relocation fix for MTP3')

    def test_image_variation_does_not_allow_checkpoint_change(self):
        self.mtp_protocol['model_identity'] = dict(
            self.mtp_protocol['model_identity'], image='mtp-patched-image',
            mounts=[dict(source='/different/checkpoint', destination='/models')])
        self.mtp_protocol['cache_identity'] = dict(
            self.mtp_protocol['cache_identity'], image='mtp-patched-image')
        rewrite_protocol(self.mtp, self.mtp_protocol)
        with self.assertRaisesRegex(ValueError, 'Checkpoint mount/path differs'):
            module.compare(self.k4, self.mtp, 'LMCache relocation fix for MTP3')

    def test_incomplete_or_wrong_quality_cannot_compare(self):
        write(self.mtp, 'status.json', {'status': 'failed'})
        with self.assertRaisesRegex(ValueError, 'did not complete'):
            module.compare(self.k4, self.mtp)
        write(self.mtp, 'status.json', {'status': 'complete'})
        report = json.loads((self.mtp/'report.json').read_text())
        report['niah']['output_format_compliance'] = {'PASS': 14, 'FAIL': 1}
        write(self.mtp, 'report.json', report)
        with self.assertRaisesRegex(ValueError, 'output_format_compliance'):
            module.compare(self.k4, self.mtp)

    def test_changed_source_or_corpus_cannot_compare(self):
        self.mtp_protocol['sources'] = {'same.py': 'changed'}
        rewrite_protocol(self.mtp, self.mtp_protocol)
        with self.assertRaisesRegex(ValueError, 'sources'):
            module.compare(self.k4, self.mtp)
        self.mtp_protocol['sources'] = self.k4_protocol['sources']
        rewrite_protocol(self.mtp, self.mtp_protocol)
        (self.mtp/'corpus.json').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'corpus changed'):
            module.compare(self.k4, self.mtp)

    def test_missing_overlap_or_invalid_acceptance_cannot_compare(self):
        report = json.loads((self.mtp/'report.json').read_text())
        report['groups']['130816/C4/high/ram']['decode_concurrency_qualified'] = False
        write(self.mtp, 'report.json', report)
        with self.assertRaisesRegex(ValueError, 'simultaneous warm decode'):
            module.compare(self.k4, self.mtp)
        report['groups']['130816/C4/high/ram']['decode_concurrency_qualified'] = True
        report['groups']['130816/C1/high/ram']['acceptance'][0]['ratio'] = None
        write(self.mtp, 'report.json', report)
        with self.assertRaisesRegex(ValueError, 'acceptance missing'):
            module.compare(self.k4, self.mtp)


if __name__ == '__main__':
    unittest.main()
