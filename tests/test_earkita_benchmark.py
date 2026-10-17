"""Offline cache-state and measurement gates. No Docker/GPU/server required."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'bench/baseline-earkita'))
spec = importlib.util.spec_from_file_location('earkita_runner',ROOT/'bench/baseline-earkita/run.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


class FocusedComparisonProtocol(unittest.TestCase):
    def test_warm_only_protocol_reuses_identical_salts_across_legs(self):
        focused, _ = b.template('glm-paged-logits-128k.json')
        self.assertEqual(focused['cache_states'], ['ram'])
        self.assertTrue(focused['warm_only_prime_once'])
        a = dict(focused, run_id='first')
        c = dict(focused, run_id='second')
        initial = b.comparison_salts(a, 130816, 'high', 4, 0, 4)
        self.assertEqual(initial, b.comparison_salts(a, 130816, 'high', 4, 4, 4))
        self.assertEqual(initial, b.comparison_salts(c, 130816, 'high', 4, 4, 4))
        self.assertEqual(4, len(set(initial)))
        self.assertNotEqual(initial, b.comparison_salts(a, 130816, 'high', 1, 0, 4))

    def test_warm_only_cold_prime_checks_cache_and_keeps_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            cases = [dict(id='niah_130816_depth005', messages=[])]
            config = dict(chunk_tokens=1152, speculative_profile='dflash-k4',
                          output_tokens=256, idle_links={}, pcie_expected={})
            row = dict(result=dict(ok=True, prompt_tokens=130816, completion_tokens=1))
            with ExitStack() as stack:
                for obj, name, value in [
                    (b, 'idle', lambda _:None),
                    (b, 'reset_gpu', lambda _:None),
                    (b, 'clear_l1', lambda _:None),
                    (b, 'snapshot', lambda *args:{}),
                    (b, 'batch', lambda p,*args: ([row],[dict(links={})],[]) if p['output_tokens']==1 else None),
                    (b, 'cache_proof', lambda *args:{'cold_store_proven':True}),
                    (b.hardware, 'check_loaded', lambda *args:None),
                ]:
                    stack.enter_context(patch.object(obj,name,side_effect=value))
                b.prime_warm_only(config,out,cases,['fixed-salt'],'high',130816,1)
            attempt = json.loads((out/'prime-130816-C1-high/result.json').read_text())
            self.assertTrue(attempt['valid'])
            self.assertEqual(attempt['cache_proof'], {'cold_store_proven':True})
            self.assertEqual(json.loads((out/'status.json').read_text())['phase'], 'prime')

    def test_existing_standard_stays_k4_and_focused_protocol_matches_frozen_corpus(self):
        full, name = b.template('glm.json')
        self.assertEqual(name, 'glm.json')
        self.assertEqual(full['concurrency'], [1, 2, 4])
        focused, name = b.template('glm-mtp3-k4.json')
        self.assertEqual(name, 'glm-mtp3-k4.json')
        self.assertEqual(focused['lengths'], [130816])
        self.assertEqual(focused['concurrency'], [1, 4])
        self.assertEqual(focused['warmup_batches'], 1)
        self.assertEqual(focused['measured_batches'], 3)
        self.assertEqual(focused['quality_gate'], 'strict_exact_completed')
        self.assertEqual(focused['required_max_model_len'], 262144)
        self.assertEqual(focused['required_kv_cache_bytes'], int(4.125 * 2**30))
        corpus = b.ROOT/'bench/results/glm-mtp3-k4-20261017-prep/corpus/corpus.json'
        if corpus.exists():
            self.assertEqual(b.digest(corpus), focused['frozen_corpus_sha256'])
        with self.assertRaisesRegex(RuntimeError, 'Protocol template'):
            b.template('../other.json')

    def test_speculative_profiles_are_explicit_and_draft_geometry_is_checked(self):
        k4 = ['--speculative-config', json.dumps(dict(method='dflash',num_speculative_tokens=4,
                    kv_cache_dtype='fp8_e4m3',draft_tensor_parallel_size=8))]
        mtp3 = ['--speculative-config', json.dumps(dict(method='mtp',num_speculative_tokens=3))]
        self.assertEqual(b.check_speculative_profile(k4,'dflash-k4')['method'],'dflash')
        self.assertEqual(b.check_speculative_profile(mtp3,'mtp3')['method'],'mtp')
        with self.assertRaises(RuntimeError):
            b.check_speculative_profile(k4,'mtp3')
        with self.assertRaises(RuntimeError):
            b.check_speculative_profile(mtp3,'dflash-k4')
        with self.assertRaises(RuntimeError):
            b.check_speculative_profile(['--speculative-config',json.dumps(dict(
                method='mtp',num_speculative_tokens=4))],'mtp3')

    def test_mtp3_prepare_freezes_exact_corpus_without_regenerating_it(self):
        results_root = b.ROOT/'bench/results'
        results_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=results_root) as tmp:
            root = Path(tmp)
            source = root/'source.json'
            cases = [dict(id=f'niah_130816_depth{depth:03}',messages=[dict(role='user',content=str(depth))])
                     for depth in (5,35,65,95)]
            source.write_text(json.dumps(cases))
            protocol, _ = b.template('glm-mtp3-k4.json')
            protocol['frozen_corpus_sha256'] = b.digest(source)
            model_args = ['--host','127.0.0.1','--tensor-parallel-size','8','--max-num-seqs','4',
                          '--kv-cache-dtype','fp8_e4m3','--speculative-config',json.dumps(dict(
                              method='mtp',num_speculative_tokens=3)),
                          '--max-model-len','262144','--kv-cache-memory',str(int(4.125*2**30)),
                          '--port','8081','--model','/models/GLM-5.3-Flash-Quark-MXFP4',
                          '--kv-transfer-config','{}']
            cache_args = ['--port','5556','--prometheus-port','9092','--l1-size-gb','128',
                          '--chunk-size','1152','--l2-adapter',json.dumps(dict(
                              type='fs_native',base_path='/kv-cache/mtp3',max_capacity_gb=512,
                              use_odirect=False))]
            model = dict(running=True,image='same',args=model_args,
                         env={'VLLM_SERVER_DEV_MODE':'1','R9K_GLM_DFLASH':'0'},mounts=[])
            cache = dict(running=True,image='same',args=cache_args,env={},mounts=[
                dict(source=str(root/'cache'),destination='/kv-cache')])
            a = SimpleNamespace(out=root/'attempt',model_container='mtp-model',
                cache_container='mtp-cache',base='http://127.0.0.1:8081',
                cache_metrics='http://127.0.0.1:9092',cache_port=5556,
                pcie_policy=None,protocol_template='glm-mtp3-k4.json',
                speculative_profile='mtp3',frozen_corpus=source)
            def command(*args):
                return 'head' if args[0]=='git' else json.dumps(dict(
                    vllm='0.3.1.dev311+ge97573215',lmcache='0.5.5',torch='test',triton='test'))
            with ExitStack() as stack:
                for obj,name,value in [
                    (b,'template',lambda _: (protocol.copy(),'glm-mtp3-k4.json')),
                    (b,'sources',lambda:{}),
                    (b,'container_identity',lambda name: model if name=='mtp-model' else cache),
                    (b,'host_memory',lambda:{}),
                    (b,'snapshot',lambda *args:{}),
                    (b.hardware,'command',command),
                    (b.hardware,'hip_mapping',lambda _:[]),
                    (b.hardware,'capture',lambda *args:{}),
                    (b.hardware,'expected_links',lambda _:{}),
                ]:
                    stack.enter_context(patch.object(obj,name,side_effect=value))
                b.prepare(a)
            frozen = json.loads((a.out/'protocol.json').read_text())
            self.assertEqual(frozen['speculative_profile'],'mtp3')
            self.assertEqual(frozen['frozen_corpus']['sha256'],b.digest(source))
            copied = json.loads((a.out/'corpus-high.json').read_text())
            self.assertEqual(copied['corpus'],cases)
            self.assertEqual(frozen['corpora']['high']['sha256'],b.digest(a.out/'corpus-high.json'))


class CacheGates(unittest.TestCase):
    def setUp(self):
        self.before = dict.fromkeys(b.COUNTERS,0.) | dict(l1_bytes=0.,running=0.,waiting=0.)

    def after(self, **kwargs):
        return self.before | dict(success=4,queries=4*8192,gpu_stores_done=4,stores_done=4,stored_objects=28) | kwargs

    def test_all_three_proven_sources(self):
        expected = 7*1152*4
        variants = [('cold',self.after()),
                    ('ram',self.after(external_hit=expected,l1_hit=expected)),
                    ('disk',self.after(external_hit=expected,l2_hit=expected,l2_load=28))]
        for state, after in variants:
            with self.subTest(state=state):
                self.assertEqual(b.cache_proof(state,self.before,after,8192,4,1152)['success'],4)

    def test_mtp3_restore_excludes_the_unwritten_relocated_mamba_chunk(self):
        # 130816 tokens contain 113 full chunks. The observed patched MTP3
        # restore returns 112: LMCache must not persist the skipped Mamba state.
        expected = 112 * 1152 * 4
        warm = self.after(external_hit=expected,l1_hit=expected)
        self.assertEqual(b.cache_proof('ram',self.before,warm,130816,4,1152,'mtp3')['l1_hit'],
                         expected)
        with self.assertRaisesRegex(RuntimeError,'Cache source not proven'):
            b.cache_proof('ram',self.before,warm,130816,4,1152,'dflash-k4')

    def test_wrong_tier_partial_hits_gpu_hits_preemption_and_foreign_work_fail(self):
        expected = 7*1152*4
        bad = [dict(l1_hit=expected),dict(external_hit=expected,l2_hit=expected-1152,l2_load=1),
               dict(external_hit=expected,l2_hit=expected,l2_load=28,gpu_hit=1152),
               dict(external_hit=expected,l2_hit=expected,l2_load=28,preempt=1),
               dict(external_hit=expected,l2_hit=expected,l2_load=28,success=5)]
        for changes in bad:
            with self.subTest(changes=changes),self.assertRaises(RuntimeError):
                b.cache_proof('disk',self.before,self.after(**changes),8192,4,1152)

    def test_l1_must_be_empty_for_disk(self):
        before = self.before | {'l1_bytes':4096}
        with self.assertRaises(RuntimeError):
            b.cache_proof('disk',before,self.after(external_hit=32256,l2_hit=32256,l2_load=28),8192,4,1152)

    def test_counter_reset_fails(self):
        with self.assertRaisesRegex(RuntimeError,'Counters reset'):
            b.cache_proof('cold',self.before|{'accepted':5},self.after(),8192,4,1152)

    def test_missing_metric_is_not_zero_but_unobserved_registered_counter_is(self):
        self.assertEqual(b.metric('# TYPE count counter\n','count'),0)
        self.assertEqual(b.metric('count{x="a"} 3\ncount{x="b"} 5\ncount_created 900\n','count'),8)
        with self.assertRaises(RuntimeError):
            b.metric('other 0\n','count')
        self.assertEqual(b.metric('other 0\n','count',unobserved_zero=True),0)

    def test_cold_requires_store_completion_evidence(self):
        with self.assertRaisesRegex(RuntimeError,'population not proven'):
            b.cache_proof('cold',self.before,self.after(gpu_stores_done=0),8192,4,1152)
        with self.assertRaisesRegex(RuntimeError,'population not proven'):
            b.cache_proof('cold',self.before,self.after(stored_objects=0),8192,4,1152)

    def test_pinned_lazy_counters_but_not_missing_gauges(self):
        vr = '\n'.join(f'# TYPE {name} counter\n{name} 0' for name in b.COUNTERS.values() if name.startswith(b.V))
        vr += '\nvllm:num_requests_running 0\nvllm:num_requests_waiting 0\n'
        lr = 'lmcache_mp_l1_memory_usage_bytes 0\n'
        p = dict(base='test-model',cache_metrics='test-cache',versions={'lmcache':'0.5.5'})
        with patch.object(b,'SEEN_COUNTERS',set()),patch.object(b,'http',side_effect=[vr,lr]):
            self.assertEqual(b.snapshot(p)['l1_hit'],0)
        with patch.object(b,'SEEN_COUNTERS',set()),patch.object(b,'http',side_effect=[vr,'']):
            with self.assertRaisesRegex(RuntimeError,'l1_memory_usage_bytes'):
                b.snapshot(p)
        with patch.object(b,'SEEN_COUNTERS',{('test-cache',b.L+'lookup_hit_l1_tokens_total')}),patch.object(b,'http',side_effect=[vr,lr]):
            with self.assertRaisesRegex(RuntimeError,'lookup_hit_l1'):
                b.snapshot(p)
        p['versions']['lmcache']='unknown'
        with patch.object(b,'http',side_effect=[vr,lr]),self.assertRaises(RuntimeError):
            b.snapshot(p)

    def test_reset_404_is_not_success(self):
        p = dict(base='http://localhost:8080',settle_timeout_seconds=0)
        with patch.object(b,'idle'),patch.object(b,'http',side_effect=HTTPError('x',404,'missing',{},None)):
            with self.assertRaises(HTTPError):
                b.reset_gpu(p)

    def test_reset_false_is_not_success(self):
        p = dict(base='http://localhost:8080',settle_timeout_seconds=0)
        with patch.object(b,'idle'),patch.object(b,'http',return_value='{"success": false}'):
            with self.assertRaisesRegex(RuntimeError,'reset did not succeed'):
                b.reset_gpu(p)

    def test_reset_does_not_clear_external_or_running_requests(self):
        with patch.object(b,'idle'),patch.object(b,'http',return_value='{"success": true}') as call:
            b.reset_gpu(dict(base='http://localhost',settle_timeout_seconds=0))
        self.assertIn('reset_running_requests=false&reset_external=false',call.call_args.args[0])

    def test_pending_stores_and_other_traffic_block_cache_clear(self):
        for row in (dict(running=1,waiting=0,stores_sent=0,stores_done=0),
                    dict(running=0,waiting=0,stores_sent=3,stores_done=2),
                    dict(running=0,waiting=0,stores_sent=0,stores_done=0,gpu_stores_sent=1,gpu_stores_done=0)):
            row = dict(gpu_stores_sent=0,gpu_stores_done=0) | row
            with patch.object(b,'snapshot',return_value=row),self.assertRaises(RuntimeError):
                b.idle(dict(settle_timeout_seconds=0))


class HardwareAndReport(unittest.TestCase):
    def test_docker_mount_order_does_not_change_identity(self):
        mounts=[{'Source':'/a','Destination':'/models'},{'Source':'/b','Destination':'/kv-cache'}]
        c=dict(Config={'Env':[],'Cmd':[]},Id='id',Image='image',State={'Running':True},Mounts=mounts)
        with patch.object(b.hardware,'command',return_value=json.dumps([c])):
            first=b.container_identity('fixture')
        c['Mounts']=list(reversed(mounts))
        with patch.object(b.hardware,'command',return_value=json.dumps([c])):
            self.assertEqual(first,b.container_identity('fixture'))

    def links(self,speed='32.0 GT/s PCIe',width='16'):
        return dict(paths={'0000:01:00.0':['0000:01:00.0']},devices={'0000:01:00.0':dict(
            max_link_speed='32.0 GT/s PCIe',max_link_width='16',numa_node='0',
            current_link_speed=speed,current_link_width=width)})

    def test_idle_power_save_allowed_but_loaded_downtraining_fails(self):
        idle = self.links('2.5 GT/s PCIe')
        expected = b.hardware.expected_links(idle)
        b.hardware.check_loaded([idle,self.links()],idle,expected)
        for samples in ([],[idle],[self.links(width='8')]):
            with self.subTest(samples=samples),self.assertRaises(RuntimeError):
                b.hardware.check_loaded(samples,idle,expected)

    def test_topology_change_fails(self):
        before = self.links()
        after = self.links()
        after['paths']['0000:01:00.0'].append('0000:00:01.0')
        with self.assertRaisesRegex(RuntimeError,'topology'):
            b.hardware.check_loaded([after],before,b.hardware.expected_links(before))

    def test_shared_x4_uplink_is_not_mistaken_for_downtraining(self):
        idle = self.links()
        idle['paths']['0000:01:00.0'] = ['0000:01:00.0', '0000:00:02.0', '0000:00:01.0']
        idle['devices']['0000:00:02.0'] = dict(
            max_link_speed='32.0 GT/s PCIe', max_link_width='16',
            current_link_speed='32.0 GT/s PCIe', current_link_width='4', numa_node='0')
        idle['devices']['0000:00:01.0'] = dict(
            max_link_speed='32.0 GT/s PCIe', max_link_width='4',
            current_link_speed='32.0 GT/s PCIe', current_link_width='4', numa_node='0')
        expected = b.hardware.expected_links(idle)
        self.assertEqual(expected['0000:00:02.0']['width'], 4)
        b.hardware.check_loaded([idle], idle, expected)
        downtrained = json.loads(json.dumps(idle))
        downtrained['devices']['0000:00:02.0']['current_link_width'] = '2'
        with self.assertRaisesRegex(RuntimeError, 'width'):
            b.hardware.check_loaded([downtrained], idle, expected)

    def test_decode_not_blended_with_ttft_or_aggregate(self):
        rows = [dict(case=str(i),start=0.,end=12.,result=dict(completion_tokens=101,
            ttft_ms=2000.,ttfa_ms=3000.,update_gaps_ms=[100.]*100,decode_tps=10.)) for i in range(2)]
        s = b.summarize(rows,8192)
        self.assertEqual(s['per_request'][0]['decode_tps'],10)
        self.assertEqual(s['per_request'][0]['effective_prompt_tps'],4096)
        self.assertAlmostEqual(s['aggregate_e2e_output_tps'],202/12)
        self.assertEqual(s['all_clients_decode_overlap_s'],10)

    def test_no_decode_overlap_for_serial_requests(self):
        rows = [dict(case=str(i),start=i*10.,end=i*10.+5.,result=dict(completion_tokens=11,
            ttft_ms=1000.,ttfa_ms=None,update_gaps_ms=[100.]*10,decode_tps=10.)) for i in range(2)]
        self.assertEqual(b.summarize(rows,8192)['all_clients_decode_overlap_s'],0)

    def test_cold_cache_diagnostic_does_not_claim_concurrent_decode(self):
        summary = dict(all_clients_decode_overlap_s=0)
        with self.assertRaisesRegex(RuntimeError,'No simultaneous decode'):
            b.qualify_decode_overlap(summary,2,'cold')
        self.assertFalse(b.qualify_decode_overlap(summary,2,'cold',True))
        for state in ('ram','disk'):
            with self.subTest(state=state), self.assertRaises(RuntimeError):
                b.qualify_decode_overlap(summary,2,state,True)
        self.assertTrue(b.qualify_decode_overlap(dict(all_clients_decode_overlap_s=1),4,'ram'))

    def test_format_only_failure_retains_retrieval_and_completion(self):
        case = dict(expected='R9700-NIAH-005-ABCDEF',scoring='exact')
        response = dict(content='The answer is R9700-NIAH-005-ABCDEF.',finish_reason='stop')
        scores = b.niah_scores(response,case)
        self.assertEqual(scores['strict_exact_match'],'FAIL')
        self.assertEqual(scores['semantic_needle_retrieval'],'PASS')
        self.assertEqual(scores['output_format_compliance'],'FAIL')
        self.assertEqual(scores['generation_completed'],'PASS')
        self.assertEqual(scores['backend_runtime_error'],'NO')

    def test_focused_quality_requires_the_completed_exact_final_answer(self):
        case = dict(expected='R9700-NIAH-005-ABCDEF',scoring='exact')
        response = dict(content='The answer is R9700-NIAH-005-ABCDEF.',finish_reason='stop')
        row = dict(niah=b.niah_scores(response,case),foreign_key=False)
        self.assertTrue(b.qualify_quality([row]))
        self.assertFalse(b.qualify_quality([row],'strict_exact_completed'))
        response['content'] = case['expected']
        row['niah'] = b.niah_scores(response,case)
        self.assertTrue(b.qualify_quality([row],'strict_exact_completed'))
        row['niah']['generation_completed'] = 'FAIL'
        self.assertFalse(b.qualify_quality([row],'strict_exact_completed'))

    def test_partial_report_excludes_warmups_and_failed_performance(self):
        s = dict(per_request=[dict(decode_tps=20,ttft_ms=300,ttfa_ms=None,effective_prompt_tps=99)],
                 aggregate_e2e_output_tps=15,all_clients_decode_overlap_s=0)
        common = dict(length=8192,concurrency=1,effort='max',cache='disk',summary=s,valid=True)
        batches = [common|dict(phase='measured'),common|dict(phase='warmup'),
                   common|dict(phase='measured',valid=False)]
        with tempfile.TemporaryDirectory() as t:
            b.report(Path(t),batches,'failed')
            report = json.loads((Path(t)/'report.json').read_text())
        group = report['groups']['8192/C1/max/disk']
        self.assertEqual(group['batches'],1)
        self.assertIsNone(group['per_request']['ttfa_ms'])
        self.assertEqual(report['status'],'failed')

    def test_report_keeps_unqualified_cold_decode_visible(self):
        summary = dict(per_request=[dict(decode_tps=20,ttft_ms=300,ttfa_ms=None,effective_prompt_tps=99)],
                       aggregate_e2e_output_tps=15,all_clients_decode_overlap_s=0)
        batch = dict(length=131072,concurrency=2,effort='low',cache='cold',summary=summary,
                     valid=True,phase='measured',decode_concurrency_qualified=False)
        with tempfile.TemporaryDirectory() as t:
            b.report(Path(t),[batch],'complete')
            result=json.loads((Path(t)/'report.json').read_text())
            self.assertFalse(result['groups']['131072/C2/low/cold']['decode_concurrency_qualified'])
            self.assertIn('| False |',(Path(t)/'report.md').read_text())


class UnattendedLifecycle(unittest.TestCase):
    def execute(self, out, fail_at=None):
        p = json.loads((b.HERE/'glm.json').read_text())
        p.update(base='http://localhost:8080',cache_metrics='http://localhost:9091',
                 cache_port=5555,model='glm',model_container='model',cache_container='cache',
                 model_identity={},cache_identity={},mapping=[],idle_links={},pcie_expected={},
                 lengths=[8192],efforts=['high'],concurrency=[1],measured_batches=1,
                 run_id='offline-test',disk_path=str(out),sources={},speculative_profile='dflash-k4')
        case = dict(id='niah_8192_depth005',messages=[{'role':'user','content':'fixture'}],
                    expected='R9700-NIAH-005-ABCDEF',scoring='exact')
        b.save(out/'corpus.json',{'corpus':[case]})
        p['corpora'] = dict(high=dict(file='corpus.json',sha256=b.digest(out/'corpus.json')))
        b.save(out/'protocol.json',p)
        (out/'protocol.sha256').write_text(b.digest(out/'protocol.json'))
        (out/'hardware-before').mkdir()
        (out/'hardware-before/power-limits.json').write_text('{}')
        counters = dict.fromkeys(b.COUNTERS,0.) | dict(running=0,waiting=0,l1_bytes=0)
        operations, salts_seen = [], []
        def clear(_):
            operations.append('l1')
            counters['l1_bytes'] = 0
        def capture(directory, _):
            directory.mkdir()
            (directory/'power-limits.json').write_text('{}')
        def batch(protocol,target,cases,salts,effort,length,quality):
            index = len(salts_seen)
            salts_seen.append(salts)
            if index == fail_at:
                raise RuntimeError('injected transport failure')
            counters['success'] += 1
            counters['queries'] += 8192
            counters['stores_sent'] += 1
            counters['stores_done'] += 1
            counters['stored_objects'] += 7
            counters['gpu_stores_sent'] += 1
            counters['gpu_stores_done'] += 1
            if index%3:
                counters['external_hit'] += 8064
                counters['l1_hit' if index%3==1 else 'l2_hit'] += 8064
                if index%3==2:
                    counters['l2_load'] += 7
            counters['l1_bytes'] = 1024
            row = dict(case=case['id'],start=0.,end=4.)
            if quality:
                response = dict(content=case['expected'],finish_reason='stop')
                row.update(response=response,niah=b.niah_scores(response,case),foreign_key=False)
            else:
                row['result'] = dict(ok=True,prompt_tokens=8192,completion_tokens=256,decode_tps=100,
                    ttft_ms=1000.,ttfa_ms=None,update_gaps_ms=[10.]*255)
            return [row],[dict(links={})],[]
        with ExitStack() as stack:
            for obj,name,value in [
                (b,'sources',lambda:{}),
                (b.importlib.metadata,'version',lambda _: '0.6.0'),
                (b.importlib.metadata,'distribution',lambda _:SimpleNamespace(read_text=lambda _:json.dumps(
                    {'vcs_info':{'commit_id':p['betterbench_commit']}}))),
                (b,'container_identity',lambda _:{}),
                (b.hardware,'hip_mapping',lambda _:[]),
                (b.hardware,'capture',capture),
                (b.hardware,'command',lambda *args:'{}'),
                (b.hardware,'check_loaded',lambda *args:None),
                (b,'http',lambda *args:json.dumps({'count':8192})),
                (b,'snapshot',lambda *args:dict(counters)),
                (b,'reset_gpu',lambda _:operations.append('gpu')),
                (b,'clear_l1',clear),
                (b,'batch',batch),
                (b.shutil,'disk_usage',lambda _:SimpleNamespace(free=10**15)),
            ]:
                stack.enter_context(patch.object(obj,name,side_effect=value))
            if fail_at is None:
                b.run(SimpleNamespace(out=out))
            else:
                with self.assertRaisesRegex(RuntimeError,'injected transport failure'):
                    b.run(SimpleNamespace(out=out))
        return operations,salts_seen

    def test_whole_matrix_cycle_uses_matching_salts_and_correct_resets(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t)
            operations,salts = self.execute(out)
            self.assertEqual(json.loads((out/'status.json').read_text())['status'],'complete')
            self.assertEqual(len(list(out.glob('batch-*'))),9)
            report = json.loads((out/'report.json').read_text())
        self.assertEqual(operations,['gpu','l1','gpu','gpu','l1']*3)
        self.assertEqual(salts[:3],[salts[0]]*3)
        self.assertNotEqual(salts[0],salts[3])
        self.assertTrue(all(g['batches']==1 for g in report['groups'].values()))
        self.assertEqual(report['niah']['strict_exact_match']['PASS'],3)

    def test_failure_preserves_partial_report_and_does_not_retry(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t)
            _,salts = self.execute(out,fail_at=4)
            self.assertEqual(len(salts),5)
            self.assertEqual(json.loads((out/'status.json').read_text())['status'],'failed')
            report = json.loads((out/'report.json').read_text())
            self.assertEqual(len(report['groups']),1)
            self.assertTrue((out/'batch-0004/result.json').exists())


if __name__=='__main__':
    unittest.main()
