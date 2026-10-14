import json
import pytest
from r9700_vllm.moe.deepseek_placement import (
    page_weights, build_plan, read_profiles, decode_weight, placement_hotness,
)


def test_migration_balances_both_memory_pools():
    from r9700_vllm.moe.deepseek_placement import migration_order
    # A GPU-first order would temporarily duplicate the whole host budget.
    jobs = [(delta, i, 'weight', None) for i, delta in enumerate(
        [-540]*20 + [-270]*20 + [540]*20 + [270]*20)]
    ordered = migration_order(jobs)
    balance = 0
    peak = 0
    for job in ordered:
        balance += job[0]
        peak = max(peak, abs(balance))
    assert balance == 0 and peak <= 540
    assert {job[1] for job in ordered} == set(range(80))


def test_reload_metadata_does_not_pin_old_host_storage():
    import gc
    import weakref
    torch = pytest.importorskip('torch')
    pytest.importorskip('vllm')
    from vllm.model_executor.model_loader.reload.layerwise import record_metadata_for_reloading, LAYERWISE_INFO
    from r9700_vllm.moe.deepseek_placement import release_reload_host_aliases, install_reload_guard
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.ones(4), requires_grad=False)
    host = torch.ones(4)
    model.weight._r9700_host_view = host
    model.weight._r9700_host_bytes = host.nbytes
    model.weight._vllm_is_uva_offloaded = True
    observed = weakref.ref(host)
    record_metadata_for_reloading(model)
    del model.weight._r9700_host_view, host
    gc.collect()
    assert observed() is not None  # The upstream meta snapshot retains it.
    assert release_reload_host_aliases(model) == 1
    gc.collect()
    assert observed() is None
    assert model.weight.shape == (4,)
    assert not hasattr(LAYERWISE_INFO[model].restore_metadata[0]['weight'], '_vllm_is_uva_offloaded')
    install_reload_guard()
    from vllm.model_executor.model_loader.reload import initialize_layerwise_reload
    with pytest.raises(RuntimeError, match='server restart'):
        initialize_layerwise_reload(model)


def test_straddling_pages_use_bytes_and_partial_tail():
    assert page_weights([1., 4.], 12, 4) == [1., 2.5, 4.]
    assert page_weights([1., 4.], 12, 5) == [1., 3.4, 4.]


def test_ep_mapping_and_budget():
    mapping=[-1]*384
    mapping[96:144]=list(range(48))
    hot={i:[100.]*384 for i in range(40)}
    hot[39][96]=0.
    plan=build_plan(hot,mapping,{'w13_weight':48*8,'w2_weight':48*4},4,13)
    assert sum(map(len,plan.values())) == 3
    assert plan[39,'w13_weight']=={0,1}
    assert plan[39,'w2_weight']=={0}
    with pytest.raises(ValueError,match='48 distinct'):
        build_plan(hot,[-1]*384,{'w13_weight':384,'w2_weight':192},4,13)


def test_profile_identity_and_corpus_worst_case(tmp_path):
    identity={'checkpoint':'abc','layout':'quark-ep8','regime':'eager',
              'stage_min_tokens':256,'reserved_staging':True}
    rows=[{'layer':i,'decode':{'calls':2,'slots':[2]*384,'calls_hit':[1]*384}} for i in range(40)]
    p=tmp_path/'a.json';p.write_text(json.dumps({'schema':1,'identity':identity,'rows':rows,'all_requests_completed':True}))
    assert read_profiles([p],identity)[0][0]==1.
    # Staging clones immutable packed bytes; its threshold is not a route or
    # layout property. Keep the calibration usable for the staging A/B.
    assert read_profiles([p],{**identity,'stage_min_tokens':0,'reserved_staging':False})[0][0]==1.
    with pytest.raises(ValueError,match='identity'):
        read_profiles([p],{**identity,'checkpoint':'different'})
    with pytest.raises(ValueError,match='identity'):
        read_profiles([p],{**identity,'layout':'different'})
    rows[0]['decode']['calls_hit'][0]=2
    q=tmp_path/'b.json';q.write_text(json.dumps({'schema':1,'identity':identity,'rows':rows,'all_requests_completed':True}))
    hot=read_profiles([p,q],identity)
    assert hot[0][0]>1.99 and hot[0][1]==1.
    rows.pop();q.write_text(json.dumps({'schema':1,'identity':identity,'rows':rows,'all_requests_completed':True}))
    with pytest.raises(ValueError,match='every backbone'):
        read_profiles([q],identity)


def test_long_prefill_slots_preserve_hotness_when_chunk_hits_saturate(tmp_path):
    identity={'checkpoint':'abc'}
    rows=[{'layer':i,
           'prefill':{'calls':2,'slots':[100,1]+[1]*382,'calls_hit':[2]*384},
           'decode':{'calls':2,'slots':[0,6]+[0]*382,'calls_hit':[0,2]+[0]*382}}
          for i in range(40)]
    path=tmp_path/'routes.json'
    path.write_text(json.dumps({'schema':1,'identity':identity,'rows':rows,
                                'all_requests_completed':True}))
    prefill=read_profiles([path],identity,phase='prefill',counter='slots')
    hits=read_profiles([path],identity,phase='prefill',counter='calls_hit')
    assert prefill[0][0]>50*prefill[0][1]
    assert hits[0][0]==hits[0][1]


def test_decode_aware_placement_blends_measured_phases(tmp_path, monkeypatch):
    identity={'checkpoint':'abc'}
    rows=[{'layer':i,
           'prefill':{'calls':2,'slots':[100,1]+[1]*382,'calls_hit':[2]*384},
           'decode':{'calls':2,'slots':[0,6]+[0]*382,'calls_hit':[0,2]+[0]*382}}
          for i in range(40)]
    path=tmp_path/'routes.json'
    path.write_text(json.dumps({'schema':1,'identity':identity,'rows':rows,
                                'all_requests_completed':True}))
    prefill=read_profiles([path],identity,phase='prefill',counter='slots')
    decode=read_profiles([path],identity,phase='decode',counter='calls_hit')
    assert placement_hotness([path],identity,0)==prefill
    blend=placement_hotness([path],identity,.5)
    assert blend[0][0]==pytest.approx((prefill[0][0]+decode[0][0])/2)
    assert blend[0][1]==pytest.approx((prefill[0][1]+decode[0][1])/2)
    monkeypatch.setenv('R9K_DEEPSEEK_PLACEMENT_DECODE_WEIGHT','.5')
    assert decode_weight()==.5
    for invalid in ('nan','inf','-0.1','1.1','garbage'):
        monkeypatch.setenv('R9K_DEEPSEEK_PLACEMENT_DECODE_WEIGHT',invalid)
        with pytest.raises(ValueError,match='decode weight'):
            decode_weight()


@pytest.mark.parametrize('fail_at',[None,2])
def test_runtime_placement_completes_or_refuses_partial(monkeypatch, fail_at):
    torch=pytest.importorskip('torch');pytest.importorskip('vllm')
    from types import SimpleNamespace as NS
    import vllm.model_executor.offloader as offload
    import vllm.model_executor.layers.fused_moe.routed_experts as routed
    from r9700_vllm.moe import deepseek_placement as placement, deepseek_residency as residency, deepseek_expert_profile as profile
    class Fake(torch.nn.Module):
        def __init__(self,index):
            super().__init__()
            self.layer_name=f'model.layers.{index}.ffn.experts'
            self.global_num_experts=384
            self.expert_map=torch.tensor(list(range(48))+[-1]*336)
            for name,shape in [('w13_weight',(48,4608,2560)),('w2_weight',(48,5120,1152))]:
                p=torch.nn.Parameter(torch.empty(shape,device='meta',dtype=torch.uint8),requires_grad=False)
                p._r9700_host_view=torch.empty(1)
                if index < 14:
                    p._vllm_is_uva_offloaded = True
                    p._r9700_host_bytes = p.nbytes
                setattr(self,name,p)
            self.w13_weight_scale=self.w2_weight_scale=torch.empty(1)
    model=torch.nn.ModuleList([Fake(i) for i in range(40)])
    monkeypatch.setattr(routed,'RoutedExperts',Fake)
    actual_host = 14 * 48 * (4608*2560 + 5120*1152)
    monkeypatch.setattr(offload,'get_offloader',lambda:NS(cpu_offload_bytes=actual_host,cpu_offload_max_bytes=11*2**30))
    monkeypatch.setattr(profile,'identity',lambda config:{})
    def measured_profiles(*args,**kwargs):
        assert kwargs=={'phase':'prefill','counter':'slots'}
        return {i:[1.]*384 for i in range(40)}
    monkeypatch.setattr(placement,'read_profiles',measured_profiles)
    monkeypatch.setattr(residency,'extension',lambda:NS(page_size=lambda *args:2**21))
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *args:None)
    calls=[]
    def migrate(source,flags):
        if len(calls)==fail_at:raise RuntimeError('injected VMM failure')
        calls.append(flags)
        return source.clone()
    monkeypatch.setattr(residency,'make_partially_resident',migrate)
    config=NS(parallel_config=NS(enable_expert_parallel=True,tensor_parallel_size=8))
    if fail_at is not None:
        with pytest.raises(RuntimeError,match='PARTIAL.*2/80'):
            placement.place_model(model,config,['test'])
    else:
        placement.place_model(model,config,['test'])
        assert len(calls)==80
        assert sum(int((~flags).sum())*2**21 for flags in calls)==actual_host
        assert all(not hasattr(p,'_r9700_host_view') for p in model.parameters())
