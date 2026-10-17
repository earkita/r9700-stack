"""CPU-only evidence tests; no model requests or launcher access."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/"bench"))
import sglang_agent_evidence as mod


SPEC=dict(spec_verify_ct=4,spec_num_correct_drafts=5,spec_num_proposed_drafts=12,
          spec_accept_rate=5/12,spec_accept_length=2.25)


def records(cache=None):
    return [dict(line='data: '+json.dumps(dict(id='owned-request',choices=[],
        sglext=dict(cached_tokens_details=cache,spec_tokens_details=SPEC))))]


def test_absent_zero_is_unknown_and_positive_gpu_hits_have_actual_source():
    result=mod.extension_evidence(records(),dict(prompt_tokens_details=None))
    assert result['cache']['cache_source']=='unknown'
    assert result['cache']['actual_cached_tokens'] is None
    result=mod.extension_evidence(records(dict(device=131008,host=0)),dict(prompt_tokens_details=dict(cached_tokens=131008)))
    assert result['cache']['cache_source']=='device' and result['cache']['actual_cached_tokens']==131008
    assert result['speculation']==SPEC and result['request_id']=='owned-request'
    assert result['gpu_prefill_seconds'] is None


def test_explicit_zero_is_valid_and_host_hits_are_not_called_gpu_hits():
    result=mod.extension_evidence(records(),dict(prompt_tokens_details=dict(cached_tokens=0)))
    assert result['cache']['cache_source']=='none' and result['cache']['actual_cached_tokens']==0
    result=mod.extension_evidence(records(dict(device=0,host=512)),dict(prompt_tokens_details=dict(cached_tokens=512)))
    assert result['cache']['cache_source']=='host'


@pytest.mark.parametrize('case',['mismatch','storage','storage_bool','boolean','missing_source','multiple_ids','conflict','invalid_spec'])
def test_bad_or_conflicting_evidence_never_qualifies(case):
    lines=records(dict(device=8,host=0));usage=dict(prompt_tokens_details=dict(cached_tokens=8))
    if case=='mismatch':usage['prompt_tokens_details']['cached_tokens']=9
    if case=='storage':lines=records(dict(device=8,host=0,storage=16))
    if case=='storage_bool':lines=records(dict(device=8,host=0,storage=False))
    if case=='boolean':usage['prompt_tokens_details']['cached_tokens']=True
    if case=='missing_source':lines=records()
    if case in ('multiple_ids','conflict','invalid_spec'):
        event=json.loads(lines[0]['line'][5:])
        if case=='multiple_ids':event['id']='another-request';lines.append(dict(line='data: '+json.dumps(event)))
        elif case=='conflict':
            event['sglext']['cached_tokens_details']['device']=7
            lines.append(dict(line='data: '+json.dumps(event)))
        else:
            event['sglext']['spec_tokens_details']['spec_accept_rate']=float('nan')
            lines=[dict(line='data: '+json.dumps(event))]
    with pytest.raises(ValueError):mod.extension_evidence(lines,usage)


def test_wrapper_preserves_original_answer_and_decode_arithmetic(tmp_path,monkeypatch):
    body=dict(messages=[dict(role='user',content='task')],temperature=1.)
    before=copy.deepcopy(body);answer=dict(content='{"status":"complete"}',usage=dict(prompt_tokens_details=None))
    metrics=dict(ttft_seconds=100.,decode_seconds=2.,post_first_update_tokens=8,decode_tps=4.)
    def transport(base,payload,stem):
        assert payload['return_cached_tokens_details'] is True and payload['return_spec_tokens_details'] is True
        stem.with_suffix('.sse').write_text('\n'.join(json.dumps(row) for row in records()))
        return answer,metrics
    monkeypatch.setattr(mod,'base_request',transport)
    result,observed=mod.request('unused',body,tmp_path/'turn')
    assert result is answer and body==before
    assert all(observed[key]==value for key,value in metrics.items())
    assert 'backend_evidence' not in metrics and observed['backend_evidence']['cache']['cache_source']=='unknown'
