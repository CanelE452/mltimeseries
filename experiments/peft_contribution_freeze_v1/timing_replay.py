"""Remeasure pre-cleanup fits with the same frozen learning procedure."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
from experiments.peft_optimization_control_v1.run import sha,save
from experiments.peft_contribution_freeze_v1.run import verify,ROOT,RUN
from experiments.hospital_shared_strength_v1.resume_monitored import admission
from experiments.peft_adaptation_scope_v1.guard import run_guarded


def prepare():
    assert not (RUN/'selection.json').exists() and not (RUN/'f0').exists()
    assert not (RUN/'timing_contract.json').exists()
    plan=json.loads((RUN/'plan.json').read_text());verify(plan)
    partial=json.loads((RUN/'partial_validation_audit.json').read_text())
    keys={(r['dataset'],r['condition'],r['arm'],r['seed']) for r in partial['rows']}
    jobs=[e for e in plan['jobs'] if tuple(e['job'][k] for k in ('dataset','condition','arm','seed')) in keys]
    assert len(jobs)==13
    sources=[Path(__file__),Path(__file__).with_name('analyse_resource_matched.py')]
    preserved=[RUN/'plan.json',RUN/'contract.json',RUN/'partial_validation_audit.json',
               RUN/'resume_20260911_0505/cleanup_result.json']
    for e in jobs:
        preserved.extend([RUN/e['key']/'output/result.json',RUN/e['key']/'output/best.pt',RUN/e['key']/'guard/status.json'])
    contract={'created_utc':datetime.now(timezone.utc).isoformat(),'E_started_at_creation':False,
        'reason':'Resource cleanup changed throughput: identical Jena FULL90 seed27000 180-step checkpoint took109.641s (ES2 pre-cleanup) versus40.344s (FULL post-cleanup).',
        'selection':'All13 fits completed before cleanup, chosen only from the pre-existing partial_validation_audit snapshot; no E scores used.',
        'jobs':jobs,'source_hashes':{p.relative_to(ROOT).as_posix():sha(p) for p in sources},
        'input_hashes':{p.relative_to(ROOT).as_posix():sha(p) for p in preserved},
        'quality_policy':'Original E checkpoints/forecasts unchanged. Require exact best.pt hash and best step/score match for each replay.',
        'cost_policy':'Replace only13 pre-cleanup fit guard times with their actual same-procedure replay times. Keep original35 post-cleanup fit times. Do not substitute minimum or faster-of-two times. Report original mixed-session results separately.',
        'gates':plan['gates'],'limitations':'One time observation per fit in resumed environment; not simultaneous measurements or proof that all machine state is matched. Original13 fits are repeated research cost and reported as such.'}
    save(RUN/'timing_contract.json',contract)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:prepare();return
    contract=json.loads((RUN/'timing_contract.json').read_text())
    for p,d in {**contract['source_hashes'],**contract['input_hashes']}.items():assert sha(ROOT/p)==d,p
    assert json.loads((RUN/'completed.json').read_text())['completed']
    assert not (RUN/'timing_completed.json').exists()
    started=time.perf_counter();rows=[]
    for e in contract['jobs']:
        parent=RUN/e['key'].replace('fits/','timing_replay/')
        if not parent.exists():
            admission(RUN/'resume_20260911_0505',str(parent.relative_to(RUN)))
            command=[sys.executable,'-m','experiments.peft_contribution_freeze_v1.fit','--contract',str(RUN/'contract.json'),
                     '--job',json.dumps(e['job']),'--output',str(parent/'output')]
            print(json.dumps({'timing_replay':e['key']}),flush=True)
            status=run_guarded(command,parent/'guard',ROOT,900,require_gpu=True)
            assert status['completed'] and not status['reasons'],f'Preserved failure: {parent}'
        status=json.loads((parent/'guard/status.json').read_text())
        result=json.loads((parent/'output/result.json').read_text())
        old=json.loads((RUN/e['key']/'output/result.json').read_text())
        assert status['completed'] and not status['reasons'] and result['completed']
        assert result['job']==old['job']==e['job']
        for key in ('contract_sha256','best_step','best_score','checkpoint_sha256','steps','frozen_step','sample_index_sha256'):
            assert result[key]==old[key],(e['key'],key)
        assert sha(parent/'output/best.pt')==old['checkpoint_sha256']
        rows.append({'key':e['key'],'replay_key':parent.relative_to(RUN).as_posix(),'fit_seconds':status['elapsed_seconds'],'checkpoint_exact_match':True})
    save(RUN/'timing_completed.json',{'completed':True,'finished_utc':datetime.now(timezone.utc).isoformat(),
         'seconds':time.perf_counter()-started,'contract_sha256':sha(RUN/'timing_contract.json'),'rows':rows})


if __name__=='__main__':main()
