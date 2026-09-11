"""Independent Study35 audit helper.

Stdlib + NumPy only.  It re-scores persisted JSON/NPZ artifacts without importing
Study35 analyse.py or torch.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, tempfile, traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "peft_head_convergence_v1"
OUT = ROOT / "results" / "peft_head_convergence_v1"
FINAL = RUN / "independent_audit.json"
PLAN_SHA = "33959c5c95e8a2cc91e7d278150bf008cfeb5d210025f7fc52dcda18ded19406"
SCHED = [0,1,2,4,8,15,30,60,120,180,240,360,540,720]
AFIT=("HEAD","WIDE","JOINT"); BFIT=("REFIT","WARM_HEAD","WARM_JOINT")
AREP=("F0","HEAD","WIDE","JOINT","CORRECTION"); BREP=("REFIT","WARM_HEAD","WARM_JOINT")
FTOL=1e-10; ATOL=1e-8


def now(): return datetime.now(timezone.utc).isoformat()
def sha(p: Path) -> str:
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def rj(p: Path): return json.loads(Path(p).read_text(encoding='utf-8'))
def wx(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f: json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
def rp(s: str) -> Path:
    p=Path(s)
    return p if p.is_absolute() else ROOT / s.replace('\\','/')
def dt(s: str) -> datetime: return datetime.fromisoformat(s.replace('Z','+00:00'))
def jv(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,Path): return x.as_posix()
    if isinstance(x,dict): return {str(k):jv(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [jv(v) for v in x]
    return x

def close(a,b,t=FTOL): return bool(np.isfinite(a) and np.isfinite(b) and abs(float(a)-float(b))<=t)
def aclose(a,b,t=ATOL): return np.shape(a)==np.shape(b) and bool(np.allclose(a,b,rtol=0,atol=t,equal_nan=True))
def eqarr(a,b):
    a=np.asarray(a); b=np.asarray(b)
    if a.shape!=b.shape: return False
    if a.dtype.kind in 'fc' or b.dtype.kind in 'fc': return bool(np.array_equal(a,b,equal_nan=True))
    return bool(np.array_equal(a,b))

class A:
    def __init__(self, mode): self.mode=mode; self.issues=[]; self.notes=[]; self.counts={}
    def issue(self, code, msg, **data): self.issues.append({'code':code,'message':msg,'data':jv(data)})
    def ck(self, cond, code, msg, **data):
        if not cond: self.issue(code,msg,**data)
        return bool(cond)

# Independent score/correction implementation copied by behavior, not import.
def pinball_components(pred, target, q):
    pred=np.sort(np.asarray(pred,dtype=np.float64),axis=2); y=np.asarray(target,dtype=np.float64); q=np.asarray(q,dtype=np.float64)
    err=y[:,:,None,:]-pred; qq=q[None,None,:,None]; valid=np.isfinite(y)
    num=np.where(valid[:,:,None,:],2*np.maximum(qq*err,(qq-1)*err),0.0).sum(axis=3)
    cnt=valid.sum(axis=2).astype(np.float64)
    return num,cnt

def score_arrays(pred,target,q,scale):
    num,cnt=pinball_components(pred,target,q); scale=np.asarray(scale,dtype=np.float64)
    if (not np.isfinite(num).all()) or np.any(cnt.sum(axis=0)<=0) or np.any(scale<=0): raise ValueError('empty/nonfinite score input')
    return float((num.sum(axis=0)/cnt.sum(axis=0)[:,None]/scale[:,None]).mean())

def score_npz(p: Path):
    with np.load(p,allow_pickle=False) as z: return score_arrays(z['prediction'],z['target'],z['quantiles'],z['scale'])
def arrs(p: Path, keys):
    with np.load(p,allow_pickle=False) as z: return {k:z[k].copy() for k in keys}
def pred_sorted(p: Path):
    with np.load(p,allow_pickle=False) as z: return bool(np.all(np.diff(z['prediction'],axis=2)>=-1e-6))

def correction_parameters(train_npz: Path):
    with np.load(train_npz,allow_pickle=False) as z:
        pred=np.sort(z['prediction'].astype(np.float64),axis=2); y=z['target'].astype(np.float64); q=z['quantiles'].astype(np.float64); scale=z['scale'].astype(np.float64)
    mi=int(np.argmin(np.abs(q-.5))); med=pred[:,:,mi,:]; slopes=[]; ints=[]; biases=[]; counts=[]; ridge=1e-6
    for c in range(y.shape[1]):
        x=med[:,c].reshape(-1); yy=y[:,c].reshape(-1); m=np.isfinite(x)&np.isfinite(yy); counts.append(int(m.sum()))
        if m.sum()<2: slopes.append(1.0); ints.append(0.0); biases.append(0.0); continue
        x=x[m]/scale[c]; yy=yy[m]/scale[c]
        s=float(np.clip(np.mean((x-x.mean())*(yy-yy.mean()))/(np.var(x)+ridge),.25,4.0))
        slopes.append(s); ints.append(float(np.mean(yy-s*x)*scale[c])); biases.append(float(np.mean(yy-x)*scale[c]))
    return {'median_quantile_index':mi,'ridge':ridge,'slope_clip':[.25,4.0],'slopes':slopes,'intercepts':ints,'biases':biases,'fit_counts':counts}

def correction_candidates():
    out=[{'name':'identity','kind':'identity','shrinkage':0.0}]
    for s in (.25,.5,1.0): out.append({'name':f'affine_s{s:g}','kind':'affine','shrinkage':s})
    for s in (.25,.5,1.0): out.append({'name':f'bias_s{s:g}','kind':'bias','shrinkage':s})
    return out

def apply_correction(pred, params, cand):
    out=np.array(pred,dtype=np.float64,copy=True); sh=float(cand['shrinkage'])
    if cand['kind']=='identity': return out
    if cand['kind']=='affine':
        sl=np.asarray(params['slopes']); it=np.asarray(params['intercepts'])
        for c in range(out.shape[1]): out[:,c]=out[:,c]+sh*(sl[c]*out[:,c]+it[c]-out[:,c])
    elif cand['kind']=='bias':
        b=np.asarray(params['biases'])
        for c in range(out.shape[1]): out[:,c]=out[:,c]+sh*b[c]
    else: raise ValueError(cand)
    return np.sort(out,axis=2)

def choose_correction(train_npz, val_npz):
    params=correction_parameters(train_npz)
    with np.load(val_npz,allow_pickle=False) as z:
        pred=z['prediction'].astype(np.float64); y=z['target'].astype(np.float64); q=z['quantiles'].astype(np.float64); scale=z['scale'].astype(np.float64)
    scored=[]
    for order,c in enumerate(correction_candidates()): scored.append(c|{'order':order,'V':score_arrays(apply_correction(pred,params,c),y,q,scale)})
    return {'parameters':params,'candidates':scored,'selected':min(scored,key=lambda x:(x['V'],x['order']))}

def choose(rows): return min(rows,key=lambda r:(r['V'],r['step'],r['recipe']))
def cell(job): return f"{job['dataset']}/s{job['seed']}"
def trainable(fam): return 1768949 if fam in ('JOINT','WIDE','WARM_JOINT') else 589301
def fdir(key): return RUN/'fit'/key/'output'
def fcdir(key,budget): return RUN/'forecast'/key/budget/'output'
def initial_pred(key):
    p=fdir(key)/'initial_val.npz'
    return arrs(p,('prediction',))['prediction'] if p.exists() else None

def sample_hash(plan, job, steps=720):
    with np.load(rp(plan['data'][job['dataset']]['fit_path']),allow_pickle=False) as z: n=len(z['train_origins'])
    seed=int(job['seed'])+(200000 if str(job['family']).startswith('WARM_') else 0)
    return hashlib.sha256(np.random.default_rng(seed).integers(n,size=(steps,8)).tobytes()).hexdigest()

def expected_eval_payload(target_values,target_indices,eval_origins,horizon,stride,quantiles,fit_std):
    origins=np.asarray(eval_origins)[::stride].copy(); idx=np.asarray(target_indices,dtype=np.int64)
    target=np.stack([np.asarray(target_values)[int(o):int(o)+int(horizon)][:,idx].T for o in origins])
    return {'target':target,'quantiles':np.asarray(quantiles).copy(),'scale':np.asarray(fit_std)[idx].copy(),'origins':origins}

def expected_forecast_payload(plan,dataset):
    spec=plan['data'][dataset]
    with np.load(rp(spec['holdout_path']),allow_pickle=False) as z:
        return expected_eval_payload(z['target_values'],z['target_indices'],z['eval_origins'],int(z['horizon']),int(plan['eval_stride']),z['quantiles'],z['fit_std'])

def load_plan(a):
    p=RUN/'plan.json'
    if not a.ck(p.exists(),'missing_plan','plan.json missing',path=p): return None,None
    plan=rj(p); ph=sha(p)
    a.ck(ph==PLAN_SHA,'plan_hash','plan hash differs from frozen',actual=ph,expected=PLAN_SHA)
    a.ck(plan.get('study')=='peft_head_convergence_v1','study','bad study',study=plan.get('study'))
    a.ck(plan.get('schedule')==SCHED,'schedule','bad schedule',schedule=plan.get('schedule'))
    a.ck(plan.get('eval_stride')==4,'eval_stride','bad eval stride',eval_stride=plan.get('eval_stride'))
    a.ck(plan.get('D_is_development') is True and plan.get('no_final_test') is True,'dev_flags','development flags not true')
    a.ck(len(plan.get('source_hashes',{}))==52,'source_count','expected 52 source hashes',count=len(plan.get('source_hashes',{})))
    a.ck(len(plan.get('input_hashes',{}))==7,'input_count','expected 7 input hashes',count=len(plan.get('input_hashes',{})))
    jobs=plan.get('jobs',[]); pc={x:sum(1 for e in jobs if e['job']['phase']==x) for x in ('A','B')}
    a.ck(len(jobs)==96 and pc=={'A':48,'B':48},'job_counts','expected 96 jobs split 48/48',count=len(jobs),phase_counts=pc)
    return plan,ph

def verify_hashes(a,plan):
    for label,m in [('source',plan.get('source_hashes',{})),('input',plan.get('input_hashes',{}))]:
        for k,v in m.items():
            p=rp(k)
            if a.ck(p.exists(),f'missing_{label}',f'{label} path missing',path=k): a.ck(sha(p)==v,f'{label}_hash',f'{label} hash mismatch',path=k)
    old34=ROOT/'runs/peft_initial_headroom_v1/plan.json'; old33=ROOT/'runs/peft_decision_transfer_v1/plan.json'
    if old34.exists():
        o=rj(old34); miss=[k for k in o.get('source_hashes',{}) if k not in plan['source_hashes']]; bad=[k for k,v in o.get('source_hashes',{}).items() if k in plan['source_hashes'] and plan['source_hashes'][k]!=v]
        a.ck(not miss and not bad,'old34_hash','Study34 source hashes not preserved',missing=miss[:5],mismatched=bad[:5],old34_count=len(o.get('source_hashes',{})))
        a.ck(plan.get('checkpoint')==o.get('checkpoint'),'old34_checkpoint','checkpoint differs from Study34')
    else: a.issue('missing_old34','Study34 plan missing')
    if old33.exists():
        o=rj(old33); inter=set(o.get('source_hashes',{}))&set(plan.get('source_hashes',{})); bad=[k for k in inter if o['source_hashes'][k]!=plan['source_hashes'][k]]
        a.ck(not bad,'old33_hash','Study33 source hash intersection mismatch',mismatched=bad[:5],intersection_count=len(inter))
        a.ck(plan.get('checkpoint')==o.get('checkpoint'),'old33_checkpoint','checkpoint differs from Study33')
    else: a.issue('missing_old33','Study33 plan missing')

def verify_prepared(a,plan):
    for p in [RUN/'prepared/summary.json',RUN/'prepared/data_audit.json']:
        if a.ck(p.exists(),'missing_prepared_json','prepared json missing',path=p): a.ck(rj(p).get('all_qc_passed') is True,'prepared_qc','prepared qc false',path=p)
    for ds,spec in plan.get('data',{}).items():
        for role in ('fit','holdout'):
            p=rp(spec[f'{role}_path']); a.ck(p.exists(),'missing_prepared_npz','prepared npz missing',dataset=ds,role=role,path=p)
            if p.exists():
                a.ck(sha(p)==spec[f'{role}_sha256'],'prepared_npz_hash','prepared npz hash mismatch',dataset=ds,role=role)
                with np.load(p,allow_pickle=False) as z:
                    a.ck(z['target_indices'].tolist()==[0,1],'target_indices','target indices not [0,1]',dataset=ds,role=role)
                    a.ck(int(z['context'])==336 and int(z['horizon'])==48,'lh','expected L336/H48',dataset=ds,role=role)

def audit_one_fit(a,plan,idx,e):
    key=e['key']; job=e['job']; folder=fdir(key); p=folder/'result.json'
    if not p.exists(): return None
    r=rj(p); ph=sha(RUN/'plan.json')
    a.ck(r.get('completed') is True and r.get('job')==job,'fit_meta','fit result metadata mismatch',key=key)
    a.ck(r.get('plan_sha256')==ph,'fit_plan','fit plan hash mismatch',key=key)
    a.ck(r.get('smoke') is False and r.get('steps')==720,'fit_smoke_steps','fit should be non-smoke 720',key=key,smoke=r.get('smoke'),steps=r.get('steps'))
    a.ck(r.get('frozen_verified') is True and r.get('restore_exact') is True and r.get('D_opened') is False,'fit_frozen','frozen/restore/D flags bad',key=key)
    a.ck(r.get('trainable')==trainable(job['family']),'trainable','trainable mismatch',key=key,actual=r.get('trainable'),expected=trainable(job['family']))
    hist=r.get('history',[]); a.ck([h.get('step') for h in hist]==SCHED,'history_schedule','history schedule mismatch',key=key)
    a.ck(set(r.get('budgets',{}))==({'S180','L720'} if job['phase']=='A' else {'L720'}),'budget_set','budget set mismatch',key=key,budgets=sorted(r.get('budgets',{})))
    if 'sample_sha256' in r: a.ck(r['sample_sha256']==sample_hash(plan,job),'sample_hash','sample stream hash mismatch',key=key)
    if (folder/'initial_val.npz').exists(): a.ck(close(score_npz(folder/'initial_val.npz'),float(r.get('initial_V',np.nan))),'initial_score','initial_val score mismatch',key=key)
    for h in hist:
        pp=folder/f"point_{int(h['step'])}.npz"
        if a.ck(pp.exists(),'missing_point','point_N npz missing',key=key,step=h.get('step')): a.ck(close(score_npz(pp),float(h['V'])),'point_score','point score != history V',key=key,step=h['step'])
    for b,bp in r.get('budgets',{}).items():
        cap=180 if b=='S180' else 720; cand=[h for h in hist if h['step']<=cap]
        if cand:
            exp=min(cand,key=lambda h:(h['V'],h['step']))
            a.ck(close(float(bp['V']),float(exp['V'])) and int(bp['step'])==int(exp['step']),'budget_min','budget min rule mismatch',key=key,budget=b,actual=bp,expected=exp)
        vp=folder/f'{b}_val.npz'; cp=folder/f'{b}.pt'
        if a.ck(vp.exists(),'missing_budget_npz','budget val npz missing',key=key,budget=b): a.ck(close(score_npz(vp),float(bp['V'])),'budget_score','budget val score mismatch',key=key,budget=b)
        if a.ck(cp.exists(),'missing_checkpoint','checkpoint missing',key=key,budget=b): a.ck(sha(cp)==bp.get('checkpoint_sha256'),'checkpoint_hash','checkpoint sha mismatch',key=key,budget=b)
    sp=RUN/'fit'/key/'guard/status.json'
    if sp.exists():
        s=rj(sp); a.ck(s.get('completed') is True and s.get('returncode')==0 and not s.get('reasons'),'fit_guard','fit guard not clean',key=key)
    return r

def audit_smokes(a,plan,need):
    for phase,fams in [('A',AFIT),('B',BFIT)]:
        for fam in fams:
            p=RUN/'smoke'/phase/fam/'output/result.json'
            if not p.exists():
                if need and phase=='A': a.issue('missing_smoke','A smoke missing',phase=phase,family=fam)
                continue
            r=rj(p); a.ck(r.get('completed') is True and r.get('smoke') is True and r.get('steps')==3,'smoke_result','bad smoke result',phase=phase,family=fam)
            sp=RUN/'smoke'/phase/fam/'guard/status.json'
            if sp.exists():
                s=rj(sp); a.ck(s.get('completed') is True and s.get('returncode')==0 and not s.get('reasons'),'smoke_guard','smoke guard not clean',phase=phase,family=fam)

def audit_fits(a,plan,need_all):
    res={}
    for i,e in enumerate(plan.get('jobs',[])):
        r=audit_one_fit(a,plan,i,e)
        if r is None:
            if need_all: a.issue('missing_fit','fit result missing',key=e['key'])
        else: res[e['key']]=r
    a.counts['fit_results_seen']=len(res)
    check_A_pairs(a,plan,res); check_B_deps(a,plan,res)
    return res

def check_A_pairs(a,plan,res):
    g={}
    for e in plan['jobs']:
        j=e['job']
        if j['phase']=='A' and e['key'] in res: g.setdefault(cell(j),[]).append((e,res[e['key']],initial_pred(e['key'])))
    for c,items in g.items():
        if len(items)!=12: continue
        a.ck(len({x[1].get('sample_sha256') for x in items})==1,'A_sample_pair','A sample hashes differ',cell=c)
        a.ck(len({x[1].get('initial_head_hash') for x in items if x[0]['job']['family'] in ('HEAD','JOINT')})==1,'A_head_pair','A HEAD/JOINT initial head hashes differ',cell=c)
        preds=[x[2] for x in items if x[2] is not None]
        if len(preds)==12: a.ck(all(np.array_equal(preds[0],p) for p in preds[1:]),'A_initial_pred','A initial predictions differ',cell=c)

def check_B_deps(a,plan,res):
    selp=RUN/'selection_A.json'
    if not selp.exists(): return
    sela=rj(selp)
    for e in plan['jobs']:
        j=e['job']; key=e['key']
        if j['phase']!='B' or key not in res: continue
        r=res[key]; dep=r.get('dependency') or {}; ac=sela.get('cells',{}).get(cell(j),{}).get('L720',{})
        if j['family']=='REFIT':
            sj=ac.get('JOINT'); sh=ac.get('HEAD')
            if sj:
                a.ck(dep.get('source_family')=='JOINT' and dep.get('source_key')==sj['key'],'B_refit_source','REFIT source not sealed A JOINT',key=key)
                p=fdir(sj['key'])/'L720.pt'
                if p.exists(): a.ck(dep.get('source_hash')==sha(p),'B_refit_source_hash','REFIT source hash mismatch',key=key)
            if sh and sh['key'] in res:
                hr=res[sh['key']]
                a.ck(r.get('initial_head_hash')==hr.get('initial_head_hash'),'B_refit_head','REFIT initial head != A HEAD init',key=key)
                a.ck(r.get('sample_sha256')==hr.get('sample_sha256'),'B_refit_sample','REFIT sample != A HEAD sample',key=key)
        elif j['family'].startswith('WARM_'):
            sh=ac.get('HEAD')
            if sh:
                a.ck(dep.get('source_family')=='HEAD' and dep.get('source_key')==sh['key'],'B_warm_source','WARM source not sealed A HEAD',key=key)
                p=fdir(sh['key'])/'L720.pt'
                if p.exists(): a.ck(dep.get('source_hash')==sha(p),'B_warm_source_hash','WARM source hash mismatch',key=key)
                ip=initial_pred(key); src=fdir(sh['key'])/'L720_val.npz'
                if ip is not None and src.exists(): a.ck(np.array_equal(ip,arrs(src,('prediction',))['prediction']),'B_warm_initial_pred','WARM initial pred != sealed A HEAD L720',key=key)
            a.ck(dep.get('initial_prediction_exact') is True,'B_warm_exact_flag','WARM exact flag missing',key=key)

def candidates(plan,res,phase):
    out={}
    for i,e in enumerate(plan['jobs']):
        j=e['job']; key=e['key']
        if j['phase']!=phase or key not in res: continue
        for b,bp in res[key].get('budgets',{}).items():
            x=dict(bp); x.update({'index':i,'key':key,'recipe':j['recipe'],'trainable':res[key].get('trainable'),'trajectory_seconds':res[key].get('trajectory_seconds')})
            out.setdefault((cell(j),b,j['family']),[]).append(x)
    return out

def audit_selection(a,plan,res,phase,need):
    p=RUN/f'selection_{phase}.json'
    if not p.exists():
        if need: a.issue('missing_selection','selection seal missing',phase=phase)
        return None
    sel=rj(p); a.ck(sel.get('plan_sha256')==sha(RUN/'plan.json'),'selection_plan','selection plan hash mismatch',phase=phase)
    a.ck(sel.get('selection_uses_V_only') is True,'selection_v_only','selection V-only flag missing',phase=phase)
    cand=candidates(plan,res,phase); exp_fore=[]
    for (c,b,f),opts in cand.items():
        a.ck(len(opts)==4,'selection_candidate_count','expected four recipes',phase=phase,cell=c,budget=b,family=f,count=len(opts))
        ch=choose(opts); exp_fore.append([ch['index'],b]); act=sel.get('cells',{}).get(c,{}).get(b,{}).get(f)
        if act is None: a.issue('selection_missing_family','selection missing family',phase=phase,cell=c,budget=b,family=f); continue
        for fld in ('index','key','recipe','step','selected_hash','checkpoint_sha256'):
            a.ck(act.get(fld)==ch.get(fld),'selection_field','selection field mismatch',phase=phase,cell=c,budget=b,family=f,field=fld,actual=act.get(fld),expected=ch.get(fld))
        a.ck(close(float(act['V']),float(ch['V'])),'selection_V','selection V mismatch',phase=phase,cell=c,budget=b,family=f)
    a.ck(sorted(sel.get('forecasts',[]))==sorted(exp_fore),'selection_forecasts','forecast list mismatch',phase=phase,actual=sel.get('forecasts'),expected=sorted(exp_fore))
    for key,h in sel.get('fit_hashes',{}).items():
        rp2=fdir(key)/'result.json'
        if rp2.exists(): a.ck(sha(rp2)==h,'selection_fit_hash','fit hash in selection mismatch',phase=phase,key=key)
    if phase=='A':
        for c,payload in sel.get('cells',{}).items():
            head0=next((e for e in plan['jobs'] if e['job']['phase']=='A' and e['job']['family']=='HEAD' and e['job']['recipe']==0 and cell(e['job'])==c),None)
            if head0:
                folder=fdir(head0['key'])
                if (folder/'F0_train.npz').exists() and (folder/'initial_val.npz').exists() and payload.get('correction'):
                    exp=choose_correction(folder/'F0_train.npz',folder/'initial_val.npz')
                    a.ck(payload['correction'].get('selected')==exp.get('selected'),'correction_selection','correction selection mismatch',cell=c,actual=payload['correction'].get('selected'),expected=exp.get('selected'))
    if sel.get('sealed_utc'):
        sealed=dt(sel['sealed_utc'])
        result_count=0; guard_count=0
        for e in plan['jobs']:
            if e['job']['phase']!=phase: continue
            key=e['key']; rp2=fdir(key)/'result.json'; sp2=RUN/'fit'/key/'guard/status.json'
            if rp2.exists():
                result_count+=1; fin=rj(rp2).get('finished_utc')
                if fin: a.ck(dt(fin)<=sealed,'fit_result_after_seal','fit result finished after phase selection seal',phase=phase,key=key,finished_utc=fin,sealed_utc=sel['sealed_utc'])
            if sp2.exists():
                guard_count+=1; fin=rj(sp2).get('finished_at')
                if fin: a.ck(dt(fin)<=sealed,'fit_guard_after_seal','fit guard finished after phase selection seal',phase=phase,key=key,finished_at=fin,sealed_utc=sel['sealed_utc'])
        a.ck(result_count==48 and guard_count==48,'phase_fit_count_before_seal','phase selection exists before all 48 fit results/guards are present',phase=phase,result_count=result_count,guard_count=guard_count)
        for idx,b in sel.get('forecasts',[]):
            key=plan['jobs'][int(idx)]['key']; sp=RUN/'forecast'/key/b/'guard/status.json'
            if sp.exists(): a.ck(dt(rj(sp)['started_at'])>=sealed,'forecast_before_seal','forecast guard started before seal',phase=phase,key=key,budget=b,sealed=sel['sealed_utc'],started=rj(sp).get('started_at'))
    return sel

def audit_forecast_payload(a,plan,dataset,path,phase,key,budget,label):
    if not path.exists(): return
    exp=expected_forecast_payload(plan,dataset)
    with np.load(path,allow_pickle=False) as z:
        for name in ('target','quantiles','scale','origins'):
            a.ck(name in z.files,'forecast_npz_missing_key','forecast NPZ missing expected key',phase=phase,key=key,budget=budget,label=label,npz=path,name=name)
        if all(name in z.files for name in ('target','quantiles','scale','origins')):
            a.ck(eqarr(z['target'],exp['target']),'forecast_target_payload','forecast target does not match prepared holdout eval_origins[::stride]',phase=phase,key=key,budget=budget,label=label,npz=path)
            a.ck(eqarr(z['quantiles'],exp['quantiles']),'forecast_quantiles_payload','forecast quantiles do not match prepared holdout',phase=phase,key=key,budget=budget,label=label,npz=path)
            a.ck(eqarr(z['scale'],exp['scale']),'forecast_scale_payload','forecast scale does not match prepared holdout fit_std[target_indices]',phase=phase,key=key,budget=budget,label=label,npz=path)
            a.ck(np.array_equal(z['origins'],exp['origins']),'forecast_origins_payload','forecast origins do not match prepared eval_origins[::stride]',phase=phase,key=key,budget=budget,label=label,npz=path,actual=z['origins'].tolist(),expected=exp['origins'].tolist())

def audit_forecasts(a,plan,sel,phase,need):
    if not sel: return
    for idx,b in sel.get('forecasts',[]):
        e=plan['jobs'][int(idx)]; key=e['key']; job=e['job']; folder=fcdir(key,b); p=folder/'result.json'
        if not p.exists():
            if need: a.issue('missing_forecast','forecast result missing',phase=phase,key=key,budget=b)
            continue
        r=rj(p); a.ck(r.get('completed') is True and r.get('job')==job and r.get('budget')==b,'forecast_meta','forecast metadata mismatch',phase=phase,key=key,budget=b)
        a.ck(r.get('plan_sha256')==sha(RUN/'plan.json'),'forecast_plan','forecast plan hash mismatch',phase=phase,key=key,budget=b)
        src=fdir(key); sr=rj(src/'result.json') if (src/'result.json').exists() else None
        if sr:
            bp=sr.get('budgets',{}).get(b,{})
            if (src/f'{b}.pt').exists(): a.ck(sha(src/f'{b}.pt')==bp.get('checkpoint_sha256'),'forecast_source_ckpt','forecast source checkpoint hash mismatch',phase=phase,key=key,budget=b)
            a.ck(r.get('selected_hash')==bp.get('selected_hash'),'forecast_selected_hash','forecast selected hash mismatch',phase=phase,key=key,budget=b)
        if (folder/'selected.npz').exists():
            a.ck(close(score_npz(folder/'selected.npz'),float(r.get('D_score',np.nan))),'forecast_score','forecast selected score mismatch',phase=phase,key=key,budget=b)
            a.ck(pred_sorted(folder/'selected.npz'),'forecast_sorted','forecast selected prediction unsorted',phase=phase,key=key,budget=b)
            audit_forecast_payload(a,plan,job['dataset'],folder/'selected.npz',phase,key,b,'selected')
        else: a.issue('missing_selected_npz','selected forecast npz missing',phase=phase,key=key,budget=b)
        if phase=='A' and job['family']=='HEAD':
            a.ck((folder/'F0.npz').exists(),'missing_F0_npz','A HEAD forecast missing F0.npz',key=key,budget=b)
            if (folder/'F0.npz').exists():
                a.ck(pred_sorted(folder/'F0.npz'),'F0_sorted','F0 prediction unsorted',key=key,budget=b)
                audit_forecast_payload(a,plan,job['dataset'],folder/'F0.npz',phase,key,b,'F0')
        sp=RUN/'forecast'/key/b/'guard/status.json'
        if sp.exists():
            s=rj(sp); a.ck(s.get('completed') is True and s.get('returncode')==0 and not s.get('reasons'),'forecast_guard','forecast guard not clean',phase=phase,key=key,budget=b)

def pred_arrays(phase,cell_payload,budget,fam):
    if fam in ('F0','CORRECTION'): sel=cell_payload[budget]['HEAD']; name='F0'
    else: sel=cell_payload[budget][fam]; name='selected'
    ar=arrs(fcdir(sel['key'],budget)/f'{name}.npz',('prediction','target','quantiles','scale'))
    if fam=='CORRECTION': ar['prediction']=apply_correction(ar['prediction'],cell_payload['correction']['parameters'],cell_payload['correction']['selected'])
    return ar,sel

def rows_for(phase,a_rows=None):
    sel=rj(RUN/f'selection_{phase}.json'); rows=[]; parts=[]
    for ck,cp in sorted(sel['cells'].items()):
        for b in (('S180','L720') if phase=='A' else ('L720',)):
            f0=None if phase=='A' else next(r['D_score'] for r in a_rows if r['cell']==ck and r['budget']=='L720' and r['family']=='F0')
            ref=None
            for fam in (AREP if phase=='A' else BREP):
                ar,seli=pred_arrays(phase,cp,b,fam); p,y,q,s=(ar[k] for k in ('prediction','target','quantiles','scale'))
                if ref is None: ref=(y,q,s)
                else:
                    for aa,bb in zip((y,q,s),ref):
                        if not eqarr(aa,bb): raise AssertionError((phase,ck,b,fam,'target/q/scale mismatch'))
                score=score_arrays(p,y,q,s)
                if fam=='F0': f0=score
                if f0 is None: raise AssertionError((phase,ck,b,fam,'missing f0'))
                num,cnt=pinball_components(p,y,q)
                rows.append({'phase':phase,'cell':ck,'dataset':cp['dataset'],'seed':cp['seed'],'budget':b,'family':fam,'D_score':score,'F0_score':f0,'D_over_F0':score/f0,'V':seli['V'] if fam not in ('F0','CORRECTION') else None,'step':seli['step'] if fam not in ('F0','CORRECTION') else 0,'recipe':seli['recipe'] if fam not in ('F0','CORRECTION') else None})
                parts.append((num,cnt,s))
    return rows,parts

def gate(rows,fam,comps):
    vals=[]
    for r in rows:
        if r['family']!=fam or r['budget']!='L720': continue
        other=[o['D_score'] for o in rows if o['cell']==r['cell'] and o['budget']=='L720' and o['family'] in comps]
        vals.append({'cell':r['cell'],'dataset':r['dataset'],'seed':r['seed'],'gain_pct_F0':100*(min(other)-r['D_score'])/r['F0_score']})
    by={d:all(v['gain_pct_F0']>.25 for v in vals if v['dataset']==d) for d in sorted({v['dataset'] for v in vals})}
    return {'passes':any(by.values()),'source_passes':by,'values':vals}

def ridx(rows): return {(r['phase'],r['cell'],r['budget'],r['family']):r for r in rows}
def cmp_rows(a,actual,expected,label):
    aa=ridx(actual); ee=ridx(expected); a.ck(set(aa)==set(ee),'row_keys','summary row keys mismatch',label=label,missing=sorted(set(ee)-set(aa))[:5],extra=sorted(set(aa)-set(ee))[:5])
    for k in sorted(set(aa)&set(ee)):
        x=aa[k]; y=ee[k]
        for f in ('dataset','seed','budget','family','step','recipe'): a.ck(x.get(f)==y.get(f),'row_field','summary row field mismatch',label=label,key=k,field=f,actual=x.get(f),expected=y.get(f))
        for f in ('D_score','F0_score','D_over_F0','V'):
            if x.get(f) is None or y.get(f) is None: a.ck(x.get(f)==y.get(f),'row_none','summary row nullable mismatch',label=label,key=k,field=f)
            else: a.ck(close(float(x[f]),float(y[f])),'row_float','summary row float mismatch',label=label,key=k,field=f,actual=x[f],expected=y[f])

def audit_summaries(a,stage_b):
    ra,pa=rows_for('A'); allrows=list(ra); parts=list(pa); pa_path=OUT/'summary_A.json'
    if pa_path.exists():
        sa=rj(pa_path); a.ck(sa.get('completed') is True and sa.get('plan_sha256')==sha(RUN/'plan.json'),'summary_A_meta','summary_A metadata bad')
        cmp_rows(a,sa.get('rows',[]),ra,'summary_A'); g1=gate(ra,'JOINT',('F0','HEAD','WIDE','CORRECTION'))
        a.ck(sa.get('gates',{}).get('G1')==jv(g1),'summary_A_gate','summary_A G1 mismatch',actual=sa.get('gates',{}).get('G1'),expected=g1)
    else: a.issue('missing_summary_A','summary_A.json missing')
    if stage_b:
        rb,pb=rows_for('B',ra); allrows+=rb; parts+=pb; pb_path=OUT/'summary_B.json'
        if pb_path.exists():
            sb=rj(pb_path); a.ck(sb.get('completed') is True and sb.get('plan_sha256')==sha(RUN/'plan.json'),'summary_B_meta','summary_B metadata bad')
            cmp_rows(a,sb.get('rows',[]),rb,'summary_B'); comb=ra+rb; g2=gate(comb,'REFIT',('F0','HEAD')); g3=gate(comb,'WARM_JOINT',('F0','WARM_HEAD'))
            gates=sb.get('gates',{}); a.ck(gates.get('G2_refit')==jv(g2),'summary_B_G2','summary_B G2 mismatch'); a.ck(gates.get('G3_warm')==jv(g3),'summary_B_G3','summary_B G3 mismatch')
            sga=rj(pa_path).get('gates',{}).get('G1',{}).get('source_passes',{}) if pa_path.exists() else {}
            same=[d for d in sga if sga[d] and g2['source_passes'].get(d) and g3['source_passes'].get(d)]
            a.ck(gates.get('same_source_all_three')==same,'same_source','same-source gate mismatch',actual=gates.get('same_source_all_three'),expected=same)
        else: a.issue('missing_summary_B','summary_B.json missing')
    return allrows,parts

def audit_metrics_losses(a,rows,parts):
    mp=OUT/'metrics.csv'
    if mp.exists():
        with mp.open('r',encoding='utf-8',newline='') as f: mr=list(csv.DictReader(f))
        a.ck(len(mr)==len(rows),'metrics_count','metrics row count mismatch',actual=len(mr),expected=len(rows))
        for i,(x,y) in enumerate(zip(mr,rows)):
            for k,v in y.items():
                if v is None: a.ck(x.get(k,'') in ('','None'),'metrics_none','metrics None mismatch',row=i,field=k)
                elif isinstance(v,float): a.ck(close(float(x[k]),v),'metrics_float','metrics float mismatch',row=i,field=k,actual=x.get(k),expected=v)
                else: a.ck(str(x.get(k))==str(v),'metrics_field','metrics field mismatch',row=i,field=k,actual=x.get(k),expected=v)
    else: a.issue('missing_metrics','metrics.csv missing')
    lp=OUT/'per_example_losses.npz'
    if lp.exists():
        nums=np.stack([p[0] for p in parts]); cnt=np.stack([p[1] for p in parts]); sc=np.stack([p[2] for p in parts]); keys=np.array([json.dumps(r,sort_keys=True) for r in rows])
        with np.load(lp,allow_pickle=False) as z:
            a.ck(aclose(z['pinball_numerator'],nums),'loss_num','loss numerators mismatch')
            a.ck(aclose(z['valid_target_count'],cnt),'loss_count','loss counts mismatch')
            a.ck(aclose(z['scale'],sc),'loss_scale','loss scales mismatch')
            a.ck(aclose(z['F0'],np.array([r['F0_score'] for r in rows])),'loss_f0','loss F0 mismatch')
            a.ck(np.array_equal(z['row_keys'],keys),'loss_keys','loss row_keys mismatch')
    else: a.issue('missing_losses','per_example_losses.npz missing')

def intervals(rows,parts):
    nums=np.stack([p[0] for p in parts]); cnt=np.stack([p[1] for p in parts]); sc=np.stack([p[2] for p in parts]); rng=np.random.default_rng(30355); weights={}
    for d in sorted({r['dataset'] for r in rows}):
        starts=rng.integers(0,20,size=(2000,10)); draws=np.stack([starts,(starts+1)%20],axis=-1).reshape(2000,20); weights[d]=np.stack([np.bincount(v,minlength=20) for v in draws])
    boots=[]
    for i,r in enumerate(rows):
        w=weights[r['dataset']]; boots.append((np.einsum('bo,ocq->bcq',w,nums[i])/(w@cnt[i])[:,:,None]/sc[i][None,:,None]).mean((1,2)))
    def idx(c,b,f): return next(i for i,r in enumerate(rows) if r['cell']==c and r['budget']==b and r['family']==f)
    contrasts=[('JOINT','HEAD'),('JOINT','WIDE'),('JOINT','CORRECTION')]
    if any(r['phase']=='B' for r in rows): contrasts += [('REFIT','HEAD'),('WARM_JOINT','WARM_HEAD')]
    out={}
    for fam,other in contrasts:
        diffs=[]
        for c in sorted({r['cell'] for r in rows}):
            i=idx(c,'L720',fam); j=idx(c,'L720',other); f=idx(c,'L720','F0'); diffs.append(100*(boots[j]-boots[i])/boots[f])
        out[f'{fam}_vs_{other}']=np.quantile(np.mean(diffs,axis=0),[.05,.95]).tolist()
    return {'conditional_90pct':out,'draws':2000,'block_origins':2,'fixed_source_periods':True}

def audit_intervals(a,rows,parts):
    p=OUT/'intervals.json'
    if not p.exists(): a.issue('missing_intervals','intervals.json missing'); return
    act=rj(p); exp=intervals(rows,parts)
    a.ck(act.get('draws')==2000 and act.get('block_origins')==2 and act.get('fixed_source_periods') is True,'interval_meta','interval metadata mismatch')
    a.ck(set(act.get('conditional_90pct',{}))==set(exp['conditional_90pct']),'interval_keys','interval keys mismatch')
    for k,v in exp['conditional_90pct'].items():
        if k in act.get('conditional_90pct',{}): a.ck(aclose(np.asarray(act['conditional_90pct'][k]),np.asarray(v),1e-10),'interval_value','interval values mismatch',key=k,actual=act['conditional_90pct'][k],expected=v)

def audit_guards(a,stage_b):
    paths=sorted(RUN.glob('**/guard/status.json')); cats={'smoke_A':0,'fit_A':0,'forecast_A':0,'smoke_B':0,'fit_B':0,'forecast_B':0,'other':0}
    for p in paths:
        parts=p.relative_to(RUN).parts
        if parts[0]=='smoke': cats[f'smoke_{parts[1]}']+=1
        elif parts[0]=='fit': cats[f'fit_{parts[1]}']+=1
        elif parts[0]=='forecast': cats[f'forecast_{parts[1]}']+=1
        else: cats['other']+=1
        s=rj(p); a.ck(s.get('completed') is True and s.get('returncode')==0 and not s.get('reasons'),'guard_clean','guard not clean',path=p)
    exp={'smoke_A':3,'fit_A':48,'forecast_A':24,'smoke_B':3 if stage_b else 0,'fit_B':48 if stage_b else 0,'forecast_B':12 if stage_b else 0,'other':0}; total=138 if stage_b else 75
    a.ck(len(paths)==total,'guard_count','guard count mismatch',actual=len(paths),expected=total,categories=cats)
    a.ck(cats==exp,'guard_categories','guard category mismatch',actual=cats,expected=exp)
    a.counts['guard_statuses']=len(paths); a.counts['guard_categories']=cats

def audit_time_order(a,plan,stage_b):
    if stage_b:
        sp=OUT/'summary_A.json'
        if a.ck(sp.exists(),'missing_summary_A_time','Stage B executed but summary_A.json is missing for timestamp guard'):
            cut=dt(rj(sp)['analysed_utc']); count=0
            for gp in sorted((RUN/'smoke'/'B').glob('*/guard/status.json'))+sorted((RUN/'fit'/'B').glob('*/*/*/*/guard/status.json')):
                count+=1; start=rj(gp).get('started_at')
                if start: a.ck(dt(start)>=cut,'B_started_before_summary_A','Stage-B smoke/fit guard started before summary_A analysed_utc',path=gp,started_at=start,summary_A_analysed_utc=rj(sp)['analysed_utc'])
            a.ck(count==51,'B_start_guard_count','expected 3 B smoke + 48 B fit guards for timestamp check',count=count)
    finishes=[]
    for gp in RUN.glob('**/guard/status.json'):
        item=rj(gp); fin=item.get('finished_at')
        if fin: finishes.append((dt(fin),gp))
    if not finishes:
        a.issue('no_guard_finishes','no guard finished_at timestamps found')
        return
    last,last_path=max(finishes,key=lambda x:x[0])
    for p,label in ((OUT/'completed.json','OUT'),(RUN/'completed.json','RUN')):
        if a.ck(p.exists(),f'missing_{label}_completed_time',f'{label} completed.json missing for timestamp guard'):
            comp=rj(p); fin=comp.get('finished_utc')
            a.ck(bool(fin),f'{label}_completed_missing_finished_utc',f'{label} completed.json missing finished_utc',path=p)
            if fin: a.ck(dt(fin)>=last,f'{label}_completed_before_last_guard',f'{label} completion timestamp is before last guard finished_at',completed_utc=fin,last_guard_finished_at=last.isoformat(),last_guard=last_path)

def payload(a,plan_hash,stage_b):
    return {'completed':a.mode=='run' and not a.issues,'mode':a.mode,'created_utc':now(),'plan_sha256':plan_hash,'expected_plan_sha256':PLAN_SHA,'stage_B_executed':stage_b,'pass':not a.issues,'issue_count':len(a.issues),'issues':a.issues,'notes':a.notes,'counts':jv(a.counts),'implementation':'stdlib+numpy independent audit; no torch or Study35 analyse import'}

def run_audit(mode):
    a=A(mode); plan,ph=load_plan(a)
    if plan is None: return payload(a,None,None)
    verify_hashes(a,plan); verify_prepared(a,plan); need=(mode=='run')
    if need:
        cp=RUN/'completed.json'
        if not cp.exists(): a.issue('run_not_completed','RUN/completed.json missing; wait for main completion'); return payload(a,ph,None)
        comp=rj(cp); a.ck(comp.get('completed') is True,'run_completed','RUN completed flag false'); a.ck(comp.get('plan_sha256')==ph,'run_plan_hash','RUN completed plan hash mismatch'); stage_b=bool(comp.get('stage_B_executed'))
    else: stage_b=(RUN/'selection_B.json').exists()
    audit_smokes(a,plan,need); res=audit_fits(a,plan,need and bool(stage_b))
    if need and not stage_b:
        for e in plan['jobs']:
            if e['job']['phase']=='A' and e['key'] not in res: a.issue('missing_fit','required A fit missing',key=e['key'])
    sela=audit_selection(a,plan,res,'A',need); audit_forecasts(a,plan,sela,'A',need)
    if stage_b:
        selb=audit_selection(a,plan,res,'B',need); audit_forecasts(a,plan,selb,'B',need)
    elif need: a.ck(not (RUN/'selection_B.json').exists(),'unexpected_selection_B','B skipped but selection_B exists')
    if need:
        rows,parts=audit_summaries(a,bool(stage_b))
        g1=gate([r for r in rows if r['phase']=='A'],'JOINT',('F0','HEAD','WIDE','CORRECTION'))
        a.ck(bool(stage_b)==bool(g1['passes']),'stage_B_vs_G1','RUN completed.stage_B_executed differs from independently recomputed G1 passes',stage_B_executed=stage_b,G1_passes=g1['passes'],G1=g1)
        audit_metrics_losses(a,rows,parts); audit_intervals(a,rows,parts); audit_guards(a,bool(stage_b)); audit_time_order(a,plan,bool(stage_b))
        if (OUT/'completed.json').exists(): a.ck(rj(OUT/'completed.json').get('completed') is True,'out_completed','results completed flag false')
    return payload(a,ph,stage_b)

def fail_path(): return RUN/f"independent_audit_failed_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
def write_run(obj):
    p=FINAL if obj.get('pass') else fail_path(); wx(p,obj); return p

def selftest():
    assert eqarr(np.array([1.0,np.nan]),np.array([1.0,np.nan]))
    assert not eqarr(np.array([1.0,np.nan]),np.array([1.0,2.0]))
    tv=np.arange(10*3,dtype=float).reshape(10,3); tv[4,1]=np.nan
    exp=expected_eval_payload(tv,np.array([0,1]),np.array([1,3,5]),2,2,np.array([.1,.5]),np.array([10.,20.,30.]))
    assert exp['origins'].tolist()==[1,5]
    assert exp['target'].shape==(2,2,2)
    assert eqarr(exp['target'][0],tv[1:3][:,[0,1]].T)
    assert eqarr(exp['scale'],np.array([10.,20.]))
    pred=np.array([[[[0.,2.],[1.,3.]],[[2.,4.],[3.,5.]]]],float); y=np.array([[[1.,np.nan],[4.,2.]]],float); q=np.array([.25,.75]); sc=np.array([2.,4.])
    manual=[]
    for c in range(2):
        tmp=[]
        for qi,qq in enumerate(q):
            s=0.; n=0
            for h in range(2):
                yy=y[0,c,h]
                if np.isfinite(yy):
                    e=yy-pred[0,c,qi,h]; s+=2*max(qq*e,(qq-1)*e); n+=1
            tmp.append(s/n/sc[c])
        manual.append(tmp)
    assert abs(score_arrays(pred,y,q,sc)-float(np.mean(manual)))<1e-12
    assert pinball_components(pred,y,q)[1].tolist()==[[1.0,2.0]]
    assert choose([{'V':1,'step':2,'recipe':0},{'V':1,'step':1,'recipe':3},{'V':.9,'step':5,'recipe':1},{'V':.9,'step':5,'recipe':0}])['recipe']==0
    with tempfile.TemporaryDirectory() as d:
        p=Path(d); qq=np.array([.1,.5,.9]); pp=np.tile(np.array([[[[1.,2.],[2.,3.],[3.,4.]]]]),(2,2,1,1)); yy=np.tile(np.array([[[2.,3.],[2.,3.]]]),(2,1,1)); ss=np.array([1.,1.]); org=np.array([0,1])
        np.savez_compressed(p/'train.npz',prediction=pp,target=yy,quantiles=qq,scale=ss,origins=org); np.savez_compressed(p/'val.npz',prediction=pp,target=yy,quantiles=qq,scale=ss,origins=org)
        cc=choose_correction(p/'train.npz',p/'val.npz'); out=apply_correction(pp,cc['parameters'],cc['selected']); assert np.all(np.diff(out,axis=2)>=-1e-12)

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--self-test',action='store_true'); ap.add_argument('--fit-only',action='store_true'); ap.add_argument('--run',action='store_true')
    ns=ap.parse_args();
    if sum(map(bool,[ns.self_test,ns.fit_only,ns.run]))!=1: ap.error('choose exactly one of --self-test, --fit-only, --run')
    if ns.self_test: selftest(); print(json.dumps({'self_test':True},indent=2)); return 0
    if ns.fit_only:
        obj=run_audit('fit_only'); print(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)); return 0 if obj['pass'] else 1
    try: obj=run_audit('run')
    except Exception as e: obj={'completed':False,'mode':'run','created_utc':now(),'pass':False,'issue_count':1,'issues':[{'code':'uncaught_exception','message':str(e),'traceback':traceback.format_exc()}],'implementation':'stdlib+numpy independent audit; no torch or Study35 analyse import'}
    dest=write_run(obj); print(json.dumps({'written':dest.as_posix(),'pass':obj.get('pass'),'issues':obj.get('issue_count')},indent=2)); return 0 if obj.get('pass') else 1
if __name__=='__main__': raise SystemExit(main())
