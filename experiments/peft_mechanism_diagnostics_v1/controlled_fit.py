"""Fixed-head, selected-backbone Chronos-2 diagnostic; separate fit and forecast stages."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='2'
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from experiments.peft_external_gap_v1 import train as shared
from experiments.peft_adaptation_scope_v1 import modeling as native
from experiments.peft_adaptation_scope_v1.guard import _available_commit
from experiments.peft_fullft_reference_v3.model import parameter_digest, snapshot_trainable, restore_trainable

ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')


class Controlled(native.AdaptationModel):
    def __init__(self,base,job,channels):
        torch.manual_seed(job['seed'])
        super().__init__(base,'H_LIN' if job['head']=='linear' else 'H_MLP',channels)
        self.initial_head_hash=parameter_digest(self,prefix='probe.')
        blocks=job['blocks']
        if blocks:
            from peft import LoraConfig, inject_adapter_in_model
            self.module_map=[f'encoder.block.{b}.layer.{layer}.self_attention.{part}'
                             for b in blocks for layer in (0,1) for part in ('q','k','v','o')]
            torch.manual_seed(job['seed']+1000)
            inject_adapter_in_model(LoraConfig(r=job['rank'],lora_alpha=2*job['rank'],lora_dropout=0.,bias='none',target_modules=self.module_map),base)
        assert all(not p.requires_grad for p in base.output_patch_embedding.parameters())
        assert all(p.requires_grad for p in self.probe.parameters())
        assert all(('lora_' in n) for n,p in base.named_parameters() if p.requires_grad)
        native.deterministic_backbone(base)

    def encode(self,context,group_ids,horizon):
        hidden,norm,loc,scale=super().encode(context,group_ids,horizon)
        residual=native.patch_to_quantiles(self.probe(hidden),self.n_quantiles,self.patch_size)
        return hidden,norm+residual.float(),loc,scale


def check(args):
    shared.legacy.check_resources(args)
    commit=_available_commit()/2**30
    if commit<6:raise RuntimeError(f'available_commit_below_limit: {commit}')
    return commit


def predict(model,panel,split,args):
    model.eval();blocks=[]
    with torch.no_grad():
        for i in range(0,len(panel.origins[split]),4):
            check(args)
            origins,n=shared.padded_origins(panel.origins[split][i:i+4])
            context,_,groups=panel.batch(origins,args.device)
            with shared.legacy.precision(args):
                _,norm,loc,scale=model.encode(context,groups,48)
                raw=native.normalized_to_raw(norm,loc,scale,model.use_arcsinh)
            blocks.append(raw.float().cpu().numpy().reshape(4,panel.count_channels,21,48)[:n,panel.target_indices])
    pred=np.sort(np.concatenate(blocks),axis=2)
    if not np.isfinite(pred).all():raise FloatingPointError('nonfinite forecast')
    metrics,_,_=shared.scores(pred,panel.targets(split),panel.fit_std[panel.target_indices],panel.quantiles)
    return pred,metrics['score']


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--contract',required=True);parser.add_argument('--job',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--forecast-fit')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();started=time.perf_counter()
    contract=json.loads(Path(args.contract).read_text());job=json.loads(args.job)
    for path,digest in contract['source_hashes'].items():
        assert sha(ROOT/path)==digest,path
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    settings=contract['reference'];spec=settings['datasets'][job['dataset']]
    forecast=args.forecast_fit is not None
    data_path=ROOT/spec['holdout_data_path' if forecast else 'fit_data_path']
    assert sha(data_path)==spec['holdout_data_sha256' if forecast else 'fit_data_sha256']
    panel=shared.Panel(data_path,'forecast' if forecast else 'fit',args.smoke)
    runtime=SimpleNamespace(checkpoint=settings['checkpoint'],device='cuda',min_free_ram_gib=5.)
    check(runtime);base=shared.load_base(runtime)
    origins,_=shared.padded_origins(next(iter(panel.origins.values()))[:4])
    context,_,groups=panel.batch(origins,'cuda')
    with torch.no_grad(),shared.legacy.precision(runtime):
        enc,(loc,scale),_,_=base.encode(context=context,group_ids=groups,num_output_patches=3)
        reference=native.patch_to_quantiles(base.output_patch_embedding(enc.last_hidden_state[:,-3:]),21,16).float()
    model=Controlled(base,job,panel.count_channels)
    with torch.no_grad(),shared.legacy.precision(runtime):
        _,initial,_,_=model.encode(context,groups,48)
    identity=float((initial-reference).abs().max());assert identity<1e-5,identity
    del enc,reference,initial,context,groups
    frozen=parameter_digest(model,trainable=False)
    params=[p for p in model.parameters() if p.requires_grad]
    count=sum(p.numel() for p in params)
    if forecast:
        fit=Path(args.forecast_fit);record=json.loads((fit/'result.json').read_text())
        assert record['job']==job and record['contract_sha256']==sha(args.contract)
        assert sha(fit/'best.pt')==record['checkpoint_sha256']
        state=torch.load(fit/'best.pt',map_location='cpu',weights_only=True)
        restore_trainable(model,state);del state
        prediction,score=predict(model,panel,'eval',runtime)
        np.savez_compressed(out/'predictions.npz',prediction=prediction,target=panel.targets('eval'),origins=panel.origins['eval'],quantiles=panel.quantiles,scale=panel.fit_std[panel.target_indices])
        save(out/'result.json',{'completed':True,'job':job,'score':score,'fit':str(fit),'seconds':time.perf_counter()-started,'trainable':count,'contract_sha256':sha(args.contract),'exposed_E':True})
        return
    steps=3 if args.smoke else 200
    interval=3 if args.smoke else 40
    opt_groups=[{'params':list(model.probe.parameters()),'lr':job['head_lr']}]
    adapter=[p for p in base.parameters() if p.requires_grad]
    if adapter:opt_groups.append({'params':adapter,'lr':job['lora_lr']})
    optimizer=torch.optim.AdamW(opt_groups,weight_decay=0.,foreach=False)
    assert {id(p) for g in optimizer.param_groups for p in g['params']}=={id(p) for p in params}
    samples=np.random.default_rng(job['seed']).integers(len(panel.origins['train']),size=(steps,8))
    q=torch.as_tensor(panel.quantiles,dtype=torch.float32,device='cuda')
    best_pred,best=predict(model,panel,'val',runtime);best_step=0
    state=snapshot_trainable(model);initial_trainable=parameter_digest(model,trainable=True)
    history=[{'step':0,'score':best}];minimum=check(runtime)
    for step in range(1,steps+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        for offset in (0,4):
            origins=panel.origins['train'][samples[step-1,offset:offset+4]]
            context,target,groups=panel.batch(origins,'cuda')
            with shared.legacy.precision(runtime):
                _,norm,loc,scale=model.encode(context,groups,48)
                loss=native.native_pinball(norm,target,loc,scale,q,model.use_arcsinh)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
            (loss/2).backward();minimum=min(minimum,check(runtime))
            del context,target,groups,norm,loc,scale,loss
        grad=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
        if step==1:assert grad>0
        optimizer.step()
        if step%interval==0:
            prediction,score=predict(model,panel,'val',runtime)
            history.append({'step':step,'score':score})
            if score<best:best,best_step,best_pred,state=score,step,prediction,snapshot_trainable(model)
            save(out/'progress.json',{'step':step,'best_score':best,'best_step':best_step,'elapsed':time.perf_counter()-started})
        minimum=min(minimum,check(runtime))
    assert initial_trainable!=parameter_digest(model,trainable=True)
    assert parameter_digest(model,trainable=False)==frozen
    restore_trainable(model,state)
    replay,replay_score=predict(model,panel,'val',runtime)
    np.testing.assert_allclose(replay,best_pred,rtol=0,atol=0)
    assert abs(replay_score-best)<1e-10
    torch.save(state,out/'best.pt')
    np.savez_compressed(out/'val_predictions.npz',prediction=best_pred,target=panel.targets('val'))
    save(out/'result.json',{'completed':True,'job':job,'best_score':best,'best_step':best_step,'history':history,'trainable':count,
        'initial_head_hash':model.initial_head_hash,'identity_error':identity,'frozen_hash':frozen,'checkpoint_sha256':sha(out/'best.pt'),
        'frozen_verified':True,'replay_verified':True,'steps':steps,'minimum_commit_gib':minimum,
        'max_cuda_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'seconds':time.perf_counter()-started,
        'contract_sha256':sha(args.contract),'holdout_opened':False})
    print(json.dumps({'done':job,'score':best,'steps':steps,'seconds':time.perf_counter()-started}),flush=True)


if __name__=='__main__':main()
