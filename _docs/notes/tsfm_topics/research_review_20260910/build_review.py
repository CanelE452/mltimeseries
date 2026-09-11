"""Render the paused research review from saved results; no model imports or fitting."""
from pathlib import Path
from datetime import datetime
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FIG = HERE / 'figures'
EVIDENCE = HERE / 'evidence'
for directory in (FIG, EVIDENCE):
    directory.mkdir(exist_ok=True)
sources = {}


def read(relative):
    path = ROOT / relative
    sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding='utf-8-sig'))


def lines(relative):
    path = ROOT / relative
    sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def save(name):
    plt.savefig(FIG / (name + '.png'), dpi=190, bbox_inches='tight', facecolor='white')
    plt.savefig(FIG / (name + '.svg'), bbox_inches='tight', facecolor='white')
    plt.close()


plt.rcParams.update({'font.family': 'Malgun Gothic', 'font.size': 10,
                     'axes.unicode_minus': False, 'axes.spines.top': False,
                     'axes.spines.right': False, 'svg.fonttype': 'none'})
blue, orange, green, gray = '#0072B2', '#D55E00', '#009E73', '#737373'
s = read('results/peft_fullft_reference_v3/summary.json')
assert s['completed'] and s['metric_rows'] == 44
rows = []
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
for ax, (dataset, d) in zip(axes, s['datasets'].items()):
    base = d['arms']['F0']['SORT']['score_mean']
    for i, (arm, color) in enumerate(zip(['HEAD_ONLY', 'LORA', 'FULL_FT'], [gray, blue, orange])):
        a = d['arms'][arm]['SORT']
        gain = (base - a['score_mean']) / base * 100
        seeds = [(base - v) / base * 100 for v in a['seed_values']]
        ax.bar(i, gain, color=color, alpha=.75, width=.6)
        ax.scatter(i + np.linspace(-.13, .13, len(seeds)), seeds, color='black', s=25, zorder=3)
        ax.annotate(f'{gain:.3f}%', (i, max(gain, max(seeds))), xytext=(0, 8), textcoords='offset points', ha='center')
        rows.append({'dataset': dataset, 'arm': arm, 'loss': a['score_mean'], 'gain_pct_f0': gain})
    ax.axhline(0, color=gray, lw=.8)
    ax.set(xticks=range(3), xticklabels=['Head', 'LoRA', 'Full FT'], title=dataset.title(),
           ylabel='F0 대비 손실 감소 (% F0, 높을수록 좋음)', ylim=(-.3, 6.3))
fig.suptitle('Study20 | 표준 LoRA의 이득과 새 방법의 필요성은 별개\n54 fits · 선택된 모델 3 seed/arm · 점은 seed 값, 시간 CI가 아님', fontsize=13)
save('01_fullft_benefit')

fig, ax = plt.subplots(figsize=(11, 4.8), layout='constrained')
labels = []
for dataset, d in s['datasets'].items():
    for contrast, e in d['effects']['SORT'].items():
        i = len(labels)
        label = 'Full FT의 LoRA 대비 이득' if contrast.startswith('FULL') else 'LoRA의 Head 대비 이득'
        labels.append(dataset.title() + ' | ' + label)
        v = e['value'] * 100; low, high = np.array(e['ci']) * 100
        ax.errorbar(v, i, xerr=[[v-low], [high-v]], fmt='o', capsize=5, color=orange if contrast.startswith('FULL') else blue)
        ax.text(high + .12, i, f'{v:+.3f} [{low:+.3f}, {high:+.3f}]', va='center', fontsize=9)
ax.axvline(0, color=gray, ls='--')
ax.set(yticks=range(len(labels)), yticklabels=labels, xlim=(-1.7, 10),
       xlabel='추가 이득 (% F0) · 오른쪽이 개선',
       title='Study20 | 95% 시간 블록 CI (7일 블록, 4,000회)\n고정된 V 선택·3 seed에 조건부 · 네 대비 다중비교 보정 없음')
ax.invert_yaxis()
save('02_fullft_contrasts')

p = read('results/peft_method_pilot_screen_v1/pattern_preserving_control.json')
fig, axes = plt.subplots(1, 2, figsize=(13, 5.7), layout='constrained')
names = ['F0', '무제약 Head', '무제약 LoRA', '패턴 보존 Head\n사후 진단', 'COARSE_LIFT\n사전 V 선택']
for ax, site, title in zip(axes, ['0', '1'], ['Eagle', 'Lamb']):
    vals = [p['study17_reference'][a][site] for a in ['F0','FROZEN_HEAD','ATTN_LORA']]
    vals += [p['PATTERN_FROM_HEAD']['mse'][site], p['study17_reference']['COARSE_LIFT'][site]]
    ax.bar(range(5), vals, color=[gray,orange,orange,blue,green], alpha=.8)
    for i,v in enumerate(vals):ax.annotate(f'{v:.4f}', (i,v), xytext=(0,5), textcoords='offset points', ha='center', fontsize=9)
    ax.axhline(vals[0], color=gray, ls='--', lw=1)
    ax.set(xticks=range(5), xticklabels=names, title=title, ylabel='정규화 시간별 MSE (낮을수록 좋음)', ylim=(0,max(vals)*1.2))
    ax.tick_params(axis='x', labelsize=8)
fig.suptitle('Study17 → 22 | 패턴 손상은 크지만, 사후 회복 ≠ 사전 선택 절차의 성공\n저장 예측의 기술적 재분해 · 2 site · 이 그림은 CI/확증 검정이 아님', fontsize=13)
save('03_coarse_supervision')

run = 'runs/hospital_shared_strength_v1_run2/stages/'
r0 = read(run+'fit_s0_lr0/attempt_01/output/fit/result.json')
r1 = lines(run+'fit_s0_lr1/attempt_01/output/fit/progress.jsonl')
stop = read(run+'fit_s0_lr1/attempt_01/guard/safety_stop.json')
failure = read(run+'fit_s0_lr1/attempt_01/output/failure.json')
stages = {}
for name in ['s0','fit_s0_lr0','fit_s0_lr1']:
    status = read(run+name+'/attempt_01/guard/status.json')
    stages[name] = {k:status.get(k) for k in ['completed','state','returncode','started_at','finished_at','elapsed_seconds','reasons']}
assert r0['completed'] and not stages['fit_s0_lr1']['completed']
assert not (ROOT/'runs/hospital_shared_strength_v1_run2/completed.json').exists()
fig, ax = plt.subplots(figsize=(10,4.7), layout='constrained')
ax.plot([x['step'] for x in r0['history']], [x['validation_score'] for x in r0['history']], '-o', color=blue, label='LR 1e-5 · fit 검증 완료')
ax.plot([x['step'] for x in r1], [x['validation_score'] for x in r1], '--s', color=orange, label='LR 3e-5 · 최종 재검증 중 중단')
ax.axhline(r0['history'][0]['validation_score'], color=gray, ls=':', label='LR 1e-5의 step0')
ax.set(xlabel='학습 update', ylabel='V1 선택 손실 (낮을수록 좋음)', xticks=[0,40,80,120,160,200],
       title='Hospital | 검증 곡선만 존재, E1/E2 성능은 미측정\nseed 24000 한 개 · 두 번째 fit의 step0은 로그 미기록')
ax.legend(fontsize=9)
save('04_hospital_validation_only')

fig, axes = plt.subplots(1, 2, figsize=(12,4.8), layout='constrained')
trace_rows=[]
for name, label, color in [('fit_s0_lr0','LR 1e-5',blue),('fit_s0_lr1','LR 3e-5',orange)]:
    relative=run+name+'/attempt_01/guard/resource_log.jsonl'
    samples=[x for x in lines(relative) if 'available_commit_gib' in x]
    t0=datetime.fromisoformat(samples[0]['timestamp'])
    times=[(datetime.fromisoformat(x['timestamp'])-t0).total_seconds() for x in samples]
    axes[0].plot(times,[x['available_commit_gib'] for x in samples],color=color,label=label)
    axes[1].plot(times,[x['child_tree_rss_gib'] for x in samples],color=color,label=label)
    for sec,x in zip(times,samples):trace_rows.append({'stage':name,'seconds':sec,**x})
axes[0].axhline(6,color=gray,ls='--',label='기존 중단 기준 6 GiB')
axes[0].scatter(times[-1],stop['last_sample']['available_commit_gib'],marker='x',s=70,color=orange,zorder=5)
axes[0].set(ylabel='시스템 available commit (GiB)',xlabel='각 fit 시작 후 경과 (초)',title='시스템 전체의 여유')
axes[1].set(ylabel='child tree RSS (GiB)',xlabel='각 fit 시작 후 경과 (초)',title='학습 프로세스의 물리 메모리 상주량')
for ax in axes:ax.legend(fontsize=9)
fig.suptitle('Hospital | 5.982 GiB에서 안전 중단 · 원인 프로세스의 commit은 미측정\n시스템 commit 감소를 child RSS 또는 child commit으로 해석하면 안 됨',fontsize=12)
save('05_hospital_memory')

hospital = {'status':'PAUSED_BY_USER_INCOMPLETE','completed_main_fits':1,'planned_main_fits':4,
    'completed_eval_forecasts':0,'no_completed_json':True,'stages':stages,
    'fit0':{k:r0[k] for k in ['seed','learning_rate','best_step','val_score','steps_completed','history','checkpoint_replay','seconds','unique_sampled_series','unseen_train_series_this_fit','nominal_window_exposure_ratio']},
    'fit1_progress_only':r1,'stop':stop,'failure':failure,
    'interpretation':'V1 is for selection, not final evaluation; no hypothesis verdict. RSS is not private commit.'}
(EVIDENCE/'hospital_partial_snapshot.json').write_text(json.dumps(hospital,indent=2,ensure_ascii=False),encoding='utf-8')
with (EVIDENCE/'hospital_resource_samples.csv').open('w',newline='',encoding='utf-8') as f:
    keys=['stage','timestamp','seconds','available_commit_gib','available_ram_gib','child_tree_rss_gib']
    writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(trace_rows)
with (EVIDENCE/'fullft_plot_values.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for relative in ['results/peft_fullft_reference_v3/summary.json','results/peft_method_pilot_screen_v1/pattern_preserving_control.json']:
    # Existing versioned results remain the source; avoid a second full copy.
    assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==sources[relative]
(EVIDENCE/'figure_sources.json').write_text(json.dumps({'sources_sha256':sources,'no_training':True,
    'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'outputs':[p.name for p in sorted(FIG.iterdir())]},indent=2),encoding='utf-8')
print(json.dumps({'figures':len(list(FIG.glob('*.png'))),'hospital':hospital['status'],'sources':len(sources)}))
