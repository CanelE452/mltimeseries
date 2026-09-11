"""Descriptive fit-only snapshot during admission wait; no E outcome reads."""
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/peft_contribution_freeze_v1'
OUT=ROOT/'results/peft_contribution_freeze_v1'


def main():
    audit=json.loads((RUN/'partial_validation_audit.json').read_text())
    assert audit['completed_fits']==13 and audit['E_predictions']==0
    OUT.mkdir(exist_ok=True,parents=True)
    colors={'FULL':'#47536a','ES2':'#d38b25','FIXED_FREEZE':'#718399','CONTRIB_FREEZE':'#007e87'}
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    for i,c in enumerate(('FULL90','SPREAD30','RECENT30')):
        for arm,color in colors.items():
            r=json.loads((RUN/f'fits/bdg2/{c}/{arm}/s27000/output/result.json').read_text())
            h=r['history'];initial=h[0]['score']
            axes[0,i].plot([v['step'] for v in h],[100*(v['score']/initial-1) for v in h],'o-',color=color,label=arm,ms=4,alpha=.8)
            if arm=='CONTRIB_FREEZE':
                points=[v for v in h if 'contribution_halves' in v]
                for half,style in ((0,'-'),(1,'--')):
                    axes[1,i].plot([v['step'] for v in points],[v['contribution_halves'][half] for v in points],style+'o',label=f'V half {half+1}',color=color,ms=4)
                if r['frozen_step'] is not None:
                    for ax in axes[:,i]:ax.axvline(r['frozen_step'],ls=':',color=color,label='C freeze' if ax==axes[1,i] else None)
        axes[0,i].set_title(c);axes[0,i].set_ylabel('V loss change vs initial (%; lower better)')
        axes[1,i].set_ylabel('Current LoRA contribution (%F0_V)')
        for ax in axes[:,i]:ax.set_xlabel('Optimizer updates');ax.axhline(0,color='black',lw=.7);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);axes[1,0].legend(fontsize=8)
    fig.suptitle('PARTIAL: BDG2 seed27000 only | validation observations, NOT E performance',fontsize=14)
    fig.text(.5,.015,'13/48 fits complete. E predictions: 0. On/off observations stop after freezing and before cap.',ha='center')
    fig.tight_layout(rect=[0,.04,1,.96]);fig.savefig(OUT/'partial_01_validation.png',dpi=160);plt.close(fig)
    admission=[json.loads(s) for s in (RUN/'admission.jsonl').read_text().splitlines()]
    x=[datetime.fromisoformat(s['utc']) for s in admission]
    fig,ax=plt.subplots(figsize=(12,4.8))
    ax.plot(x,[s['available_commit_gib'] for s in admission],color='#007e87',label='Before-job available commit')
    ax.axhline(13,color='#b54141',ls='--',label='Start gate: 13 GiB, twice 5s apart')
    failed=[s for s in admission if not s['admitted'] and s['available_commit_gib']<13]
    if failed:ax.axvspan(datetime.fromisoformat(failed[0]['utc']),x[-1],color='#b54141',alpha=.1,label='New GPU job withheld')
    ax.set_ylabel('Available commit (GiB)');ax.set_xlabel('UTC on 2026-09-10 (KST = UTC + 9 h)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'));ax.grid(alpha=.2);ax.legend(fontsize=9)
    ax.set_title('Resource admission snapshot | completed fits preserved; threshold unchanged')
    fig.tight_layout();fig.savefig(OUT/'partial_02_admission.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
