"""Presentation-only rendering from the frozen metrics CSV; no model selection."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).parent
with (OUT/'metrics.csv').open(newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
for filename, columns, labels, ylabel, baseline in (
    ('01_loss_ratios', ['F0', 'HEAD', 'WIDE', 'LORA_ONLY', 'JOINT'],
     ['F0', 'HEAD', 'WIDE', 'LORA_ONLY', 'JOINT'], 'D loss / F0 (lower better)', 1),
    ('02_contrasts', ['G_LORA_ONLY', 'G_JOINT', 'I_LORA_ONLY', 'I_JOINT'],
     ['LoRA-only vs F0', 'JOINT vs F0', 'LoRA-only vs WIDE', 'JOINT vs WIDE'], '%F0 (positive better)', 0)):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5))
    for ax, budget in zip(axes, ('S180', 'L720')):
        selected = [r for r in rows if r['budget'] == budget]
        x = np.arange(len(selected))
        width = .8/len(columns)
        for i, (column, label) in enumerate(zip(columns, labels)):
            values = [float(r[column])/float(r['F0']) if baseline else float(r[column]) for r in selected]
            ax.bar(x+(i-(len(columns)-1)/2)*width, values, width, label=label)
        ax.set_xticks(x, [f"{r['dataset']} / {r['seed']}" for r in selected])
        ax.axhline(baseline, color='black', linewidth=.7)
        ax.set(title=budget+' — exposed development data', ylabel=ylabel)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='upper center', ncol=len(columns), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(OUT/'figures'/f'{filename}.png', dpi=160)
    plt.close(fig)
