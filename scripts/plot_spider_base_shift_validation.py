#!/usr/bin/env python3
"""Render the sealed Spider base-shift prediction and observed transition."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/research'
phase=json.loads((OUT/'synthetic_selective_phase_extended_v1.json').read_text())['summary']
precisions=sorted({float(r['proposal_precision']) for r in phase});clusters=sorted({int(r['cluster_count']) for r in phase})
lookup={(int(r['cluster_count']),float(r['proposal_precision'])):100*float(r['learned_minus_best_constant_mean']) for r in phase if float(r['label_noise'])==0}
matrix=np.asarray([[lookup[(c,p)] for p in precisions] for c in clusters])

fig,(ax,bx)=plt.subplots(1,2,figsize=(11.8,4.2),gridspec_kw={'width_ratios':[1.45,1]},constrained_layout=True)
im=ax.imshow(matrix,origin='lower',aspect='auto',extent=[min(precisions),max(precisions),0,len(clusters)-1],cmap='RdBu_r',vmin=-8,vmax=8)
def ypos(c): return clusters.index(c)
weak=(.49279538904899134,ypos(20));clean=(.14655172413793102,ypos(116))
ax.annotate('',xy=clean,xytext=weak,arrowprops={'arrowstyle':'->','lw':2.5,'color':'black'})
ax.scatter(*weak,s=150,marker='o',c='#ffd92f',edgecolor='black',zorder=5);ax.scatter(*clean,s=160,marker='s',c='#e78ac3',edgecolor='black',zorder=5)
ax.annotate('Weak base\n(49.28%, 0, 20)\nobserved +11.51 pt',weak,xytext=(8,9),textcoords='offset points',fontsize=9,weight='bold')
ax.annotate('Clean base\n(14.66%, 0, 116)\nobserved +0.061 pt',clean,xytext=(8,-38),textcoords='offset points',fontsize=9,weight='bold')
ax.set_xlabel('Proposal precision');ax.set_ylabel('Independent database clusters');ax.set_yticks(range(len(clusters)));ax.set_yticklabels(clusters);ax.set_title('Frozen phase prediction at zero label noise')
cb=fig.colorbar(im,ax=ax,shrink=.85);cb.set_label('Predicted selective minus best global action (pt)')

x=np.arange(2);width=.34
bx.bar(x-width/2,[49.28,14.66],width,label='Proposal precision (%)',color='#66c2a5')
bx.bar(x+width/2,[11.51,.061],width,label='Observed selective gain (pt)',color='#fc8d62')
bx.set_xticks(x);bx.set_xticklabels(['Weak base','Clean base']);bx.set_ylabel('Percentage points');bx.set_ylim(0,55);bx.legend(frameon=False,fontsize=8);bx.set_title('Observed transition')
bx.text(0-width/2,50.2,'49.28',ha='center',fontsize=9);bx.text(1-width/2,15.6,'14.66',ha='center',fontsize=9)
bx.text(0+width/2,12.5,'+11.51',ha='center',fontsize=9);bx.text(1+width/2,1.1,'+0.061',ha='center',fontsize=9)
bx.annotate('70.26% relative\nprecision collapse',xy=(1-width/2,14.66),xytext=(.55,36),arrowprops={'arrowstyle':'->'},ha='center',fontsize=9)
fig.suptitle('Sealed prediction validated under a Spider base-pipeline shift')
fig.savefig(OUT/'figure_spider_base_shift_validation.png',dpi=220,bbox_inches='tight')
