#!/usr/bin/env python3
"""Plot the Spider precision-collapse mechanism against synthetic phase slices."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"/"research"

def main():
    rows=list(csv.DictReader((OUT/"synthetic_selective_phase_extended_v1_summary.csv").open()))
    fig,ax=plt.subplots(figsize=(6.6,4.2))
    styles={20:("#4477AA","20-cluster synthetic slice"),116:("#CC6677","116-cluster synthetic slice")}
    for clusters,(color,label) in styles.items():
        subset=sorted((r for r in rows if int(r["cluster_count"])==clusters and float(r["label_noise"])==0),key=lambda r:float(r["proposal_precision"]))
        ax.plot([100*float(r["proposal_precision"]) for r in subset],[100*float(r["learned_gain_mean"]) for r in subset],"o-",color=color,alpha=.8,label=label)
    weak={"precision":49.28,"gain":11.51};clean={"precision":14.66,"gain":.061}
    ax.scatter(weak["precision"],weak["gain"],s=110,marker="*",color="#228833",edgecolor="black",zorder=5,label="Spider weak (observed)")
    ax.scatter(clean["precision"],clean["gain"],s=110,marker="X",color="#AA3377",edgecolor="black",zorder=5,label="Spider clean (observed)")
    ax.annotate("base-pipeline shift\nprecision collapse",xy=(clean["precision"],clean["gain"]),xytext=(weak["precision"]-3,weak["gain"]-1),
                arrowprops=dict(arrowstyle="->",lw=1.5,color="black"),ha="right",va="top")
    ax.axhline(0,color="black",lw=.8);ax.set(xlabel="Proposal precision (%)",ylabel="Selective gain over no-op (points)",title="Proposal precision and selective gain under the Spider shift")
    ax.legend(frameon=False,fontsize=8,ncol=2);fig.tight_layout();fig.savefig(OUT/"spider_precision_gain_decomposition.png",dpi=240);plt.close(fig)
    payload={"synthetic_slices":{"label_noise":0,"cluster_counts":[20,116]},"observed":{"Spider weak":weak,"Spider clean":clean},
             "interpretation":"The real points are observations, not fitted to the synthetic curves; curves show registered phase slices at each domain's cluster count."}
    (OUT/"spider_precision_gain_decomposition.json").write_text(json.dumps(payload,indent=2)+"\n")

if __name__=="__main__":main()
