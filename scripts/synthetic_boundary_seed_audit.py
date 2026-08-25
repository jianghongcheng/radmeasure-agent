#!/usr/bin/env python3
"""Repeat mechanically selected phase-boundary cells under new simulator seeds."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"/"research"

def load_base():
    path=ROOT/"scripts"/"synthetic_selective_correction_phase.py"
    spec=importlib.util.spec_from_file_location("boundary_base_phase",path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module

BASE=load_base()
CONDITIONS=[("no_op",.2,.1,100),("apply_all",.8,.2,176)]
SEEDS=list(range(20260825,20260835))

def evaluate(side,p,n,c,seed):
    values=[BASE.evaluate_condition(BASE.Condition(p,n,c,r),samples_per_cluster=40,seed=seed) for r in range(40)]
    margins=np.array([x["learned_minus_best_constant"] for x in values])
    return {"side":side,"proposal_precision":p,"label_noise":n,"cluster_count":c,"simulator_seed":seed,
            "mean_margin":float(margins.mean()),"win_probability":float((margins>0).mean()),
            "reliable":bool(margins.mean()>0 and (margins>0).mean()>=.8)}

def main():
    rows=Parallel(n_jobs=8,prefer="threads")(delayed(evaluate)(*condition,seed) for condition in CONDITIONS for seed in SEEDS)
    summary=[]
    for side,p,n,c in CONDITIONS:
        group=[x for x in rows if x["side"]==side];m=np.array([x["mean_margin"] for x in group]);w=np.array([x["win_probability"] for x in group])
        summary.append({"side":side,"proposal_precision":p,"label_noise":n,"cluster_count":c,
                        "mean_margin":float(m.mean()),"sd_margin":float(m.std(ddof=1)),
                        "mean_win_probability":float(w.mean()),"sd_win_probability":float(w.std(ddof=1)),
                        "reliable_seed_fraction":float(np.mean([x["reliable"] for x in group]))})
    payload={"protocol":"SYNTHETIC_BOUNDARY_SEED_AUDIT_V1.json","rows":rows,"summary":summary}
    (OUT/"synthetic_boundary_seed_audit_v1.json").write_text(json.dumps(payload,indent=2)+"\n")
    with (OUT/"synthetic_boundary_seed_audit_v1.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,ax=plt.subplots(figsize=(6.4,3.8))
    for i,s in enumerate(summary):
        ax.errorbar(i,100*s["mean_win_probability"],yerr=100*s["sd_win_probability"],fmt="o",capsize=5)
    ax.axhline(80,color="black",ls="--",lw=1);ax.set_xticks([0,1],["no-op-side\np=.2, q=.1, C=100","apply-all-side\np=.8, q=.2, C=176"])
    ax.set_ylabel("Win probability across repeats (%)");ax.set_ylim(0,105);ax.set_title("New-seed stability at registered phase boundaries")
    fig.tight_layout();fig.savefig(OUT/"synthetic_boundary_seed_audit_v1.png",dpi=220);plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
