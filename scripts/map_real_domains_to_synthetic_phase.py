#!/usr/bin/env python3
"""Map three real selective-correction domains onto the frozen synthetic phase grid."""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/research"


def read(name):
    return json.loads((OUT / name).read_text())


def xray_coordinate():
    result = read("learned_proposal_cv.json")
    all_edit = result["aggregate"]["learned_repair_all"]
    benefit = all_edit["success_rate"]["mean"]
    harm = all_edit["joint_harm_rate"]["mean"]
    precision = benefit / (benefit + harm)
    split_clusters = []
    for fold in result["folds"]:
        # Three detector predictions exist for every patient.
        split_clusters.append({k: v // 3 for k, v in fold["n"].items()})

    adaptive = read("adaptive_action_advantage_cv.json")
    by_identifier = {}
    for fold in adaptive["folds"]:
        for row in fold["cases"]["adaptive_regression"]:
            gain = row["before"] - row["after"]
            by_identifier.setdefault(row["identifier"], []).append(gain)
    disagreements = []
    robust_disagreements = []
    for gains in by_identifier.values():
        for first, second in itertools.combinations(gains, 2):
            if abs(first) > 1e-6 and abs(second) > 1e-6:
                disagreements.append(np.sign(first) != np.sign(second))
            if abs(first) > 0.5 and abs(second) > 0.5:
                robust_disagreements.append(np.sign(first) != np.sign(second))

    no_repair = result["aggregate"]["no_repair"]["mean_MAE"]["mean"]
    selective = result["aggregate"]["learned_selector_learned_repair"]["mean_MAE"]["mean"]
    return {
        "domain": "X-ray",
        "clusters": len(by_identifier),
        "per_fold_patient_clusters": split_clusters,
        "proposal_precision": precision,
        "proposal_precision_robust_all_cases": benefit,
        "advantage_noise": float(np.mean(disagreements)),
        "advantage_noise_robust_tau_0_5": float(np.mean(robust_disagreements)),
        "noise_pairs": len(disagreements),
        "robust_noise_pairs": len(robust_disagreements),
        "observed_gain": no_repair - selective,
        "observed_gain_unit": "degrees MAE reduction",
        "observed_interpretation": "worse_than_no_op",
    }


def spider_coordinates():
    weak = read("spider_selector_statistics.json")["pooled"]
    proposal = weak["first_executable_edit"]
    weak_precision = proposal["benefit_rate"] / (proposal["benefit_rate"] + proposal["harm_rate"])
    weak_gain = weak["learned_exact_advantage"]["execution_accuracy"] - weak["no_repair"]["execution_accuracy"]

    clean = read("spider_clean_base_quadruple_cv.json")["pooled"]
    proposal = clean["learned_proposal_all"]
    clean_precision = proposal["benefit_count"] / (proposal["benefit_count"] + proposal["harm_count"])
    clean_gain = clean["learned_selector_learned_proposal"]["absolute_gain"]
    return [
        {
            "domain": "Spider weak", "clusters": 20,
            "proposal_precision": weak_precision,
            "proposal_precision_robust_all_cases": weak["first_executable_edit"]["benefit_rate"],
            "advantage_noise": 0.0, "observed_gain": weak_gain,
            "observed_gain_unit": "execution-accuracy fraction",
            "observed_interpretation": "better_than_no_op",
        },
        {
            "domain": "Spider clean", "clusters": 116,
            "proposal_precision": clean_precision,
            "proposal_precision_robust_all_cases": proposal["benefit_count"] / proposal["n"],
            "advantage_noise": 0.0, "observed_gain": clean_gain,
            "observed_gain_unit": "execution-accuracy fraction",
            "observed_interpretation": "indistinguishable_from_no_op",
        },
    ]


def interpolate(summary, clusters, precision, noise, metric):
    cs = sorted({int(r["cluster_count"]) for r in summary})
    ps = sorted({float(r["proposal_precision"]) for r in summary})
    ns = sorted({float(r["label_noise"]) for r in summary})
    clipped = {"clusters": min(max(clusters, cs[0]), cs[-1]),
               "precision": min(max(precision, ps[0]), ps[-1]),
               "noise": min(max(noise, ns[0]), ns[-1])}

    def bracket(values, value, log=False):
        if value <= values[0]: return values[0], values[0], 0.0
        if value >= values[-1]: return values[-1], values[-1], 0.0
        hi = next(v for v in values if v >= value); lo = values[values.index(hi)-1]
        transform = np.log if log else (lambda x: x)
        return lo, hi, float((transform(value)-transform(lo))/(transform(hi)-transform(lo)))

    c0,c1,wc = bracket(cs, clipped["clusters"], True)
    p0,p1,wp = bracket(ps, clipped["precision"])
    n0,n1,wn = bracket(ns, clipped["noise"])
    lookup={(int(r["cluster_count"]),float(r["proposal_precision"]),float(r["label_noise"])):float(r[metric]) for r in summary}
    value=0.0
    for c,w_c in [(c0,1-wc),(c1,wc)] if c0!=c1 else [(c0,1.0)]:
        for p,w_p in [(p0,1-wp),(p1,wp)] if p0!=p1 else [(p0,1.0)]:
            for n,w_n in [(n0,1-wn),(n1,wn)] if n0!=n1 else [(n0,1.0)]:
                value += w_c*w_p*w_n*lookup[(c,p,n)]
    return value, clipped


def enrich(domains, summary):
    for row in domains:
        predicted, clipped = interpolate(summary,row["clusters"],row["proposal_precision"],row["advantage_noise"],"learned_gain_mean")
        vs_noop, _ = interpolate(summary,row["clusters"],row["proposal_precision"],row["advantage_noise"],"learned_gain_mean")
        vs_best, _ = interpolate(summary,row["clusters"],row["proposal_precision"],row["advantage_noise"],"learned_minus_best_constant_mean")
        apply_all = 2.0 * clipped["precision"] - 1.0
        outside = any(abs(row[{"clusters":"clusters","precision":"proposal_precision","noise":"advantage_noise"}[k]]-v)>1e-12 for k,v in clipped.items())
        row.update({
            "synthetic_lookup_coordinate": clipped,
            "synthetic_extrapolation_required": outside,
            "clipped_phase_selective_gain": predicted,
            "clipped_phase_margin_vs_no_op": vs_noop,
            "clipped_phase_margin_vs_apply_all": predicted-apply_all,
            "phase_predicted_selective_gain": None if outside else predicted,
            "phase_predicted_beats_no_op": None if outside else vs_noop > 0,
            "phase_predicted_beats_apply_all": None if outside else predicted > apply_all,
            "phase_predicted_beats_best_constant": None if outside else vs_best > 0,
            "phase_prediction": "out_of_synthetic_support" if outside else ("selective region" if vs_noop>0 and predicted>apply_all else "global/no-op region"),
        })


def grid(summary, cluster_count, metric):
    ps=sorted({float(r["proposal_precision"]) for r in summary}); ns=sorted({float(r["label_noise"]) for r in summary})
    return ps,ns,np.asarray([[interpolate(summary,cluster_count,p,n,metric)[0] for p in ps] for n in ns])


def figure1(path, domains, summary):
    fig,axes=plt.subplots(1,3,figsize=(15.5,4.4),sharey=True,constrained_layout=True)
    for ax,row in zip(axes,domains):
        ps,ns,z=grid(summary,row["clusters"],"learned_minus_best_constant_mean")
        im=ax.imshow(100*z,origin="lower",aspect="auto",extent=[min(ps),max(ps),min(ns),max(ns)],cmap="RdBu_r",vmin=-8,vmax=8)
        x=row["proposal_precision"]; y=row["advantage_noise"]
        ax.scatter(x,y,s=130,c="gold",edgecolor="black",marker="*",zorder=5)
        ax.annotate(row["domain"]+(" (out of support)" if row["synthetic_extrapolation_required"] else ""),(x,y),xytext=(6,7),textcoords="offset points",fontsize=9,weight="bold")
        ax.set_title(f'{row["domain"]}: {row["clusters"]} clusters')
        ax.set_xlabel("Proposal precision"); ax.set_xlim(min(ps)-.02,max(ps)+.02); ax.set_ylim(min(ns)-.02,max(ns)+.02)
    axes[0].set_ylabel("Advantage sign-noise proxy")
    cb=fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.82,pad=.02); cb.set_label("Synthetic selective minus best global action (points)")
    fig.suptitle("Real-domain coordinates on the synthetic learnability phase")
    fig.savefig(path,dpi=220,bbox_inches="tight"); plt.close(fig)


def figure2(path, summary):
    clusters=sorted({int(r["cluster_count"]) for r in summary}); ps=sorted({float(r["proposal_precision"]) for r in summary}); ns=sorted({float(r["label_noise"]) for r in summary})
    fig,axes=plt.subplots(2,len(clusters),figsize=(3.6*len(clusters),7.2),sharex=True,sharey=True,constrained_layout=True)
    for j,c in enumerate(clusters):
        _,_,learned=grid(summary,c,"learned_gain_mean")
        apply=np.tile(2*np.asarray(ps)-1,(len(ns),1))
        p_noop=np.asarray([[interpolate(summary,c,p,n,"prob_learned_beats_no_op")[0] for p in ps] for n in ns])
        p_all=np.asarray([[interpolate(summary,c,p,n,"prob_learned_beats_apply_all")[0] for p in ps] for n in ns])
        matrices=[learned,learned-apply]; probabilities=[p_noop,p_all]
        for i,(matrix,probability,label) in enumerate(zip(matrices,probabilities,["Selective > no-op","Selective > apply-all"])):
            state=((matrix>0)&(probability>=.8)).astype(int)
            axes[i,j].imshow(state,origin="lower",aspect="auto",cmap=ListedColormap(["#d95f5f","#4daf7c"]),norm=BoundaryNorm([-.5,.5,1.5],2))
            if matrix.min() < 0 < matrix.max(): axes[i,j].contour(matrix,levels=[0],colors="black",linewidths=1)
            axes[i,j].set_title(f"{label}\n{c} clusters")
            axes[i,j].set_xticks(range(len(ps))); axes[i,j].set_xticklabels([f"{p:.1f}" for p in ps])
            axes[i,j].set_yticks(range(len(ns))); axes[i,j].set_yticklabels([f"{n:.1f}" for n in ns])
            axes[i,j].set_xlabel("Proposal precision")
        axes[0,j].set_ylabel("Label noise"); axes[1,j].set_ylabel("Label noise")
    fig.suptitle("Selective-correction reliable-win boundaries (green: mean margin > 0 and win probability >= 80%)")
    fig.savefig(path,dpi=220,bbox_inches="tight"); plt.close(fig)


def main():
    synthetic=read("synthetic_selective_phase.json"); summary=synthetic["summary"]
    domains=[xray_coordinate(),*spider_coordinates()]; enrich(domains,summary)
    result={"coordinate_definitions":{
        "clusters":"total independent clusters available to the CV protocol; X-ray fold train/validation/test counts are also reported",
        "proposal_precision":"beneficial / (beneficial + harmful) for the concrete apply-all proposal policy",
        "robust_precision":"beneficial / all cases, counting zero-advantage proposals as non-beneficial",
        "xray_noise":"pairwise cross-detector disagreement of realized advantage signs for the same patient",
        "spider_noise":"zero because execution reward is deterministic",
        "phase_mapping":"trilinear interpolation, log-linear in cluster count; out-of-grid coordinates are clipped and flagged",
    },"domains":domains}
    (OUT/"real_domain_phase_coordinates.json").write_text(json.dumps(result,indent=2)+"\n")
    flat=[]
    for r in domains:
        flat.append({k:r[k] for k in ["domain","clusters","proposal_precision","proposal_precision_robust_all_cases","advantage_noise","observed_gain","observed_gain_unit","phase_predicted_selective_gain","phase_predicted_beats_no_op","phase_predicted_beats_apply_all","phase_predicted_beats_best_constant","phase_prediction","synthetic_extrapolation_required","clipped_phase_selective_gain","clipped_phase_margin_vs_no_op","clipped_phase_margin_vs_apply_all"]})
    with (OUT/"real_domain_phase_coordinates.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    figure1(OUT/"figure1_real_domains_on_synthetic_phase.png",domains,summary)
    figure2(OUT/"figure2_selective_strategy_boundaries.png",summary)
    print(json.dumps(flat,indent=2))


if __name__ == "__main__": main()
