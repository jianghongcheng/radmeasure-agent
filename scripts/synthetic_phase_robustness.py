#!/usr/bin/env python3
"""Preregistered robustness analyses for the selective-correction phase."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "research"


def load_base_phase():
    path = ROOT / "scripts" / "synthetic_selective_correction_phase.py"
    spec = importlib.util.spec_from_file_location("base_phase", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base_phase()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def intercept_for_mean(score: np.ndarray, target: float) -> float:
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if sigmoid(score + mid).mean() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def fit_scores(x_train, y_train, x_cal, x_test, seed):
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_cal = scaler.transform(x_cal)
    x_test = scaler.transform(x_test)
    if np.unique(y_train).size < 2:
        value = float(y_train[0])
        return np.full(len(x_cal), value), np.full(len(x_test), value)
    model = LogisticRegression(C=1.0, max_iter=300, random_state=seed)
    model.fit(x_train, y_train)
    return model.predict_proba(x_cal)[:, 1], model.predict_proba(x_test)[:, 1]


def best_threshold(scores: np.ndarray, advantages: np.ndarray) -> float:
    order = np.argsort(-scores, kind="mergesort")
    sorted_adv = advantages[order]
    gains = np.r_[0.0, np.cumsum(sorted_adv)] / len(advantages)
    best = gains.max()
    # The smallest selected prefix among ties is the conservative action.
    selected_n = int(np.flatnonzero(np.isclose(gains, best))[0])
    if selected_n == 0:
        return float(np.nextafter(scores.max(), np.inf))
    return float(scores[order[selected_n - 1]])


def seed_stability() -> dict:
    seeds = list(range(20260824, 20260834))
    conditions = [(0.5, 0.2, 30), (0.5, 0.2, 100)]
    def evaluate_seed(precision, noise, clusters, seed):
            values = [BASE.evaluate_condition(
                BASE.Condition(precision, noise, clusters, repeat),
                samples_per_cluster=40,
                seed=seed,
            ) for repeat in range(40)]
            margins = np.array([v["learned_minus_best_constant"] for v in values])
            return {
                "proposal_precision": precision,
                "label_noise": noise,
                "cluster_count": clusters,
                "simulator_seed": seed,
                "mean_margin": float(margins.mean()),
                "win_probability": float((margins > 0).mean()),
                "reliable": bool(margins.mean() > 0 and (margins > 0).mean() >= 0.8),
            }
    rows = Parallel(n_jobs=8, prefer="threads")(delayed(evaluate_seed)(precision, noise, clusters, seed)
                              for precision, noise, clusters in conditions for seed in seeds)
    summary = []
    for clusters in (30, 100):
        group = [r for r in rows if r["cluster_count"] == clusters]
        margins = np.array([r["mean_margin"] for r in group])
        wins = np.array([r["win_probability"] for r in group])
        summary.append({
            "proposal_precision": 0.5,
            "label_noise": 0.2,
            "cluster_count": clusters,
            "mean_margin_across_seeds": float(margins.mean()),
            "sd_margin_across_seeds": float(margins.std(ddof=1)),
            "mean_win_probability": float(wins.mean()),
            "sd_win_probability": float(wins.std(ddof=1)),
            "reliable_seed_fraction": float(np.mean([r["reliable"] for r in group])),
        })
    return {"rows": rows, "summary": summary}


def magnitude_condition(precision, noise, clusters, repeat, definition, seed=20260824):
    rng = np.random.default_rng(seed + 100_003 * repeat + 101 * clusters)
    n = clusters * 40
    cluster = np.repeat(np.arange(clusters), 40)
    cluster_effect = rng.normal(0, 0.75, clusters)
    signal = rng.normal(size=n)
    nuisance = rng.normal(size=(n, 5))
    score = 1.35 * signal + 0.55 * cluster_effect[cluster]
    intercept = intercept_for_mean(score, precision)
    beneficial = rng.random(n) < sigmoid(score + intercept)
    magnitude = np.clip(np.abs(rng.normal(0.75 + 0.35 * np.abs(signal), 0.35)), 0.05, None)
    advantage = np.where(beneficial, magnitude, -magnitude)
    features = np.column_stack([signal + rng.normal(0, 0.65, n), nuisance])
    train_ids, cal_ids, test_ids = BASE.cluster_split(clusters, seed, repeat)
    train = np.isin(cluster, train_ids); cal = np.isin(cluster, cal_ids); test = np.isin(cluster, test_ids)
    if definition == "strict":
        train_positive = advantage[train] > 0
    elif definition == "meaningful":
        train_positive = advantage[train] > 0.5
    elif definition == "rank":
        cutoff = np.quantile(advantage[train], 0.8)
        train_positive = advantage[train] >= cutoff
    else:
        raise ValueError(definition)
    flips = rng.random(train.sum()) < noise
    train_positive = np.logical_xor(train_positive, flips).astype(int)
    cal_score, test_score = fit_scores(features[train], train_positive, features[cal], features[test], seed)
    threshold = best_threshold(cal_score, advantage[cal])
    learned = float(np.mean(advantage[test] * (test_score >= threshold)))
    apply_all = float(advantage[test].mean())
    best_global = max(0.0, apply_all if advantage[cal].mean() > 0 else 0.0)
    return {
        "proposal_precision": precision, "label_noise": noise, "cluster_count": clusters,
        "repeat": repeat, "definition": definition,
        "effective_training_positive_rate": float(train_positive.mean()),
        "learned_gain": learned, "apply_all_gain": apply_all,
        "learned_minus_best_global": learned - best_global,
        "beats_no_op": learned > 0, "beats_apply_all": learned > apply_all,
    }


def definition_sensitivity() -> dict:
    rows = Parallel(n_jobs=8, prefer="threads", verbose=5)(delayed(magnitude_condition)(p, n, c, r, d)
            for p in (0.3, 0.4, 0.5, 0.6, 0.7)
            for n in (0.0, 0.2, 0.4)
            for c in (30, 100)
            for d in ("strict", "meaningful", "rank")
            for r in range(40))
    summary = []
    reliable_sets = {}
    for d in ("strict", "meaningful", "rank"):
        reliable_sets[d] = set()
        for p in (0.3, 0.4, 0.5, 0.6, 0.7):
            for n in (0.0, 0.2, 0.4):
                for c in (30, 100):
                    g = [x for x in rows if (x["definition"],x["proposal_precision"],x["label_noise"],x["cluster_count"]) == (d,p,n,c)]
                    margins = np.array([x["learned_minus_best_global"] for x in g])
                    no_op = np.mean([x["beats_no_op"] for x in g])
                    apply = np.mean([x["beats_apply_all"] for x in g])
                    reliable = margins.mean() > 0 and no_op >= .8 and apply >= .8
                    if reliable: reliable_sets[d].add((p,n,c))
                    summary.append({"definition":d,"proposal_precision":p,"label_noise":n,"cluster_count":c,
                        "mean_margin":float(margins.mean()),"prob_beats_no_op":float(no_op),
                        "prob_beats_apply_all":float(apply),"reliable":bool(reliable),
                        "effective_training_positive_rate":float(np.mean([x["effective_training_positive_rate"] for x in g]))})
    comparisons=[]
    for a,b in (("strict","meaningful"),("strict","rank"),("meaningful","rank")):
        union=reliable_sets[a]|reliable_sets[b]; inter=reliable_sets[a]&reliable_sets[b]
        directions=[(next(x for x in summary if (x["definition"],x["proposal_precision"],x["label_noise"],x["cluster_count"])==(a,*cell))["mean_margin"]>0)==
                    (next(x for x in summary if (x["definition"],x["proposal_precision"],x["label_noise"],x["cluster_count"])==(b,*cell))["mean_margin"]>0)
                    for cell in sorted({(x["proposal_precision"],x["label_noise"],x["cluster_count"]) for x in summary})]
        comparisons.append({"a":a,"b":b,"reliable_region_jaccard":float(len(inter)/len(union)) if union else 1.0,
                            "directional_agreement":float(np.mean(directions)),"reliable_cells_a":len(reliable_sets[a]),"reliable_cells_b":len(reliable_sets[b])})
    return {"rows": rows, "summary": summary, "comparisons": comparisons}


def candidate_condition(precision, noise, clusters, count, dependence, repeat, seed=20260824):
    rng=np.random.default_rng(seed+100_003*repeat+101*clusters+17*count)
    cases=clusters*40; candidate_n=cases*count
    cluster=np.repeat(np.arange(clusters),40); row_cluster=np.repeat(cluster,count)
    case_signal=rng.normal(size=cases); candidate_signal=rng.normal(size=candidate_n)
    shared=np.repeat(rng.normal(0,0.9,size=cases),count) if dependence=="shared_case" else np.zeros(candidate_n)
    latent=1.05*candidate_signal+0.55*np.repeat(case_signal,count)+shared
    intercept=intercept_for_mean(latent,precision)
    beneficial=rng.random(candidate_n)<sigmoid(latent+intercept)
    advantage=np.where(beneficial,1.0,-1.0)
    features=np.column_stack([candidate_signal+rng.normal(0,.65,candidate_n),np.repeat(case_signal,count),rng.normal(size=(candidate_n,4))])
    flip=rng.random(candidate_n)<noise; noisy=np.logical_xor(beneficial,flip).astype(int)
    train_ids,cal_ids,test_ids=BASE.cluster_split(clusters,seed,repeat)
    train=np.isin(row_cluster,train_ids); cal=np.isin(row_cluster,cal_ids); test=np.isin(row_cluster,test_ids)
    cal_score,test_score=fit_scores(features[train],noisy[train],features[cal],features[test],seed)
    def collapse(mask,scores):
        a=advantage[mask].reshape(-1,count); s=scores.reshape(-1,count); j=s.argmax(1)
        return s[np.arange(len(j)),j],a[np.arange(len(j)),j]
    cal_best_score,cal_best_adv=collapse(cal,cal_score); test_best_score,test_best_adv=collapse(test,test_score)
    threshold=best_threshold(cal_best_score,cal_best_adv)
    learned=float(np.mean(test_best_adv*(test_best_score>=threshold))); apply=float(test_best_adv.mean())
    return {"proposal_precision":precision,"label_noise":noise,"cluster_count":clusters,"candidate_count":count,
            "dependence":dependence,"repeat":repeat,"learned_gain":learned,"apply_all_gain":apply,
            "margin_over_no_op":learned,"margin_over_apply_all":learned-apply,
            "beats_no_op":learned>0,"beats_apply_all":learned>apply}


def candidate_distribution() -> dict:
    rows=Parallel(n_jobs=8, prefer="threads", verbose=5)(delayed(candidate_condition)(p,n,c,k,d,r)
          for p in (.3,.5,.7) for n in (0,.2,.4) for c in (30,100)
          for k in (2,5,10,20) for d in ("independent","shared_case") for r in range(40))
    summary=[]
    for p in (.3,.5,.7):
      for n in (0,.2,.4):
       for c in (30,100):
        for k in (2,5,10,20):
         for d in ("independent","shared_case"):
          g=[x for x in rows if (x["proposal_precision"],x["label_noise"],x["cluster_count"],x["candidate_count"],x["dependence"])==(p,n,c,k,d)]
          no=np.mean([x["beats_no_op"] for x in g]); app=np.mean([x["beats_apply_all"] for x in g])
          summary.append({"proposal_precision":p,"label_noise":n,"cluster_count":c,"candidate_count":k,"dependence":d,
                          "learned_gain_mean":float(np.mean([x["learned_gain"] for x in g])),
                          "margin_over_no_op_mean":float(np.mean([x["margin_over_no_op"] for x in g])),
                          "margin_over_apply_all_mean":float(np.mean([x["margin_over_apply_all"] for x in g])),
                          "prob_beats_no_op":float(no),"prob_beats_apply_all":float(app),
                          "reliable":bool(no>=.8 and app>=.8 and np.mean([x["learned_gain"] for x in g])>0)})
    return {"rows":rows,"summary":summary}


def write_csv(path: Path, rows: list[dict]):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def plot_seed(result):
    fig,ax=plt.subplots(figsize=(5.8,3.8))
    for s in result["summary"]:
        ax.errorbar(s["cluster_count"],100*s["mean_win_probability"],yerr=100*s["sd_win_probability"],fmt="o",capsize=5,label=f'{s["cluster_count"]} clusters')
    ax.axhline(80,color="black",ls="--",lw=1,label="reliable-win rule")
    ax.set(xlabel="Independent clusters",ylabel="Selective win probability (%)",ylim=(0,105),title="Boundary repeatability across 10 simulator seeds")
    ax.legend(frameon=False);fig.tight_layout();fig.savefig(OUT/"synthetic_phase_seed_stability.png",dpi=220);plt.close(fig)


def plot_candidates(result):
    fig,axes=plt.subplots(1,2,figsize=(9,3.7),sharey=True)
    subset=[x for x in result["summary"] if x["proposal_precision"]==.5 and x["label_noise"]==.2 and x["cluster_count"]==100]
    for ax,dep in zip(axes,("independent","shared_case")):
        g=[x for x in subset if x["dependence"]==dep]
        ax.plot([x["candidate_count"] for x in g],[100*x["margin_over_no_op_mean"] for x in g],"o-",label="vs no-op")
        ax.plot([x["candidate_count"] for x in g],[100*x["margin_over_apply_all_mean"] for x in g],"s-",label="vs apply-all")
        ax.axhline(0,color="black",lw=.8);ax.set_title(dep.replace("_"," "));ax.set_xlabel("Candidates per case")
    axes[0].set_ylabel("Selective margin (gain points)");axes[1].legend(frameon=False);fig.tight_layout();fig.savefig(OUT/"synthetic_candidate_distribution_ablation.png",dpi=220);plt.close(fig)


def main():
    seed=seed_stability(); definitions=definition_sensitivity(); candidates=candidate_distribution()
    payload={"protocol_sha256":"16345da72142c07fbd8c86d4109f4bb27dd282c10b9eaf274749541c3bc4b960",
             "seed_stability":seed,"advantage_definition_sensitivity":definitions,"candidate_distribution":candidates}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"synthetic_phase_robustness_v1.json").write_text(json.dumps(payload,indent=2)+"\n")
    write_csv(OUT/"synthetic_phase_seed_stability.csv",seed["rows"])
    write_csv(OUT/"synthetic_advantage_definition_sensitivity.csv",definitions["summary"])
    write_csv(OUT/"synthetic_candidate_distribution_ablation.csv",candidates["summary"])
    plot_seed(seed);plot_candidates(candidates)
    print(json.dumps({"seed_stability":seed["summary"],"definition_comparisons":definitions["comparisons"],
                      "candidate_reliable_cells":sum(x["reliable"] for x in candidates["summary"]),
                      "candidate_total_cells":len(candidates["summary"])},indent=2))


if __name__ == "__main__":
    main()
