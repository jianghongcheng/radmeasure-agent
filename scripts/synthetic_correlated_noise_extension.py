#!/usr/bin/env python3
"""Correlated detector-noise extension around the X-ray operating point.

This is an independent extension: it reads no frozen phase artifact and writes
new files under a distinct prefix.  Two detector labels are generated from a
latent true advantage sign.  Their Bernoulli error indicators have marginal
error q and Pearson correlation rho.  Therefore

    disagreement = P(E1 != E2) = 2 q (1-q) (1-rho).

Negative error correlation represents case-dependent, opposing proposal signs
(one detector tends to be wrong exactly where the other is right).  Unlike
ordinary shared positive error, it permits disagreement above 0.5.  Detector 1
provides training labels; detector 2 is used only to measure disagreement.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def rho_bounds(q: float) -> tuple[float, float]:
    """Fréchet bounds for correlation of equal-marginal Bernoulli(q) errors."""
    lower = -min(q / (1.0 - q), (1.0 - q) / q)
    return lower, 1.0


def implied_disagreement(q: float, rho: float) -> float:
    lo, hi = rho_bounds(q)
    if not lo - 1e-12 <= rho <= hi + 1e-12:
        raise ValueError(f"rho={rho} infeasible for q={q}; bounds=[{lo}, {hi}]")
    return 2.0 * q * (1.0 - q) * (1.0 - rho)


def joint_errors(rng: np.random.Generator, n: int, q: float, rho: float) -> tuple[np.ndarray, np.ndarray]:
    """Sample paired Bernoulli errors with equal marginal q and correlation rho."""
    p11 = q * q + rho * q * (1.0 - q)
    p10 = q - p11
    p01 = q - p11
    p00 = 1.0 - p11 - p10 - p01
    probs = np.asarray([p00, p01, p10, p11])
    if probs.min() < -1e-10:
        raise ValueError(f"invalid joint probabilities {probs}")
    states = rng.choice(4, size=n, p=np.clip(probs, 0, 1) / probs.sum())
    return np.isin(states, [2, 3]), np.isin(states, [1, 3])


def intercept_for_mean(score: np.ndarray, target: float) -> float:
    lo, hi = -20.0, 20.0
    for _ in range(70):
        mid = (lo + hi) / 2
        mean = (1.0 / (1.0 + np.exp(-np.clip(score + mid, -30, 30)))).mean()
        if mean < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def one_run(q: float, rho: float, repeat: int, *, clusters: int, samples_per_cluster: int, precision: float, seed: int) -> dict:
    rng = np.random.default_rng(seed + repeat * 100_003 + round(q * 10_000) * 97 + round((rho + 1) * 10_000))
    n = clusters * samples_per_cluster
    cluster = np.repeat(np.arange(clusters), samples_per_cluster)
    cluster_effect = rng.normal(0, .75, clusters)
    x = rng.normal(size=n)
    nuisance = rng.normal(size=(n, 5))
    score = 1.35 * x + .55 * cluster_effect[cluster]
    intercept = intercept_for_mean(score, precision)
    truth = rng.random(n) < 1 / (1 + np.exp(-np.clip(score + intercept, -30, 30)))
    e1, e2 = joint_errors(rng, n, q, rho)
    label1, label2 = np.logical_xor(truth, e1), np.logical_xor(truth, e2)
    features = np.column_stack([x + rng.normal(0, .65, n), nuisance])

    ids = rng.permutation(clusters)
    nt, nc = round(.6 * clusters), round(.2 * clusters)
    train_ids, cal_ids, test_ids = ids[:nt], ids[nt:nt+nc], ids[nt+nc:]
    train, cal, test = (np.isin(cluster, z) for z in (train_ids, cal_ids, test_ids))
    scaler = StandardScaler().fit(features[train])
    model = LogisticRegression(C=1.0, max_iter=500, random_state=seed).fit(scaler.transform(features[train]), label1[train].astype(int))
    cal_score = model.predict_proba(scaler.transform(features[cal]))[:, 1]
    test_score = model.predict_proba(scaler.transform(features[test]))[:, 1]
    cal_adv = np.where(truth[cal], 1.0, -1.0)
    test_adv = np.where(truth[test], 1.0, -1.0)
    thresholds = np.unique(np.r_[0, cal_score, 1 + 1e-9])
    cal_gain = np.asarray([np.mean(cal_adv * (cal_score >= t)) for t in thresholds])
    threshold = thresholds[np.flatnonzero(np.isclose(cal_gain, cal_gain.max()))[-1]]
    learned = float(np.mean(test_adv * (test_score >= threshold)))
    apply_all = float(test_adv.mean())
    best_constant = max(0.0, apply_all if cal_adv.mean() > 0 else 0.0)
    return {
        "q_marginal_label_error": q, "rho_error_correlation": rho,
        "theoretical_disagreement": implied_disagreement(q, rho), "repeat": repeat,
        "clusters": clusters, "proposal_precision": precision,
        "realized_disagreement": float(np.mean(label1[test] != label2[test])),
        "learned_gain": learned, "apply_all_gain": apply_all,
        "no_op_gain": 0.0, "best_tuned_constant_gain": best_constant,
        "learned_minus_best_constant": learned - best_constant,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/research")
    a = p.parse_args()
    qs = [.35, .375, .40, .425, .45, .475, .50]
    rhos = [-.50, -.35, -.20, 0.0, .20]
    rows = []
    for q in qs:
        for rho in rhos:
            if rho >= rho_bounds(q)[0] - 1e-12:
                rows += [one_run(q, rho, r, clusters=176, samples_per_cluster=1, precision=.40, seed=20260825) for r in range(a.repeats)]
    keys = sorted({(r["q_marginal_label_error"], r["rho_error_correlation"]) for r in rows})
    summary = []
    for q, rho in keys:
        rr = [r for r in rows if r["q_marginal_label_error"] == q and r["rho_error_correlation"] == rho]
        item = {"q_marginal_label_error": q, "rho_error_correlation": rho,
                "theoretical_disagreement": implied_disagreement(q, rho), "repeats": len(rr)}
        for m in ["realized_disagreement", "learned_gain", "apply_all_gain", "learned_minus_best_constant"]:
            x = np.asarray([r[m] for r in rr]); item[m+"_mean"] = float(x.mean()); item[m+"_ci_low"] = float(np.quantile(x,.025)); item[m+"_ci_high"] = float(np.quantile(x,.975))
        item["prob_beats_best_constant"] = float(np.mean([r["learned_minus_best_constant"] > 0 for r in rr]))
        summary.append(item)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "synthetic_correlated_noise_xray"
    write_csv(a.output_dir/f"{prefix}_raw.csv", rows); write_csv(a.output_dir/f"{prefix}_summary.csv", summary)
    payload = {"design": {"clusters":176,"samples_per_cluster":1,"proposal_precision":.40,"q_grid":qs,"rho_grid":rhos,"repeats":a.repeats,
        "model":"paired Bernoulli detector errors; D=2q(1-q)(1-rho)","observed_xray_disagreement":[.476,.622]}, "summary":summary}
    (a.output_dir/f"{prefix}.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8.2,5.2))
    for rho in rhos:
        ss=[x for x in summary if x["rho_error_correlation"]==rho]
        ax.plot([x["theoretical_disagreement"] for x in ss],[100*x["learned_minus_best_constant_mean"] for x in ss],marker="o",label=fr"$\rho={rho:+.2f}$")
    ax.axhline(0,color="black",lw=1); ax.axvline(.476,color="#377eb8",ls="--",label="X-ray primary D=.476"); ax.axvline(.622,color="#e41a1c",ls="--",label="X-ray robust D=.622")
    ax.set(xlabel="Pairwise detector disagreement D",ylabel="Learned minus best constant (gain points)",title="X-ray neighborhood under correlated detector-label errors")
    ax.legend(ncol=2,fontsize=8); fig.tight_layout(); fig.savefig(a.output_dir/f"{prefix}.png",dpi=220); plt.close(fig)
    print(json.dumps({"runs":len(rows),"conditions":len(summary),"output":str(a.output_dir/f'{prefix}.json')},indent=2))


if __name__ == "__main__": main()
