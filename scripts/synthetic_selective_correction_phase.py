#!/usr/bin/env python3
"""Synthetic phase diagram for selective correction.

The experiment independently controls proposal precision, advantage-label noise,
and the number of independent clusters.  Cluster-disjoint train/calibration/test
splits compare a learned selective policy with no-op and a calibration-tuned
constant action (apply every proposal or apply none).
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Condition:
    proposal_precision: float
    label_noise: float
    cluster_count: int
    repeat: int


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _intercept_for_mean(score: np.ndarray, target: float) -> float:
    lo, hi = -20.0, 20.0
    for _ in range(70):
        mid = (lo + hi) / 2.0
        if float(_sigmoid(score + mid).mean()) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def generate_condition(
    condition: Condition,
    *,
    samples_per_cluster: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Generate clustered proposals with controlled marginal precision.

    Labels are +1 for a beneficial edit and -1 for a harmful edit.  Features
    contain a stable signal plus cluster-specific nuisance variation.  Label
    noise flips training supervision only; true counterfactual advantages are
    retained for calibration and held-out evaluation.
    """
    rng = np.random.default_rng(seed + 100_003 * condition.repeat)
    k = condition.cluster_count
    n = k * samples_per_cluster
    cluster = np.repeat(np.arange(k), samples_per_cluster)
    cluster_effect = rng.normal(0.0, 0.75, size=k)
    x_signal = rng.normal(size=n)
    nuisance = rng.normal(size=(n, 5))
    score = 1.35 * x_signal + 0.55 * cluster_effect[cluster]
    intercept = _intercept_for_mean(score, condition.proposal_precision)
    beneficial = rng.binomial(1, _sigmoid(score + intercept)).astype(bool)
    advantage = np.where(beneficial, 1.0, -1.0)

    # The learner observes a noisy proxy of the stable signal and nuisance
    # dimensions, not cluster identity or the true advantage.
    features = np.column_stack(
        [x_signal + rng.normal(0.0, 0.65, size=n), nuisance]
    )
    noisy_beneficial = beneficial.copy()
    flips = rng.random(n) < condition.label_noise
    noisy_beneficial[flips] = ~noisy_beneficial[flips]
    return {
        "cluster": cluster,
        "features": features,
        "beneficial": beneficial,
        "noisy_beneficial": noisy_beneficial,
        "advantage": advantage,
    }


def cluster_split(cluster_count: int, seed: int, repeat: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed + 7_919 * repeat + cluster_count)
    ids = rng.permutation(cluster_count)
    n_train = max(2, int(round(0.6 * cluster_count)))
    n_cal = max(1, int(round(0.2 * cluster_count)))
    if n_train + n_cal >= cluster_count:
        n_train = cluster_count - 2
        n_cal = 1
    return ids[:n_train], ids[n_train : n_train + n_cal], ids[n_train + n_cal :]


def evaluate_condition(
    condition: Condition,
    *,
    samples_per_cluster: int = 40,
    seed: int = 20260824,
) -> dict[str, float | int]:
    data = generate_condition(condition, samples_per_cluster=samples_per_cluster, seed=seed)
    train_ids, cal_ids, test_ids = cluster_split(condition.cluster_count, seed, condition.repeat)
    masks = [np.isin(data["cluster"], ids) for ids in (train_ids, cal_ids, test_ids)]
    train, cal, test = masks
    scaler = StandardScaler().fit(data["features"][train])
    x_train = scaler.transform(data["features"][train])
    x_cal = scaler.transform(data["features"][cal])
    x_test = scaler.transform(data["features"][test])
    y_train = data["noisy_beneficial"][train].astype(int)

    # Degenerate high-noise/small-cluster replicates get a constant score.
    if np.unique(y_train).size < 2:
        cal_score = np.full(cal.sum(), float(y_train[0]))
        test_score = np.full(test.sum(), float(y_train[0]))
    else:
        model = LogisticRegression(C=1.0, max_iter=500, random_state=seed)
        model.fit(x_train, y_train)
        cal_score = model.predict_proba(x_cal)[:, 1]
        test_score = model.predict_proba(x_test)[:, 1]

    cal_adv = data["advantage"][cal]
    test_adv = data["advantage"][test]
    thresholds = np.unique(np.r_[0.0, cal_score, 1.0 + 1e-9])
    cal_gains = np.array([np.mean(cal_adv * (cal_score >= t)) for t in thresholds])
    # Conservative tie-breaking prefers less intervention.
    best_gain = cal_gains.max()
    best_t = float(thresholds[np.flatnonzero(np.isclose(cal_gains, best_gain))[-1]])
    selected = test_score >= best_t
    learned_gain = float(np.mean(test_adv * selected))

    apply_all_cal_gain = float(cal_adv.mean())
    tuned_apply_all = apply_all_cal_gain > 0.0
    tuned_gain = float(test_adv.mean()) if tuned_apply_all else 0.0
    apply_all_gain = float(test_adv.mean())
    no_op_gain = 0.0
    best_constant_gain = max(tuned_gain, no_op_gain)
    selected_adv = test_adv[selected]
    return {
        **asdict(condition),
        "train_clusters": int(len(train_ids)),
        "calibration_clusters": int(len(cal_ids)),
        "test_clusters": int(len(test_ids)),
        "test_examples": int(test.sum()),
        "realized_proposal_precision": float(data["beneficial"][test].mean()),
        "learned_gain": learned_gain,
        "tuned_constant_gain": tuned_gain,
        "apply_all_gain": apply_all_gain,
        "no_op_gain": no_op_gain,
        "best_constant_gain": best_constant_gain,
        "learned_minus_best_constant": learned_gain - best_constant_gain,
        "learned_coverage": float(selected.mean()),
        "learned_benefit_rate": float(np.mean(selected_adv > 0)) if selected.any() else 0.0,
        "learned_harm_rate": float(np.mean(selected_adv < 0)) if selected.any() else 0.0,
        "tuned_action": "apply_all" if tuned_apply_all else "stop",
        "threshold": best_t,
    }


def aggregate(rows: list[dict[str, float | int]]) -> list[dict[str, float | int]]:
    keys = ("proposal_precision", "label_noise", "cluster_count")
    groups: dict[tuple[float, float, int], list[dict[str, float | int]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)  # type: ignore[arg-type]
    output = []
    metrics = (
        "realized_proposal_precision",
        "learned_gain",
        "tuned_constant_gain",
        "best_constant_gain",
        "learned_minus_best_constant",
        "learned_coverage",
    )
    for condition, values in sorted(groups.items()):
        item: dict[str, float | int] = dict(zip(keys, condition))
        item["repeats"] = len(values)
        for metric in metrics:
            array = np.asarray([float(v[metric]) for v in values])
            item[f"{metric}_mean"] = float(array.mean())
            item[f"{metric}_ci_low"] = float(np.quantile(array, 0.025))
            item[f"{metric}_ci_high"] = float(np.quantile(array, 0.975))
        differences = np.asarray([float(v["learned_minus_best_constant"]) for v in values])
        item["prob_learned_beats_best_constant"] = float(np.mean(differences > 0.0))
        item["prob_learned_beats_no_op"] = float(
            np.mean([float(v["learned_gain"]) > 0.0 for v in values])
        )
        item["prob_learned_beats_apply_all"] = float(
            np.mean([float(v["learned_gain"]) > float(v["apply_all_gain"]) for v in values])
        )
        output.append(item)
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_phase(path: Path, summary: list[dict[str, float | int]]) -> None:
    clusters = sorted({int(row["cluster_count"]) for row in summary})
    precisions = sorted({float(row["proposal_precision"]) for row in summary})
    noises = sorted({float(row["label_noise"]) for row in summary})
    fig, axes = plt.subplots(1, len(clusters), figsize=(4.4 * len(clusters), 4.2), squeeze=False)
    for ax, count in zip(axes[0], clusters):
        matrix = np.full((len(noises), len(precisions)), np.nan)
        for row in summary:
            if int(row["cluster_count"]) == count:
                i = noises.index(float(row["label_noise"]))
                j = precisions.index(float(row["proposal_precision"]))
                matrix[i, j] = 100.0 * float(row["learned_minus_best_constant_mean"])
        image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-8, vmax=8)
        ax.set_title(f"{count} clusters")
        ax.set_xticks(range(len(precisions)), [f"{p:.2f}" for p in precisions])
        ax.set_yticks(range(len(noises)), [f"{n:.2f}" for n in noises])
        ax.set_xlabel("Proposal precision")
        ax.set_ylabel("Advantage-label flip probability")
        for i in range(len(noises)):
            for j in range(len(precisions)):
                ax.text(j, i, f"{matrix[i, j]:+.1f}", ha="center", va="center", fontsize=8)
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.83)
    colorbar.set_label("Learned minus best tuned constant (gain points)")
    fig.suptitle("When does learned selective correction beat a global action?", y=1.02)
    fig.subplots_adjust(left=0.07, right=0.91, bottom=0.16, top=0.86, wspace=0.33)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def boundary_rows(summary: list[dict[str, float | int]]) -> list[dict[str, float | int | str]]:
    output = []
    pairs = sorted({(int(r["cluster_count"]), float(r["label_noise"])) for r in summary})
    for cluster_count, noise in pairs:
        candidates = sorted(
            (r for r in summary if int(r["cluster_count"]) == cluster_count and float(r["label_noise"]) == noise),
            key=lambda r: float(r["proposal_precision"]),
        )
        reliable = [
            r for r in candidates
            if float(r["learned_minus_best_constant_mean"]) > 0
            and float(r["prob_learned_beats_best_constant"]) >= 0.8
        ]
        reliable_precisions = [float(r["proposal_precision"]) for r in reliable]
        output.append({
            "cluster_count": cluster_count,
            "label_noise": noise,
            "reliable_precision_values": (
                ";".join(f"{value:.2f}" for value in reliable_precisions)
                if reliable_precisions else "not_observed"
            ),
            "minimum_reliable_precision": min(reliable_precisions) if reliable_precisions else "not_observed",
            "maximum_reliable_precision": max(reliable_precisions) if reliable_precisions else "not_observed",
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--samples-per-cluster", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "research")
    parser.add_argument("--prefix", default="synthetic_selective_phase")
    parser.add_argument("--precisions", default="0.30,0.40,0.50,0.60,0.70,0.80")
    parser.add_argument("--noises", default="0.00,0.10,0.20,0.30,0.40")
    parser.add_argument("--cluster-counts", default="10,30,100,300")
    args = parser.parse_args()
    precisions = tuple(float(value) for value in args.precisions.split(","))
    noises = tuple(float(value) for value in args.noises.split(","))
    cluster_counts = tuple(int(value) for value in args.cluster_counts.split(","))
    rows = []
    for count in cluster_counts:
        for noise in noises:
            for precision in precisions:
                for repeat in range(args.repeats):
                    rows.append(evaluate_condition(
                        Condition(precision, noise, count, repeat),
                        samples_per_cluster=args.samples_per_cluster,
                        seed=args.seed,
                    ))
    summary = aggregate(rows)
    boundaries = boundary_rows(summary)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / f"{args.prefix}_raw.csv", rows)
    write_csv(out / f"{args.prefix}_summary.csv", summary)
    write_csv(out / f"{args.prefix}_boundaries.csv", boundaries)
    payload = {
        "design": {
            "seed": args.seed,
            "repeats": args.repeats,
            "samples_per_cluster": args.samples_per_cluster,
            "proposal_precisions": precisions,
            "advantage_label_flip_probabilities": noises,
            "cluster_counts": cluster_counts,
            "split": "60% train clusters / 20% calibration clusters / 20% test clusters",
            "true_advantage": "+1 beneficial, -1 harmful",
            "tuned_constant": "calibration chooses apply-all or no-op",
        },
        "summary": summary,
        "boundaries": boundaries,
    }
    (out / f"{args.prefix}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_phase(out / f"{args.prefix}.png", summary)
    print(json.dumps({"conditions": len(summary), "runs": len(rows), "boundaries": boundaries}, indent=2))


if __name__ == "__main__":
    main()
