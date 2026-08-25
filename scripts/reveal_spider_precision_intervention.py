#!/usr/bin/env python3
"""Run the preregistered Spider matched-switch precision intervention."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]


def load_script(name, filename):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def fold_of(db_id, folds, seed):
    return int(hashlib.sha256(f"precision-reveal:{seed}:{db_id}".encode()).hexdigest()[:8], 16) % folds


def metric(cases):
    n = len(cases)
    benefit = sum(not c["before_correct"] and c["after_correct"] for c in cases)
    harm = sum(c["before_correct"] and not c["after_correct"] for c in cases)
    return {
        "n": n,
        "accuracy": sum(c["after_correct"] for c in cases) / n,
        "gain": (benefit - harm) / n,
        "benefit_count": benefit,
        "harm_count": harm,
        "coverage": sum(c["edited"] for c in cases) / n,
        "cases": cases,
    }


def evaluate(records, scores, risk_weight, threshold, mode):
    cases = []
    for record in records:
        candidate = record["candidates"][0]
        if mode == "no_op":
            edited = False
        elif mode == "apply_all":
            edited = True
        else:
            utility = float(
                scores[record["index"]]["benefit"][0]
                - risk_weight * scores[record["index"]]["harm"][0]
            )
            edited = utility > threshold
        after = candidate["correct"] if edited else record["base_correct"]
        cases.append({
            "index": record["index"],
            "db_id": record["db_id"],
            "before_correct": record["base_correct"],
            "after_correct": after,
            "edited": edited,
        })
    return metric(cases)


def calibrate(records, scores):
    values = []
    for risk_weight in [0, .25, .5, 1, 2, 4, 8]:
        utility = np.asarray([
            scores[r["index"]]["benefit"][0]
            - risk_weight * scores[r["index"]]["harm"][0]
            for r in records
        ])
        thresholds = np.unique(np.r_[-np.inf, 0, np.quantile(utility, np.linspace(0, 1, 41)), np.inf])
        for threshold in thresholds:
            result = evaluate(records, scores, risk_weight, float(threshold), "learned")
            values.append({
                "risk_weight": risk_weight,
                "threshold": float(threshold),
                "accuracy": result["accuracy"],
                "harm_count": result["harm_count"],
                "coverage": result["coverage"],
            })
    return max(values, key=lambda row: (row["accuracy"], -row["harm_count"], -row["coverage"]))


def cluster_bootstrap_difference(left_cases, right_cases, seed, repeats=4000):
    left = {row["index"]: row for row in left_cases}
    right = {row["index"]: row for row in right_cases}
    assert left.keys() == right.keys()
    by_db = {}
    for index, row in left.items():
        by_db.setdefault(row["db_id"], []).append(index)
    dbs = sorted(by_db)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repeats):
        sampled = rng.choice(dbs, len(dbs), replace=True)
        numerator = denominator = 0
        for db in sampled:
            for index in by_db[db]:
                l = int(left[index]["after_correct"])
                r = int(right[index]["after_correct"])
                numerator += l - r
                denominator += 1
        draws.append(numerator / denominator)
    return {
        "mean": float(np.mean(draws)),
        "ci_low": float(np.quantile(draws, .025)),
        "ci_high": float(np.quantile(draws, .975)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path,
        default=ROOT / "outputs/research/spider_clean_base_executable_edits.json"
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "outputs/research/spider_precision_intervention_manifest.json"
    )
    parser.add_argument(
        "--preregistration", type=Path,
        default=ROOT / "outputs/research/spider_precision_intervention_preregistered.json"
    )
    parser.add_argument(
        "--phase", type=Path,
        default=ROOT / "outputs/research/synthetic_selective_phase_extended_v1.json"
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs/research/spider_precision_intervention_revealed.json"
    )
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    source_bytes = args.source.read_bytes()
    manifest_bytes = args.manifest.read_bytes()
    prereg_bytes = args.preregistration.read_bytes()
    phase_bytes = args.phase.read_bytes()
    prereg = json.loads(prereg_bytes)
    manifest = json.loads(manifest_bytes)
    assert prereg["status"].startswith("PREREGISTERED")
    assert prereg["frozen_inputs"]["source_sha256"] == sha256(source_bytes)
    assert prereg["frozen_inputs"]["manifest_sha256"] == sha256(manifest_bytes)
    assert prereg["frozen_inputs"]["phase_sha256"] == sha256(phase_bytes)

    source = json.loads(source_bytes)["records"]
    by_index = {r["index"]: r for r in source}
    pair_by_index = {
        row["index"]: row
        for row in manifest["benefit_pairs"] + manifest["harm_pairs"]
    }
    selector = load_script("precision_selector", "spider_advantage_selector_cv.py")
    targets = prereg["design"]["target_precisions"]
    results = {}

    for target in targets:
        allocation = manifest["allocations"][str(target)]
        active = set(allocation["benefit_active_indices"] + allocation["harm_active_indices"])
        records = []
        for index, pair in pair_by_index.items():
            original = by_index[index]
            candidate_index = pair["signed_candidate_index"] if index in active else pair["neutral_candidate_index"]
            row = {k: v for k, v in original.items() if k != "candidates"}
            row["candidates"] = [original["candidates"][candidate_index]]
            records.append(row)
        records.sort(key=lambda row: row["index"])
        folds = []
        seed = prereg["design"]["database_split_seed"]
        for outer in range(args.folds):
            test = [r for r in records if fold_of(r["db_id"], args.folds, seed) == outer]
            calibration = [r for r in records if fold_of(r["db_id"], args.folds, seed) == (outer + 1) % args.folds]
            train = [r for r in records if fold_of(r["db_id"], args.folds, seed) not in {outer, (outer + 1) % args.folds}]
            models = selector.train(train)
            calibration_scores = selector.score(calibration, models)
            test_scores = selector.score(test, models)
            policy = calibrate(calibration, calibration_scores)
            methods = {
                mode: evaluate(test, test_scores, policy["risk_weight"], policy["threshold"], mode)
                for mode in ("no_op", "apply_all", "learned")
            }
            folds.append({"fold": outer, "policy": policy, "methods": methods})
        pooled = {}
        for mode in ("no_op", "apply_all", "learned"):
            cases = sorted(
                [case for fold in folds for case in fold["methods"][mode]["cases"]],
                key=lambda row: row["index"],
            )
            pooled[mode] = metric(cases)
        versus_noop = cluster_bootstrap_difference(
            pooled["learned"]["cases"], pooled["no_op"]["cases"], seed + round(target * 1000)
        )
        versus_apply_all = cluster_bootstrap_difference(
            pooled["learned"]["cases"], pooled["apply_all"]["cases"], seed + 10000 + round(target * 1000)
        )
        results[str(target)] = {
            "realized_precision": allocation["realized_precision"],
            "pooled": {k: {x: y for x, y in v.items() if x != "cases"} for k, v in pooled.items()},
            "learned_minus_no_op": versus_noop,
            "learned_minus_apply_all": versus_apply_all,
            "reliably_beats_both": versus_noop["ci_low"] > 0 and versus_apply_all["ci_low"] > 0,
            "folds": [
                {
                    "fold": fold["fold"],
                    "policy": fold["policy"],
                    "methods": {
                        k: {x: y for x, y in v.items() if x != "cases"}
                        for k, v in fold["methods"].items()
                    },
                }
                for fold in folds
            ],
        }
        print(json.dumps({"target": target, **{k: v for k, v in results[str(target)].items() if k != "folds"}}, indent=2), flush=True)

    predicted = [prereg["phase_predictions"][str(p)]["learned_gain_mean"] for p in targets]
    observed = [results[str(p)]["pooled"]["learned"]["gain"] for p in targets]
    rho, rho_p = spearmanr(predicted, observed)
    predicted_transition = next(
        (p for p in targets if prereg["phase_predictions"][str(p)]["reliable_against_both"]),
        None,
    )
    observed_transition = next((p for p in targets if results[str(p)]["reliably_beats_both"]), None)
    if predicted_transition is None or observed_transition is None:
        transition_adjacent = predicted_transition == observed_transition
    else:
        transition_adjacent = abs(targets.index(predicted_transition) - targets.index(observed_transition)) <= 1
    flat = max(observed) - min(observed) < .01
    revealed = {
        "status": "REVEALED AFTER HASH-VERIFIED PREREGISTRATION",
        "preregistration_sha256": sha256(prereg_bytes),
        "results": results,
        "registered_tests": {
            "spearman_rho": float(rho),
            "spearman_p": float(rho_p),
            "curve_agreement_pass": bool(rho > .8),
            "predicted_transition": predicted_transition,
            "observed_transition": observed_transition,
            "transition_within_one_adjacent_target": transition_adjacent,
            "observed_curve_flat_under_one_point_range": flat,
        },
        "interpretation": (
            "supports_precision_intervention" if rho > .8 and transition_adjacent
            else "feature_fidelity_bottleneck" if flat
            else "phase_miss_or_manipulation_artifact"
        ),
        "neutral_mass_limitation": (
            "The matched cohort holds 468 neutral proposals in addition to 612 "
            "non-neutral proposals. The frozen phase has no explicit neutral-mass "
            "axis; curve shape is the primary comparison, not absolute calibration."
        ),
    }
    args.output.write_text(json.dumps(revealed, indent=2) + "\n")
    print(json.dumps(revealed["registered_tests"], indent=2))
    print(f"interpretation={revealed['interpretation']}")


if __name__ == "__main__":
    main()
