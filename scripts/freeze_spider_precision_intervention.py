#!/usr/bin/env python3
"""Freeze the matched-switch Spider precision intervention before outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sign(value):
    return "benefit" if value > 0 else "harm" if value < 0 else "neutral"


def operator_type(action):
    return action.split(":", 1)[0]


def difficulty_proxy(sql):
    text = " " + " ".join(sql.lower().split()) + " "
    score = (
        text.count(" join ")
        + max(0, text.count(" select ") - 1) * 2
        + int(" where " in text)
        + int(" group by " in text)
        + int(" order by " in text)
        + int(" having " in text)
        + 2 * sum(text.count(f" {token} ") for token in ("union", "intersect", "except"))
    )
    return "easy" if score <= 1 else "medium" if score <= 3 else "hard" if score <= 5 else "extra"


def rank(seed, *parts):
    value = ":".join([str(seed), *(str(x) for x in parts)])
    return hashlib.sha256(value.encode()).hexdigest()


def choose_pair(record, desired_sign, seed):
    grouped = defaultdict(lambda: defaultdict(list))
    for index, candidate in enumerate(record["candidates"]):
        grouped[operator_type(candidate["action"])][sign(candidate["advantage"])].append(index)
    options = []
    for operation, labels in grouped.items():
        if labels[desired_sign] and labels["neutral"]:
            signed = min(labels[desired_sign], key=lambda i: rank(seed, record["index"], operation, desired_sign, i))
            neutral = min(labels["neutral"], key=lambda i: rank(seed, record["index"], operation, "neutral", i))
            options.append((operation, signed, neutral))
    if not options:
        return None
    return min(options, key=lambda row: rank(seed, record["index"], *row))


def bracket(values, target):
    if target in values:
        return target, target, 0.0
    lower = max(x for x in values if x < target)
    upper = min(x for x in values if x > target)
    return lower, upper, (target - lower) / (upper - lower)


def phase_prediction(rows, precision, clusters):
    rows = [r for r in rows if float(r["label_noise"]) == 0.0]
    ps = sorted({float(r["proposal_precision"]) for r in rows})
    cs = sorted({int(r["cluster_count"]) for r in rows})
    p0, p1, wp = bracket(ps, precision)
    c0, c1, wc = bracket(cs, clusters)
    lookup = {
        (float(r["proposal_precision"]), int(r["cluster_count"])): r
        for r in rows
    }
    metrics = [
        "learned_gain_mean",
        "prob_learned_beats_no_op",
        "prob_learned_beats_apply_all",
    ]
    output = {}
    for metric in metrics:
        v00 = float(lookup[(p0, c0)][metric])
        v01 = float(lookup[(p1, c0)][metric])
        v10 = float(lookup[(p0, c1)][metric])
        v11 = float(lookup[(p1, c1)][metric])
        low = v00 * (1 - wp) + v01 * wp
        high = v10 * (1 - wp) + v11 * wp
        output[metric] = low * (1 - wc) + high * wc
    output["reliable_against_both"] = min(
        output["prob_learned_beats_no_op"],
        output["prob_learned_beats_apply_all"],
    ) >= 0.8
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "outputs/research/spider_clean_base_executable_edits.json"
    )
    parser.add_argument(
        "--phase", type=Path,
        default=ROOT / "outputs/research/synthetic_selective_phase_extended_v1.json"
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "outputs/research/spider_precision_intervention_manifest.json"
    )
    parser.add_argument(
        "--preregistration", type=Path,
        default=ROOT / "outputs/research/spider_precision_intervention_preregistered.json"
    )
    parser.add_argument("--seed", type=int, default=260825)
    parser.add_argument("--active-mass", type=int, default=612)
    parser.add_argument("--targets", default="0.15,0.25,0.35,0.45,0.50")
    args = parser.parse_args()

    source_bytes = args.input.read_bytes()
    phase_bytes = args.phase.read_bytes()
    records = json.loads(source_bytes)["records"]
    benefit_pairs, harm_pairs = [], []
    for record in records:
        for desired, destination in (("benefit", benefit_pairs), ("harm", harm_pairs)):
            pair = choose_pair(record, desired, args.seed)
            if pair is not None:
                operation, signed, neutral = pair
                destination.append({
                    "index": record["index"],
                    "db_id": record["db_id"],
                    "difficulty": difficulty_proxy(record["gold_sql"]),
                    "operator_type": operation,
                    "signed_candidate_index": signed,
                    "neutral_candidate_index": neutral,
                    "sign": desired,
                })

    # Benefit and harm cases are disjoint under binary execution accuracy.
    assert not ({x["index"] for x in benefit_pairs} & {x["index"] for x in harm_pairs})
    benefit_pairs.sort(key=lambda x: rank(args.seed, "benefit-order", x["db_id"], x["index"]))
    harm_pairs.sort(key=lambda x: rank(args.seed, "harm-order", x["db_id"], x["index"]))
    targets = [float(x) for x in args.targets.split(",")]
    allocations = {}
    for precision in targets:
        benefit_n = round(precision * args.active_mass)
        harm_n = args.active_mass - benefit_n
        if benefit_n > len(benefit_pairs) or harm_n > len(harm_pairs):
            raise ValueError(f"Unsupported target {precision}")
        allocations[str(precision)] = {
            "benefit_active_indices": [x["index"] for x in benefit_pairs[:benefit_n]],
            "harm_active_indices": [x["index"] for x in harm_pairs[:harm_n]],
            "benefit_count": benefit_n,
            "harm_count": harm_n,
            "realized_precision": benefit_n / args.active_mass,
        }

    manifest = {
        "status": "FROZEN BEFORE SELECTOR OUTCOMES",
        "source_sha256": sha256_bytes(source_bytes),
        "seed": args.seed,
        "cohort_definition": "all cases with a same-question, same-operator non-neutral/neutral switch pair",
        "candidate_count_per_case": 1,
        "benefit_pairs": benefit_pairs,
        "harm_pairs": harm_pairs,
        "allocations": allocations,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    args.manifest.write_bytes(manifest_bytes)
    cohort_dbs = {x["db_id"] for x in benefit_pairs + harm_pairs}
    phase = json.loads(phase_bytes)["summary"]
    predictions = {
        str(p): phase_prediction(phase, p, len(cohort_dbs))
        for p in targets
    }
    prereg = {
        "status": "PREREGISTERED; MANIPULATED SELECTOR OUTCOMES NOT YET RUN",
        "design": {
            "domain": "Spider clean train databases",
            "questions_fixed_across_targets": True,
            "base_predictions_fixed": True,
            "candidate_count_per_case": 1,
            "within_case_matching": "same question and operator type; signed versus neutral existing candidate",
            "active_nonneutral_mass": args.active_mass,
            "target_precisions": targets,
            "database_split_seed": args.seed,
            "reward": "exact observed-database execution accuracy difference",
            "features": "frozen selector text features",
            "cohort_questions": len(benefit_pairs) + len(harm_pairs),
            "independent_database_clusters": len(cohort_dbs),
        },
        "artifact_qualification": {
            "direct_stratified_subsampling": "rejected: insufficient matched sign support",
            "harmful_injection": "rejected before preregistration: grouped origin AUC 0.806 > 0.60",
            "primary_method": "within-case matched switch",
        },
        "frozen_inputs": {
            "source_sha256": sha256_bytes(source_bytes),
            "phase_sha256": sha256_bytes(phase_bytes),
            "manifest_sha256": sha256_bytes(manifest_bytes),
        },
        "phase_predictions": predictions,
        "success_criteria": {
            "curve_agreement": "Spearman correlation between five frozen predicted and observed selective gains > 0.80",
            "transition_agreement": "first reliably selective target differs by at most one adjacent target; reliable means probability >=0.80 in phase and clustered 95% CI lower bound >0 against both no-op and apply-all in real data",
            "interpretation_if_flat": "precision is not the dominant omitted bottleneck on Spider clean; feature fidelity limits learning",
            "interpretation_if_reversed": "manipulation artifact or phase miss; do not claim causal precision effect",
        },
    }
    prereg_bytes = (json.dumps(prereg, indent=2, sort_keys=True) + "\n").encode()
    args.preregistration.write_bytes(prereg_bytes)
    digest_path = args.preregistration.with_suffix(".sha256")
    digest_path.write_text(f"{sha256_bytes(prereg_bytes)}  {args.preregistration.name}\n")
    print(json.dumps(prereg, indent=2))
    print(f"preregistration_sha256={sha256_bytes(prereg_bytes)}")


if __name__ == "__main__":
    main()
