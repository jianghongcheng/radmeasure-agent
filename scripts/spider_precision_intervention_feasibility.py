#!/usr/bin/env python3
"""Audit support for a one-variable Spider proposal-precision intervention.

This script deliberately does not train a selector or evaluate selective gain.
It asks whether precision can be manipulated with existing candidates while
holding questions, per-question candidate counts, operator composition, and
difficulty composition fixed.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sign(advantage):
    return "benefit" if advantage > 0 else "harm" if advantage < 0 else "neutral"


def difficulty_proxy(sql):
    """Frozen lexical SQL-complexity proxy for train-set records.

    The clean artifact contains Spider train examples, whereas official
    evaluator hardness annotations are only directly indexed for dev here.
    """
    text = " " + re.sub(r"\s+", " ", sql.lower()) + " "
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


def operator_type(action):
    return action.split(":", 1)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "outputs/research/spider_clean_base_executable_edits.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/research/spider_precision_intervention_feasibility.json",
    )
    parser.add_argument("--targets", default="0.15,0.25,0.35,0.45,0.50")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    records = payload["records"]

    strata = defaultdict(Counter)
    action_counts = Counter()
    case_support = Counter()
    base_by_sign = defaultdict(Counter)
    total = Counter()
    matched_switch = Counter()

    for record in records:
        difficulty = difficulty_proxy(record["gold_sql"])
        available = set()
        within_operator = defaultdict(set)
        for candidate in record["candidates"]:
            label = sign(candidate["advantage"])
            action = operator_type(candidate["action"])
            strata[(difficulty, action)][label] += 1
            action_counts[action] += 1
            base_by_sign["base_correct" if record["base_correct"] else "base_wrong"][label] += 1
            total[label] += 1
            available.add(label)
            within_operator[action].add(label)
        if any({"benefit", "neutral"} <= labels for labels in within_operator.values()):
            matched_switch["benefit_to_neutral"] += 1
        if any({"harm", "neutral"} <= labels for labels in within_operator.values()):
            matched_switch["harm_to_neutral"] += 1
        case_support["+".join(sorted(available)) if available else "no_candidates"] += 1

    nonneutral = total["benefit"] + total["harm"]
    observed_precision = total["benefit"] / nonneutral if nonneutral else None
    mixed_sign_cases = sum(
        count for key, count in case_support.items()
        if "benefit" in key and "harm" in key
    )
    both_sign_strata = sum(
        1 for counts in strata.values()
        if counts["benefit"] and counts["harm"]
    )

    # A target is support-feasible within every occupied stratum only if each
    # stratum with non-neutral candidates contains both signs. This is a
    # necessary (not sufficient) condition for preserving exact stratum margins.
    occupied_nonneutral_strata = [
        counts for counts in strata.values()
        if counts["benefit"] + counts["harm"]
    ]
    strict_stratified_support = all(
        counts["benefit"] and counts["harm"]
        for counts in occupied_nonneutral_strata
    )
    targets = [float(x) for x in args.targets.split(",")]
    # A common number of non-neutral proposals N permits every target p when
    # pN benefit switches and (1-p)N harm switches are available. Floors are
    # conservative because exact integer allocations are checked later.
    max_common_n = min(
        matched_switch["benefit_to_neutral"] / max(targets),
        matched_switch["harm_to_neutral"] / max(1 - p for p in targets),
    )
    common_n = int(max_common_n)
    matched_target_support = {
        str(p): {
            "benefit_switches_required": round(p * common_n),
            "harm_switches_required": common_n - round(p * common_n),
            "supported": (
                round(p * common_n) <= matched_switch["benefit_to_neutral"]
                and common_n - round(p * common_n) <= matched_switch["harm_to_neutral"]
            ),
        }
        for p in targets
    }

    result = {
        "status": "method-feasibility audit only; no selector trained and no gain evaluated",
        "input": str(args.input),
        "n_questions": len(records),
        "target_precisions": targets,
        "difficulty_definition": "Frozen lexical SQL-complexity proxy; not official Spider hardness",
        "observed_candidate_precision": observed_precision,
        "candidate_sign_counts": dict(total),
        "base_correctness_by_candidate_sign": {
            key: dict(value) for key, value in base_by_sign.items()
        },
        "case_level_sign_support": dict(case_support),
        "n_cases_with_both_benefit_and_harm_candidates": mixed_sign_cases,
        "n_difficulty_action_strata": len(strata),
        "n_occupied_nonneutral_strata": len(occupied_nonneutral_strata),
        "n_strata_with_both_benefit_and_harm": both_sign_strata,
        "strict_within_stratum_subsampling_support": strict_stratified_support,
        "within_case_same_operator_matched_switch_counts": dict(matched_switch),
        "largest_common_nonneutral_mass_across_targets": common_n,
        "matched_switch_target_support": matched_target_support,
        "identification_warning": (
            "With binary execution accuracy, beneficial candidates occur only "
            "when the base is wrong and harmful candidates only when the base is "
            "correct. Changing their ratio can therefore change base-correctness "
            "and feature composition unless both signs have matched support."
        ),
        "strata": [
            {
                "difficulty": difficulty,
                "action": action,
                **dict(counts),
            }
            for (difficulty, action), counts in sorted(strata.items())
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "strata"}, indent=2))


if __name__ == "__main__":
    main()
