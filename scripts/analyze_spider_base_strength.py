#!/usr/bin/env python3
"""Pooled base-strength, headroom, and difficulty analysis for Spider dev."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/research"


def load_evaluator():
    path = ROOT / "third_party/spider/evaluation.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("spider_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Evaluator()


def pooled_cases(selector: dict, method: str) -> list[dict]:
    return [case for fold in selector["folds"] for case in fold["methods"][method]["cases"]]


def metrics(records: list[dict], cases: list[dict], indices: list[int]) -> dict:
    selected = [cases[index] for index in indices]
    base = sum(case["before_correct"] for case in selected)
    after = sum(case["after_correct"] for case in selected)
    benefit = sum(not case["before_correct"] and case["after_correct"] for case in selected)
    harm = sum(case["before_correct"] and not case["after_correct"] for case in selected)
    edited = sum(case["edited"] for case in selected)
    invalid_benefit = sum(
        not records[index]["base_executable"]
        and not cases[index]["before_correct"]
        and cases[index]["after_correct"]
        for index in indices
    )
    n = len(selected)
    return {
        "n": n,
        "base_accuracy": base / n,
        "after_accuracy": after / n,
        "absolute_gain": (after - base) / n,
        "benefit_count": benefit,
        "harm_count": harm,
        "edit_count": edited,
        "benefit_from_invalid_base_count": invalid_benefit,
        "benefit_from_valid_but_wrong_base_count": benefit - invalid_benefit,
    }


def main() -> None:
    dev = json.loads((ROOT / "third_party/spider_data/spider_data/dev.json").read_text())
    evaluator = load_evaluator()
    difficulty = [evaluator.eval_hardness(row["sql"]) for row in dev]
    inputs = {
        "repository_example_weak": ("spider_executable_edits.json", "spider_advantage_selector_cv.json"),
        **{f"CodeS-{size}": (f"spider_codes_{size}_executable_edits.json", f"spider_codes_{size}_selector_cv.json") for size in ["1b", "3b", "7b", "15b"]},
    }
    result = {
        "status": "exploratory Spider dev analysis; not untouched confirmation",
        "difficulty_definition": "Official Spider Evaluator.eval_hardness on gold SQL",
        "models": {},
    }
    rows = []
    for name, (record_file, selector_file) in inputs.items():
        artifact = json.loads((OUT / record_file).read_text())
        records = artifact["records"]
        selector = json.loads((OUT / selector_file).read_text())
        by_method = {}
        for method in ["learned_exact_advantage", "oracle_candidate"]:
            cases = sorted(pooled_cases(selector, method), key=lambda row: row["index"])
            assert [case["index"] for case in cases] == list(range(len(records)))
            overall = metrics(records, cases, list(range(len(records))))
            strata = {
                level: metrics(records, cases, [index for index, value in enumerate(difficulty) if value == level])
                for level in ["easy", "medium", "hard", "extra"]
            }
            by_method[method] = {"overall": overall, "by_difficulty": strata}
        oracle_gain = by_method["oracle_candidate"]["overall"]["absolute_gain"]
        learned_gain = by_method["learned_exact_advantage"]["overall"]["absolute_gain"]
        learned_benefit = by_method["learned_exact_advantage"]["overall"]["benefit_count"] / len(records)
        summary = {
            "prediction_sha256": artifact.get("prediction_sha256"),
            "base_accuracy": artifact["summary"]["baseline_execution_accuracy"],
            "candidate_oracle_headroom": oracle_gain,
            "learned_gain": learned_gain,
            "gross_benefit_capture": learned_benefit / oracle_gain if oracle_gain else None,
            "net_oracle_recovery": learned_gain / oracle_gain if oracle_gain else None,
            "methods": by_method,
        }
        result["models"][name] = summary
        rows.append({"model": name, **{key: value for key, value in summary.items() if key != "methods"}})
    (OUT / "spider_base_strength_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    with (OUT / "spider_base_strength_analysis.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "prediction_sha256", "base_accuracy", "candidate_oracle_headroom", "learned_gain", "gross_benefit_capture", "net_oracle_recovery"])
        writer.writeheader()
        writer.writerows(rows)
    ordered = list(result["models"].items())
    x = [row["base_accuracy"] * 100 for _, row in ordered]
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.plot(x, [row["candidate_oracle_headroom"] * 100 for _, row in ordered], "o-", label="Candidate-oracle headroom")
    axis.plot(x, [row["learned_gain"] * 100 for _, row in ordered], "s-", label="Realized learned gain")
    for (name, _), left, top in zip(ordered, x, [row["candidate_oracle_headroom"] * 100 for _, row in ordered]):
        axis.annotate(name, (left, top), xytext=(3, 4), textcoords="offset points", fontsize=8)
    axis.axhline(0, color="black", linewidth=.8)
    axis.set_xlabel("Base execution accuracy (%)")
    axis.set_ylabel("Absolute accuracy change (points)")
    axis.set_title("Cross-model diagnostic (model identity is confounded)")
    axis.grid(alpha=.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUT / "spider_base_strength_curve.png", dpi=180)
    print(json.dumps({name: {key: value for key, value in row.items() if key != "methods"} for name, row in result["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
