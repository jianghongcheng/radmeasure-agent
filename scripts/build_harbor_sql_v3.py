#!/usr/bin/env python3
"""Materialize the public and hidden artifacts for Harbor SQL v3."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "data/benchmarks/sql_repair_v3_confirmatory.json"
TASK = ROOT / "harbor/tasks/radmeasure_sql_repair_v3"


def main() -> None:
    payload = json.loads(SUITE.read_text())
    public = []
    hidden = []
    for case in payload["cases"]:
        database = payload["databases"][case["domain"]]
        public.append({
            "id": case["id"],
            "domain": case["domain"],
            "request": case["request"],
            "current_sql": case["broken_sql"],
            "schema": database["schema"],
        })
        hidden.append({
            "id": case["id"],
            "expected_action": case["expected_action"],
            "gold_sql": case["gold_sql"],
            "setup_sql": database["setup_sql"],
        })
    (TASK / "environment").mkdir(parents=True, exist_ok=True)
    (TASK / "tests").mkdir(parents=True, exist_ok=True)
    (TASK / "solution").mkdir(parents=True, exist_ok=True)
    (TASK / "environment/cases.json").write_text(json.dumps(public, indent=2) + "\n")
    (TASK / "tests/expected.json").write_text(json.dumps(hidden, indent=2) + "\n")
    oracle = [{
        "id": case["id"],
        "action": case["expected_action"],
        "tool": "" if case["expected_action"] == "STOP" else "sql_query",
        "arguments": {"sql": case["gold_sql"]},
        "source": "oracle",
    } for case in payload["cases"]]
    (TASK / "solution/oracle_submission.json").write_text(json.dumps(oracle, indent=2) + "\n")
    print(f"materialized {len(public)} Harbor v3 cases in {TASK}")


if __name__ == "__main__":
    main()
