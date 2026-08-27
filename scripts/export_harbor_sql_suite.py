#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "harbor/tasks/radmeasure_sql_repair_v1"

REPAIRS = {
    "fix-column": "SELECT name FROM employees WHERE department='Systems'",
    "fix-table": "SELECT name FROM employees",
    "fix-filter": "SELECT name FROM employees WHERE department='AI'",
    "fix-aggregate": "SELECT COUNT(*) AS total FROM employees WHERE department='Systems'",
    "fix-id": "SELECT id, name FROM employees",
    "fix-department-column": "SELECT name, department FROM employees",
    "fix-salary-column": "SELECT name, salary FROM employees ORDER BY salary DESC",
    "fix-count-alias": "SELECT COUNT(*) AS total FROM employees",
    "fix-ai-filter": "SELECT name FROM employees WHERE department='AI'",
    "fix-systems-filter": "SELECT name FROM employees WHERE department='Systems'",
    "fix-average": "SELECT AVG(salary) AS avg_salary FROM employees",
    "fix-minimum": "SELECT MIN(salary) AS min_salary FROM employees",
}


def write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def main() -> None:
    cases = json.loads((ROOT / "data/benchmarks/sql_repair_v1.json").read_text())
    public_cases = [
        {"id": row["id"], "request": row["request"], "current_sql": row["broken_sql"]}
        for row in cases
    ]
    expected = []
    oracle = []
    for row in cases:
        action = row["expected_action"]
        gold_sql = ""
        if action == "KEEP":
            gold_sql = row["broken_sql"]
        elif action == "REPAIR":
            gold_sql = REPAIRS[row["id"]]
        expected.append({"id": row["id"], "expected_action": action, "gold_sql": gold_sql})
        oracle.append({
            "id": row["id"],
            "action": action,
            "tool": "" if action == "STOP" else "sql_query",
            "arguments": {"sql": gold_sql},
            "source": "oracle",
        })

    write(TASK / "environment/cases.json", json.dumps(public_cases, indent=2) + "\n")
    write(TASK / "tests/expected.json", json.dumps(expected, indent=2) + "\n")
    write(TASK / "solution/oracle_submission.json", json.dumps(oracle, indent=2) + "\n")


if __name__ == "__main__":
    main()
