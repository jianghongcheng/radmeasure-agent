from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def frozen_sql_submission(cases_path: Path, results_path: Path) -> list[dict[str, Any]]:
    """Convert the frozen Qwen planner artifact into Harbor's submission schema."""
    case_payload = json.loads(cases_path.read_text())
    cases = case_payload["cases"] if isinstance(case_payload, dict) else case_payload
    results = json.loads(results_path.read_text())
    generations = results["generations"]
    rows: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["id"])
        proposal = json.loads(generations[case_id]["content"])
        rows.append({
            "id": case_id,
            "action": str(proposal.get("action", "STOP")).upper(),
            "tool": str(proposal.get("tool", "")),
            "arguments": {"sql": str(proposal.get("sql", ""))},
            "source": "qwen3:8b-frozen",
        })
    return rows
