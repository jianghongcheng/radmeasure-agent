#!/usr/bin/env python3
"""Extract immutable model generations from a full SQL harness result."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = json.loads(args.result.read_text())
    rows = {row["id"]: row for row in result["layers"]["llm_only"]["cases"]}
    payload = {
        "benchmark": result["benchmark"],
        "model": result["model"],
        "case_suite_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "generation_count": len(result["generations"]),
        "generation_ms": {case_id: rows[case_id]["planner_generation_ms"]
                          for case_id in result["generations"]},
        "generations": result["generations"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"froze {payload['generation_count']} generations to {args.output}")


if __name__ == "__main__":
    main()
