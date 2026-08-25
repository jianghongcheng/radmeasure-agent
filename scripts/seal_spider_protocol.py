#!/usr/bin/env python3
"""Seal database-level Spider train partitions before confirmatory evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "third_party/spider_data/spider_data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank(seed: str, db_id: str) -> str:
    return hashlib.sha256(f"{seed}:{db_id}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default="geomed-spider-confirmatory-v1-2026-08-24")
    parser.add_argument("--confirmatory-databases", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/research/spider_protocol_seal_v1.json",
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    source_paths = [DATA / "train_spider.json", DATA / "train_others.json"]
    db_counts: dict[str, int] = {}
    for path in source_paths:
        # Partition construction reads db_id only; query/gold fields are never used.
        for row in json.loads(path.read_text()):
            db_counts[row["db_id"]] = db_counts.get(row["db_id"], 0) + 1

    ordered = sorted(db_counts, key=lambda db_id: rank(args.seed, db_id))
    confirmatory = sorted(ordered[: args.confirmatory_databases])
    development = sorted(ordered[args.confirmatory_databases :])
    code_paths = [
        ROOT / "scripts/build_spider_executable_edits.py",
        ROOT / "scripts/spider_advantage_selector_cv.py",
        ROOT / "docs/SPIDER_EVALUATION_PROTOCOL.md",
    ]
    result = {
        "protocol": "Spider database-grouped confirmatory seal",
        "version": 1,
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "partition_seed": args.seed,
        "partition_uses_fields": ["db_id"],
        "partition_excludes_fields": ["query", "sql", "query_toks", "query_toks_no_value"],
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
        "frozen_method_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in code_paths},
        "counts": {
            "all_databases": len(db_counts),
            "all_examples": sum(db_counts.values()),
            "development_databases": len(development),
            "development_examples": sum(db_counts[db] for db in development),
            "confirmatory_databases": len(confirmatory),
            "confirmatory_examples": sum(db_counts[db] for db in confirmatory),
        },
        "development_databases": development,
        "confirmatory_databases": confirmatory,
        "confirmatory_labels_inspected": False,
        "rule": "No confirmatory gold execution until model outputs and method hashes are frozen.",
    }

    if args.output.exists():
        existing = json.loads(args.output.read_text())
        comparable = {key: value for key, value in result.items() if key != "sealed_at_utc"}
        old_comparable = {key: value for key, value in existing.items() if key != "sealed_at_utc"}
        if not args.verify:
            raise SystemExit(f"Refusing to overwrite existing seal: {args.output}")
        if comparable != old_comparable:
            raise SystemExit("SEAL VERIFICATION FAILED: current inputs differ from frozen manifest")
        print(json.dumps({"seal_verified": True, "output": str(args.output)}, indent=2))
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
