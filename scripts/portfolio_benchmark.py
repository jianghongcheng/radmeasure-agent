#!/usr/bin/env python3
"""Repeat the full demo workflow and emit hiring-relevant reliability metrics."""

from __future__ import annotations

import argparse
import json
import math
import statistics

from geomed_copilot.production import DemoService
from geomed_copilot.tools import GeoMedTools


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def run(iterations: int) -> dict:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    tools = GeoMedTools(DemoService())
    results = [tools.analyze_radiograph("demo-foot-001", top_k=2) for _ in range(iterations)]
    latencies = [item["total_latency_ms"] for item in results]
    traces = [trace for item in results for trace in item["traces"]]
    return {
        "evaluation_type": "deterministic_portfolio_reliability_check_not_clinical_validation",
        "runs": iterations,
        "successful_runs": sum(item["status"] == "complete" for item in results),
        "tool_success_rate": sum(trace["ok"] for trace in traces) / len(traces),
        "citation_presence_rate": sum(bool(item["citations"]) for item in results) / iterations,
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 3),
            "p50": round(statistics.median(latencies), 3),
            "p95": round(percentile(latencies, 0.95), 3),
        },
        "provenance_mode": results[0]["provenance"]["mode"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(run(args.iterations), indent=2))


if __name__ == "__main__":
    main()
