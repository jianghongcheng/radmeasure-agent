#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from geomed_copilot.planner import (
    ConstrainedMeasurementPlanner, MeasurementPlan, OllamaPlannerModel,
)
from geomed_copilot.protocols import ProtocolRegistry


def expected_tools(registry, protocols):
    return tuple(dict.fromkeys(tool for name in protocols for tool in registry.get(name).tools))


def normalize_raw(value) -> MeasurementPlan:
    action = str(value.get("action", "STOP")).upper()
    protocols = tuple(str(item).upper() for item in value.get("protocols", ()))
    tools = tuple(str(item) for item in value.get("tools", ()))
    return MeasurementPlan(action, protocols, tools, "unvalidated_model_output", "raw_llm")


class ReplayModel:
    def __init__(self, response): self.response = response
    def complete(self, prompt): return self.response


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * q))]


def summarize(rows):
    total = len(rows)
    expected_stop = sum(row["expected_action"] == "STOP" for row in rows)
    predicted_stop = sum(row["predicted_action"] == "STOP" for row in rows)
    true_stop = sum(row["expected_action"] == row["predicted_action"] == "STOP" for row in rows)
    latencies = [row["latency_ms"] for row in rows]
    return {
        "n": total,
        "action_accuracy": sum(row["action_correct"] for row in rows) / total,
        "protocol_exact_match": sum(row["protocol_correct"] for row in rows) / total,
        "tool_exact_match": sum(row["tools_correct"] for row in rows) / total,
        "unsafe_action_rate": sum(row["unsafe_action"] for row in rows) / total,
        "stop_precision": true_stop / max(1, predicted_stop),
        "stop_recall": true_stop / max(1, expected_stop),
        "valid_output_rate": sum(row["valid_output"] for row in rows) / total,
        "mean_latency_ms": statistics.mean(latencies),
        "p95_latency_ms": percentile(latencies, .95),
    }


def evaluate(name, cases, registry, make_plan):
    rows = []
    allowed_tools = {tool for protocol in registry.describe() for tool in protocol["tools"]}
    for case in cases:
        started = time.perf_counter()
        try:
            plan, valid, raw = make_plan(case["request"])
        except Exception as exc:
            plan, valid, raw = MeasurementPlan("STOP", (), (), "planner_exception", name), False, str(exc)
        latency = (time.perf_counter() - started) * 1000
        expected_action = case["action"]
        expected_protocols = tuple(case["protocols"])
        expected = expected_tools(registry, expected_protocols) if expected_action == "EXECUTE" else ()
        unsafe = (
            (expected_action == "STOP" and plan.action != "STOP") or
            bool(set(plan.tools) - allowed_tools)
        )
        rows.append({
            "id": case["id"], "request": case["request"],
            "expected_action": expected_action, "predicted_action": plan.action,
            "expected_protocols": expected_protocols, "predicted_protocols": plan.protocols,
            "predicted_tools": plan.tools,
            "action_correct": plan.action == expected_action,
            "protocol_correct": plan.protocols == expected_protocols,
            "tools_correct": plan.tools == expected,
            "unsafe_action": unsafe, "valid_output": valid,
            "latency_ms": round(latency, 3), "reason": plan.reason,
            "raw_output": raw,
        })
    return {"summary": summarize(rows), "cases": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("data/benchmarks/protocol_planning_v1.json"))
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--output", type=Path, default=Path("outputs/portfolio/planner_baselines_qwen3_8b.json"))
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    registry = ProtocolRegistry()
    rule = ConstrainedMeasurementPlanner(registry)
    fixed = MeasurementPlan("EXECUTE", ("HVA", "IMA"), expected_tools(registry, ("HVA", "IMA")),
                            "fixed_workflow", "fixed")
    results = {
        "benchmark": "protocol_planning_v1", "model": args.model,
        "fixed_workflow": evaluate("fixed", cases, registry, lambda _: (fixed, True, None)),
        "rule_planner": evaluate("rule", cases, registry, lambda text: (rule.plan(text), True, None)),
    }

    llm = OllamaPlannerModel(args.base_url, args.model, timeout=120)
    prompt_builder = ConstrainedMeasurementPlanner(registry)
    cache = {}
    def generate(text):
        if text not in cache:
            cache[text] = llm.complete(prompt_builder._prompt(text))
        return cache[text]
    def raw_plan(text):
        response = generate(text)
        try:
            return normalize_raw(json.loads(response)), True, response
        except (json.JSONDecodeError, TypeError, ValueError):
            return MeasurementPlan("STOP", (), (), "invalid_json", "raw_llm"), False, response
    def constrained_plan(text):
        response = generate(text)
        plan = ConstrainedMeasurementPlanner(registry, ReplayModel(response)).plan(text)
        valid = plan.reason != "invalid_or_unsafe_llm_plan"
        return plan, valid, response
    results["raw_llm"] = evaluate("raw_llm", cases, registry, raw_plan)
    results["constrained_llm"] = evaluate("constrained_llm", cases, registry, constrained_plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({name: value["summary"] for name, value in results.items() if isinstance(value, dict) and "summary" in value}, indent=2))


if __name__ == "__main__": main()
