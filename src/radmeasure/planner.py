from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from typing import Protocol
import urllib.request

from .protocols import ProtocolRegistry


class PlannerModel(Protocol):
    def complete(self, prompt: str) -> str: ...


class OpenAICompatiblePlannerModel:
    """Small provider adapter; compatible with local vLLM and hosted chat APIs."""

    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 10.0) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def complete(self, prompt: str) -> str:
        body = json.dumps({
            "model": self.model,
            "temperature": 0,
            "max_tokens": 256,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }).encode()
        request = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            value = json.loads(response.read())
        return str(value["choices"][0]["message"]["content"])


class OllamaPlannerModel:
    """Ollama-native adapter with thinking disabled and JSON constrained output."""

    def __init__(self, base_url: str, model: str, timeout: float = 30.0) -> None:
        self.url = base_url.rstrip("/") + "/api/chat"
        self.model = model
        self.timeout = timeout

    def complete(self, prompt: str) -> str:
        return self.complete_with_metadata(prompt)[0]

    def complete_with_metadata(self, prompt: str) -> tuple[str, dict]:
        body = json.dumps({
            "model": self.model,
            "stream": False,
            "think": False,
            "format": "json",
            "options": {"temperature": 0, "num_predict": 256},
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        request = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            value = json.loads(response.read())
        metadata = {
            "prompt_tokens": int(value.get("prompt_eval_count", 0)),
            "completion_tokens": int(value.get("eval_count", 0)),
            "total_duration_ns": int(value.get("total_duration", 0)),
        }
        return str(value["message"]["content"]), metadata


@dataclass(frozen=True)
class MeasurementPlan:
    action: str
    protocols: tuple[str, ...]
    tools: tuple[str, ...]
    reason: str
    source: str

    def to_dict(self) -> dict:
        return asdict(self)


class ConstrainedMeasurementPlanner:
    """LLM-compatible planner whose output is always checked against a registry."""

    def __init__(self, registry: ProtocolRegistry, model: PlannerModel | None = None) -> None:
        self.registry = registry
        self.model = model

    def plan(self, request: str) -> MeasurementPlan:
        if self.model:
            try:
                raw = self.model.complete(self._prompt(request))
                return self._validate(json.loads(raw), "llm")
            except Exception:  # provider, parse, and policy failures all fail closed
                # Invalid model output fails closed; it never becomes an executable plan.
                return MeasurementPlan("STOP", (), (), "invalid_or_unsafe_llm_plan", "llm")
        return self._rule_plan(request)

    def _prompt(self, request: str) -> str:
        return json.dumps({
            "instruction": "/no_think Return JSON only. Select registered protocols and tools or STOP. Do not explain.",
            "request": request,
            "registry": self.registry.describe(),
            "schema": {"action": "EXECUTE|STOP", "protocols": ["HVA"], "tools": ["tool"]},
        })

    def _rule_plan(self, request: str) -> MeasurementPlan:
        lowered = request.lower()
        selected: list[str] = []
        if "hallux" in lowered or "hva" in lowered:
            selected.append("HVA")
        if "intermetatarsal" in lowered or "ima" in lowered:
            selected.append("IMA")
        if not selected and "measure" in lowered and "foot" in lowered:
            selected = ["HVA", "IMA"]
        if not selected:
            return MeasurementPlan("STOP", (), (), "no_supported_protocol_requested", "constrained_fallback")
        tools = tuple(dict.fromkeys(
            tool for name in selected for tool in self.registry.get(name).tools
        ))
        return MeasurementPlan("EXECUTE", tuple(selected), tools, "registered_protocol_match", "constrained_fallback")

    def _validate(self, value: dict, source: str) -> MeasurementPlan:
        action = str(value.get("action", "STOP")).upper()
        if action == "STOP":
            return MeasurementPlan("STOP", (), (), "planner_requested_stop", source)
        if action != "EXECUTE":
            raise ValueError("unsupported planner action")
        protocols = tuple(str(item).upper() for item in value.get("protocols", ()))
        tools = tuple(str(item) for item in value.get("tools", ()))
        if not protocols:
            raise ValueError("an executable plan requires a protocol")
        for name in protocols:
            self.registry.validate_tools(name, tools)
        return MeasurementPlan("EXECUTE", protocols, tools, "registry_validated", source)


def planner_from_env(registry: ProtocolRegistry) -> ConstrainedMeasurementPlanner:
    """Enable a real LLM planner only when an explicit endpoint is configured."""
    base_url = os.environ.get("RADMEASURE_PLANNER_BASE_URL", "").strip()
    model = os.environ.get("RADMEASURE_PLANNER_MODEL", "").strip()
    if not base_url or not model:
        return ConstrainedMeasurementPlanner(registry)
    timeout = float(os.environ.get("RADMEASURE_PLANNER_TIMEOUT_SECONDS", "10"))
    if os.environ.get("RADMEASURE_PLANNER_PROVIDER", "openai_compatible").strip().lower() == "ollama":
        adapter = OllamaPlannerModel(base_url=base_url, model=model, timeout=timeout)
    else:
        adapter = OpenAICompatiblePlannerModel(
            base_url=base_url,
            model=model,
            api_key=os.environ.get("RADMEASURE_PLANNER_API_KEY", ""),
            timeout=timeout,
        )
    return ConstrainedMeasurementPlanner(registry, adapter)
