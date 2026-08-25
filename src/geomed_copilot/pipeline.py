from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from .jobs import Job
from .inference_client import InferenceClient
from .agent_controller import MeasurementAgentController
from .planner import ConstrainedMeasurementPlanner, planner_from_env
from .protocols import ProtocolRegistry
from .tools import GeoMedTools


@dataclass(frozen=True)
class PipelineOutcome:
    status: str
    result: dict[str, Any]


class JobPipeline:
    def __init__(self, tools: GeoMedTools,
                 inference_client: InferenceClient | None = None,
                 planner: ConstrainedMeasurementPlanner | None = None,
                 controller: MeasurementAgentController | None = None) -> None:
        self.tools = tools
        self.inference_client = inference_client
        registry = ProtocolRegistry()
        self.planner = planner or planner_from_env(registry)
        self.controller = controller or MeasurementAgentController(registry)

    def run(self, job: Job) -> PipelineOutcome:
        if job.job_type == "evaluation_analysis":
            question = str(job.payload.get("question") or "Measure HVA and IMA with supporting evidence.")
            plan = self.planner.plan(question)
            if plan.action == "STOP":
                return PipelineOutcome("needs_review", {
                    "measurements": [],
                    "agent_plan": plan.to_dict(),
                    "agent_trajectory": [
                        {"step": "plan", **plan.to_dict()},
                        {"step": "decision", "action": "STOP", "reason": plan.reason},
                    ],
                    "routing": {"decision": "STOP", "reason": plan.reason},
                    "trace_id": job.payload.get("_trace_id"),
                })
            result = self.tools.analyze_radiograph(
                image_id=str(job.payload["image_id"]),
                question=question,
                top_k=int(job.payload.get("top_k", 3)),
            )
            if self.inference_client:
                remote = self.inference_client.predict(str(job.payload["image_id"]))
                local = {item["name"]: item["predicted_degrees"] for item in result["measurements"]}
                disagreement = any(
                    abs(local[name] - value) > 0.001
                    for name, value in remote["measurements"].items()
                )
                result["inference_service"] = {
                    "model": remote["model"],
                    "agreement_with_workflow": not disagreement,
                }
            else:
                disagreement = False
            agent = self.controller.execute(plan, result)
            result["measurements"] = agent.measurements
            result["agent_plan"] = plan.to_dict()
            result["agent_trajectory"] = agent.trajectory
            result["repair_attempts"] = agent.repair_attempts
            review = agent.decision == "STOP" or disagreement
            result["routing"] = {
                "decision": "STOP" if review else agent.decision,
                "reason": (
                    "inference_service_disagreement" if disagreement else
                    agent.reason
                ),
            }
            result["trace_id"] = job.payload.get("_trace_id")
            return PipelineOutcome("needs_review" if review else "completed", result)
        if job.job_type == "uploaded_radiograph":
            artifact = job.payload["artifact"]
            if not self.inference_client:
                return PipelineOutcome("needs_review", {
                    "artifact": artifact,
                    "routing": {"decision": "human_review", "reason": "live_inference_adapter_unavailable"},
                    "provenance": {"mode": "ingested_without_live_inference", "live_encoder_inference": False, "clinical_use": False},
                    "trace_id": job.payload.get("_trace_id"),
                })
            prediction = self.inference_client.predict_artifact(
                image_id=artifact["sha256"], artifact_uri=artifact["path"],
                media_type=artifact.get("media_type", "image/jpeg"),
            )
            repair_model_id = os.environ.get("GEOMED_REPAIR_MODEL_ID", "").strip()
            if repair_model_id:
                candidate = self.inference_client.predict_artifact(
                    image_id=artifact["sha256"], artifact_uri=artifact["path"],
                    media_type=artifact.get("media_type", "image/jpeg"),
                    model_id=repair_model_id,
                )
                proposal = candidate.get("repair_proposal")
                if proposal:
                    proposal = dict(proposal)
                    proposal["model"] = candidate["model"]
                    proposal["cross_model_discrepancy"] = {
                        name: abs(float(proposal["measurements"][name]) - float(prediction["measurements"][name]))
                        for name in ("HVA", "IMA")
                    }
                    proposal["accepted"] = bool(proposal.get("accepted")) and all(
                        proposal["cross_model_discrepancy"][name] <= limit
                        for name, limit in {"HVA": 5.0, "IMA": 3.0}.items()
                    )
                    if not proposal["accepted"]:
                        proposal["policy_rejection_reason"] = "cross_model_disagreement"
                    prediction["repair_proposal"] = proposal
            quality = prediction.get("quality", {"passed": False, "reasons": ["quality_metrics_unavailable"]})
            direct_identifiers = prediction.get("image_metadata", {}).get("contains_direct_identifiers", False)
            reasons = list(quality.get("reasons", []))
            if direct_identifiers:
                reasons.append("dicom_contains_direct_identifiers")
            proposal = prediction.get("repair_proposal")
            measurements = prediction["measurements"]
            trajectory = [
                {"step": "plan", "action": "EXECUTE", "protocols": ["HVA", "IMA"],
                 "source": "uploaded_radiograph_policy"},
                {"step": "detect", "action": "landmark_detector",
                 "model_id": prediction["model"]["model_id"]},
            ]
            if proposal and proposal.get("accepted") and quality.get("passed") and not direct_identifiers:
                trajectory.extend([
                    {"step": "verify", "action": "REPAIR",
                     "confidence": proposal["confidence"], "threshold": proposal["threshold"]},
                    {"step": "repair", "action": "REPAIR", "status": "executed",
                     "tool": proposal["action"], "maximum_steps": proposal["maximum_steps"]},
                    {"step": "decision", "action": "STOP",
                     "reason": "post_repair_human_review_required"},
                ])
                measurements = proposal["measurements"]
                reason = "post_repair_human_review_required"
            else:
                rejection = reasons[0] if reasons else (
                    proposal.get("policy_rejection_reason", "repair_verifier_rejected_proposal") if proposal
                    else "first_pass_live_model_requires_review"
                )
                trajectory.extend([
                    {"step": "verify", "action": "STOP", "reason": rejection},
                    {"step": "decision", "action": "STOP", "reason": rejection},
                ])
                reason = rejection
            return PipelineOutcome("needs_review", {
                "artifact": artifact,
                "measurements": measurements,
                "initial_measurements": prediction["measurements"],
                "repair_proposal": proposal,
                "agent_trajectory": trajectory,
                "model": prediction["model"],
                "quality": quality,
                "image_metadata": prediction.get("image_metadata", {}),
                "routing": {
                    "decision": "STOP",
                    "reason": reason,
                    "all_reasons": reasons or [reason],
                },
                "provenance": {
                    "mode": "live_image_inference",
                    "live_encoder_inference": True,
                    "clinical_use": False,
                },
                "trace_id": job.payload.get("_trace_id"),
            })
        raise ValueError(f"unknown job type: {job.job_type}")
