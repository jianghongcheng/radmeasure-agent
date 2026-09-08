from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from .jobs import Job
from .inference_client import InferenceClient
from .agent_controller import MeasurementAgentController
from .planner import ConstrainedMeasurementPlanner, planner_from_env
from .protocols import ProtocolRegistry
from .tools import RadMeasureTools
from .execution_record import ExecutionRecord
from .measurement_contract import MeasurementContract


@dataclass(frozen=True)
class PipelineOutcome:
    status: str
    result: dict[str, Any]


class JobPipeline:
    """Registered radiographic measurement and uploaded-image review only."""

    def __init__(self, tools: RadMeasureTools,
                 inference_client: InferenceClient | None = None,
                 planner: ConstrainedMeasurementPlanner | None = None,
                 controller: MeasurementAgentController | None = None) -> None:
        self.tools = tools
        self.inference_client = inference_client
        registry = controller.registry if controller else ProtocolRegistry()
        self.registry = registry
        self.planner = planner or planner_from_env(registry)
        self.controller = controller or MeasurementAgentController(registry)

    @staticmethod
    def _record(job: Job, contract: MeasurementContract, repair_limit: int) -> ExecutionRecord:
        record = ExecutionRecord(contract.snapshot(), job.attempts, job.max_attempts, repair_limit)
        record.record("collect", {"image_id": job.payload.get("image_id"),
                                  "artifact": job.payload.get("artifact"),
                                  "trace_id": job.payload.get("_trace_id"),
                                  "replay_of_job_id": job.payload.get("_replay_of_job_id")})
        expected = job.payload.get("_execution_contract_sha256")
        if expected and expected != record.contract.sha256:
            raise ValueError("replay contract differs from current execution contract")
        return record

    @staticmethod
    def _finish(record: ExecutionRecord, status: str, result: dict[str, Any],
                repairs: int = 0) -> PipelineOutcome:
        routing = result["routing"]
        # Domain STOP means review, never automatic clinical approval.
        decision = "KEEP" if status == "completed" else "STOP"
        record.record("verify", {"routing": routing,
                                  "contract_verification": result.get("contract_verification"),
                                  "agent_trajectory": result.get("agent_trajectory", []),
                                  "provenance": result.get("provenance", {}),
                                  "inference_service": result.get("inference_service")})
        result["execution_record"] = record.finish(decision, routing["reason"], repairs)
        return PipelineOutcome(status, result)

    def run(self, job: Job) -> PipelineOutcome:
        if job.job_type not in {"evaluation_analysis", "uploaded_radiograph"}:
            raise ValueError("RadMeasure supports radiographic measurement jobs only")
        if job.job_type == "evaluation_analysis":
            question = str(job.payload.get("question") or "Measure HVA and IMA with supporting evidence.")
            plan = self.planner.plan(question)
            contract = MeasurementContract.from_registry(self.registry, plan.protocols)
            record = self._record(job, contract, self.controller.max_repairs)
            record.record("collect", {"authorized_plan": plan.to_dict()})
            if plan.action == "STOP":
                return self._finish(record, "needs_review", {
                    "measurements": [],
                    "agent_plan": plan.to_dict(),
                    "agent_trajectory": [
                        {"step": "plan", **plan.to_dict()},
                        {"step": "decision", "action": "STOP", "reason": plan.reason},
                    ],
                    "routing": {"decision": "STOP", "reason": plan.reason},
                    "trace_id": job.payload.get("_trace_id"),
                })
            record.record("act", {"tool": "analyze_radiograph", "protocols": plan.protocols})
            result = self.tools.analyze_radiograph(
                image_id=str(job.payload["image_id"]),
                question=question,
                top_k=int(job.payload.get("top_k", 3)),
            )
            if self.inference_client:
                remote = self.inference_client.predict(str(job.payload["image_id"]))
                local_errors = contract.validate(result.get("measurements"), geometry=True)
                remote_errors = contract.validate(remote.get("measurements"), geometry=False)
                if local_errors or remote_errors:
                    disagreement = True
                else:
                    local = {item["name"]: item["predicted_degrees"] for item in result["measurements"]}
                    disagreement = any(
                        abs(local[p.name] - remote["measurements"][p.name]) > 0.001
                        for p in contract.protocols
                    )
                result["inference_service"] = {
                    "model": remote["model"],
                    "agreement_with_workflow": not disagreement,
                    "contract_errors": list(local_errors + remote_errors),
                }
            else:
                disagreement = False
            record.record("collect", {"measurements": result.get("measurements"),
                                      "provenance": result.get("provenance", {}),
                                      "tool_traces": result.get("traces", [])})
            agent = self.controller.execute(plan, result, contract)
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
            return self._finish(record, "needs_review" if review else "completed", result,
                                agent.repair_attempts)
        if job.job_type == "uploaded_radiograph":
            contract = MeasurementContract.from_registry(self.registry, ("HVA", "IMA"),
                                                         mandatory_review=True)
            record = self._record(job, contract, 1)
            artifact = job.payload["artifact"]
            if not self.inference_client:
                return self._finish(record, "needs_review", {
                    "artifact": artifact,
                    "routing": {"decision": "human_review", "reason": "live_inference_adapter_unavailable"},
                    "provenance": {"mode": "ingested_without_live_inference", "live_encoder_inference": False, "clinical_use": False},
                    "trace_id": job.payload.get("_trace_id"),
                })
            record.record("act", {"tool": "predict_artifact", "artifact_sha256": artifact["sha256"]})
            prediction = self.inference_client.predict_artifact(
                image_id=artifact["sha256"], artifact_uri=artifact["path"],
                media_type=artifact.get("media_type", "image/jpeg"),
            )
            record.record("collect", {"model": prediction.get("model"),
                                      "measurements": prediction.get("measurements"),
                                      "quality": prediction.get("quality")})
            initial_errors = contract.validate(prediction.get("measurements"), geometry=False)
            if initial_errors:
                return self._finish(record, "needs_review", {
                    "artifact": artifact, "measurements": {},
                    "contract_verification": {"passed": False, "errors": initial_errors},
                    "routing": {"decision": "STOP", "reason": initial_errors[0]},
                    "model": prediction.get("model"), "trace_id": job.payload.get("_trace_id"),
                })
            repair_model_id = os.environ.get("RADMEASURE_REPAIR_MODEL_ID", "").strip()
            if repair_model_id:
                candidate = self.inference_client.predict_artifact(
                    image_id=artifact["sha256"], artifact_uri=artifact["path"],
                    media_type=artifact.get("media_type", "image/jpeg"),
                    model_id=repair_model_id,
                )
                record.record("collect", {"repair_model": candidate.get("model"),
                                          "repair_proposal": candidate.get("repair_proposal")})
                proposal = candidate.get("repair_proposal")
                if proposal:
                    proposal = dict(proposal)
                    proposal["model"] = candidate["model"]
                    repair_errors = contract.validate(proposal.get("measurements"), geometry=False)
                    proposal["cross_model_discrepancy"] = {} if repair_errors else {
                        name: abs(float(proposal["measurements"][name]) - float(prediction["measurements"][name]))
                        for name in ("HVA", "IMA")
                    }
                    proposal["accepted"] = not repair_errors and bool(proposal.get("accepted")) and all(
                        proposal["cross_model_discrepancy"][name] <= limit
                        for name, limit in {"HVA": 5.0, "IMA": 3.0}.items()
                    )
                    if not proposal["accepted"]:
                        proposal["policy_rejection_reason"] = repair_errors[0] if repair_errors else "cross_model_disagreement"
                    prediction["repair_proposal"] = proposal
            quality = prediction.get("quality", {"passed": False, "reasons": ["quality_metrics_unavailable"]})
            direct_identifiers = prediction.get("image_metadata", {}).get("contains_direct_identifiers", False)
            reasons = list(quality.get("reasons", []))
            if direct_identifiers:
                reasons.append("dicom_contains_direct_identifiers")
            proposal = prediction.get("repair_proposal")
            if proposal:
                proposal = dict(proposal)
                repair_errors = contract.validate(proposal.get("measurements"), geometry=False)
                if repair_errors:
                    proposal["accepted"] = False
                    proposal["policy_rejection_reason"] = repair_errors[0]
                    record.record("collect", {"rejected_repair": proposal})
                    proposal.pop("measurements", None)
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
            return self._finish(record, "needs_review", {
                "artifact": artifact,
                "measurements": measurements,
                "initial_measurements": prediction["measurements"],
                "repair_proposal": proposal,
                "contract_verification": {"passed": True, "errors": [],
                                          "scope": "measurement_structure_only"},
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
            }, repairs=int(any(step.get("step") == "repair" and step.get("status") == "executed"
                               for step in trajectory)))
        raise ValueError(f"unknown job type: {job.job_type}")
