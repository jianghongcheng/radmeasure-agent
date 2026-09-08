# Architecture

RadMeasure implements a bounded medical workflow in ordinary Python, not
LangGraph. A planner selects registered actions; it does not generate unrestricted
executable code or authoritative clinical measurements.

## Source map

All files below are relative to `src/geomed_copilot/`.

| Component | Files |
| --- | --- |
| Task orchestration and review | `pipeline.py` |
| Planning and authorization | `planner.py`, `protocols.py` |
| Contracts and geometry | `measurement_contract.py`, `geometry.py` |
| KEEP / REPAIR / STOP controller | `agent_controller.py` |
| Image inference and decoding | `inference.py`, `inference_api.py`, `imaging.py` |
| Persisted jobs and workers | `jobs.py`, `worker.py` |
| Records and replay | `execution_record.py`, `replay.py` |
| HTTP, dashboard, and MCP | `api.py`, `dashboard.py`, `mcp_server.py` |

## Measurement boundaries

The pipeline accepts registered-case evaluation and uploaded-radiograph jobs.
Contracts reject missing or duplicate measurements, invalid numbers, and
inconsistent geometry evidence. The controller checks repair eligibility and
budget before retaining or stopping a result.

Geometric consistency is not anatomical correctness: a plausible angle can come
from an incorrectly located axis. Synthetic geometry and saved predictions must
not be presented as live image-model validation.

Uploads use a separate inference path and always require review, including after
a proposed repair. A completed synthetic evaluation is not clinical approval.
An unavailable inference adapter does not produce a fabricated upload result.

## Service boundaries

The local demo persists jobs in SQLite. Worker claims, leases, retries, and
execution records support inspectable processing. PostgreSQL and object storage
adapters are optional; their presence is not proof of a tested deployment.

API-key roles distinguish reading, submission, and admin review. Review can
approve, reject, or record corrected measurements; it is not clinical certification.
Traces connect jobs and events. Replays preserve lineage and check contract
compatibility, but rerunning an upload with changed weights is not deterministic
replay of its former prediction.

MCP direct analysis does not automatically inherit the durable API queue.
No prospective clinical validation, production-user impact, or sustained-load
service guarantee is demonstrated.
