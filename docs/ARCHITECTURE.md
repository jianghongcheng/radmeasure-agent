# Architecture

RadMeasure implements a bounded medical workflow in Python. A planner selects
registered actions, measurement tools produce outputs, and the controller
checks whether to retain, repair, or route a result for review.

## Source map

All files below are relative to `src/radmeasure/`.

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

## Measurement execution

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

## Jobs and interfaces

The local demo persists jobs in SQLite. Workers claim tasks, retry eligible
failures, and retain execution records. PostgreSQL and object storage adapters
are optional integrations with separate deployment-validation requirements.

API-key roles distinguish reading, submission, and admin review. Review can
approve, reject, or record corrected measurements.
Traces connect jobs and events. Replays preserve lineage and check contract
compatibility, but rerunning an upload with changed weights is not deterministic
replay of its former prediction.

MCP direct analysis runs synchronously; durable job handling belongs to the API
and worker path. Clinical validation and sustained-load operation have not been
established by the local software checks.
