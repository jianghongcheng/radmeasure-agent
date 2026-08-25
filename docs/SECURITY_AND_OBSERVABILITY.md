# Security and observability

## API-key roles

`GEOMED_API_KEYS` maps secret keys to principal metadata. Keys are compared with
`hmac.compare_digest` and are never written to jobs, events, metrics, or logs.

- `viewer`: read jobs, audit events, operational counts, and metrics;
- `operator`: viewer permissions plus submit analyses and uploads;
- `admin`: reserved for future user and key administration.

The values in `compose.yaml` are local demo credentials. A real deployment must
inject rotated keys from a secret manager and terminate TLS at the ingress.

## Correlation

Clients may send `X-Request-ID`; otherwise the API generates one. It is returned
in the response header and persisted as the job trace ID. The same identifier
appears in the submitted event, worker log, result, and terminal log.

Viewer-authorized clients can query `GET /v1/traces/{trace_id}` to retrieve the
persisted jobs and audit events for an execution lineage. Operators can request
`POST /v1/jobs/{job_id}/replay` only after the source job reaches a terminal
state. Replays receive a new trace ID and retain immutable parent/root lineage.
Evaluation jobs replay locked artifacts; uploaded images reuse the same
content-addressed input with the currently deployed model, which is reported as
a weaker guarantee rather than mislabeled deterministic replay.

## Metrics

The viewer-protected `/metrics` endpoint exposes Prometheus text format:

- HTTP request count by method, normalized route, and status;
- HTTP duration histogram;
- current job counts by status.

Dynamic IDs are normalized to `{job_id}` to avoid high-cardinality labels.
API keys, questions, image IDs, and artifact names are never metric labels.

## Structured logs

Application logs are JSON with timestamp, level, logger, message, trace ID, job
ID, worker ID, and event type. Integration testing confirmed trace continuity
across API acceptance, PostgreSQL, worker claim, result, and audit history.
