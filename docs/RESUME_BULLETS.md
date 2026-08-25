# Resume-ready project entry

## GeoMed Agent Platform — Python, FastAPI, MCP, Docker, multimodal retrieval

- Built an auditable radiography AI workflow that exposes geometry verification,
  similar-case retrieval, evidence-grounded reporting, and execution traces
  through shared REST and MCP tool interfaces.
- Packaged 176 persisted MedImageInsight evaluation outputs into a reproducible
  replay service with per-case targets, absolute errors, artifact hashes, and
  explicit provenance; reported HVA/IMA MAE of 3.56°/2.13° from the locked
  artifact without claiming live inference or clinical validity.
- Engineered a PostgreSQL-backed asynchronous job platform using `SKIP LOCKED`
  concurrency, atomic worker leases,
  bounded retries, idempotent submission, SHA-256 artifact deduplication,
  human-review routing, and an append-only state audit trail across independent
  API and worker containers; verified 50 concurrent jobs across two workers with
  zero duplicate claims and successful crash-lease recovery.
- Implemented S3-compatible MinIO storage with allow-listed uploads, SHA-256
  content addressing, atomic object creation, and deduplication.
- Added role-based API-key authorization, cross-process trace correlation,
  structured JSON logs, normalized Prometheus metrics, and append-only audit
  history without exposing keys or high-cardinality job identifiers.
- Decoupled model execution into an internally authenticated inference service
  with immutable model version/hash metadata, timeouts, exponential retries,
  circuit breaking, readiness checks, and failure-without-fabricated-fallback
  behavior verified through an outage test.
- Added typed validation, partial-failure handling, citations, per-tool latency,
  25 automated tests, GitHub Actions, and Docker deployment; audited all 176
  geometry reconstructions to less than 0.0005° maximum target delta after
  correcting image aspect ratio.

## Short version

- Developed a Dockerized multimodal AI agent with FastAPI and MCP tools for
  radiographic geometry verification and evidence-grounded retrieval, including
  durable asynchronous workers, artifact provenance, human-review routing, 30
  tests, and reproducible evaluation replay across 176 cases.

## Claim guardrails

Do not write that the current repository performs live image inference, is
clinically validated, or proves patient-disjoint performance. The persisted
evaluation artifact does not align with the current regenerated split manifests;
the service exposes this limitation directly in provenance.
## Live-model version

- Productionized a PyTorch ResNet50 radiograph regression model behind an
  authenticated inference microservice, integrating content-addressed MinIO
  storage, durable PostgreSQL jobs, idempotent FastAPI ingestion, two concurrent
  workers, model/version/hash provenance, and mandatory human-review routing.
- Reproduced HVA/IMA performance on a locked 120-image test split (3.12°/1.50°
  MAE) and validated the full upload-to-inference path with 32 automated tests;
  sustained about 11.5 images/s on local CPU in offline evaluation.
