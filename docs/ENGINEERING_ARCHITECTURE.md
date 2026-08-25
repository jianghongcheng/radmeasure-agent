# Engineering architecture

```text
Browser / MCP / API client
          |
      FastAPI API
          |
  durable idempotent jobs ---- content-addressed artifact store
          |
     atomic claim
          |
 independent workers
          |
 validation -> remote inference service -> geometry -> retrieval -> review routing
          |
 completed | needs_review | retry -> failed
```

## Implemented guarantees

- A unique idempotency key prevents duplicate jobs.
- SQLite WAL and `BEGIN IMMEDIATE` provide an atomic single-job claim across
  API and worker processes in the reference deployment.
- Only a running job can enter a successful terminal state.
- Operational failures retry up to a bounded attempt budget; invalid jobs fail
  immediately.
- Uploaded bytes are size bounded, media-type allow-listed, named by SHA-256,
  and atomically moved into the artifact store. Repeated content is deduplicated.
- Uploaded images never receive fabricated predictions. Until a live inference
  adapter is configured, they terminate as `needs_review` with an explicit
  reason.
- Evaluation replay jobs route geometry disagreements to human review.

## Deployment adapters

Unit tests use SQLite and a local artifact directory. The Docker deployment uses:

- PostgreSQL with `FOR UPDATE SKIP LOCKED` for concurrent worker claims;
- two independently running workers with bounded leases;
- S3-compatible MinIO with SHA-256 keys and deduplicated uploads.

Still pending: a separately scaled GPU inference service, authentication,
managed queue backpressure, and OpenTelemetry/Prometheus operations.

## Integration evidence

- 50 concurrent submissions produced 50 unique terminal jobs at one attempt each.
- Two workers processed 27 and 23 jobs with no duplicate claim.
- Every job recorded `submitted → claimed → needs_review`.
- Repeating identical upload bytes produced one MinIO object and then a dedupe hit.
- A simulated crashed worker produced
  `submitted → claimed → lease_expired → claimed → needs_review` and completed on
  attempt two after recovery by the second worker.
- The independently deployed inference service exposes version/hash/readiness
  metadata and internal token authentication. With the service stopped, a test
  job exhausted exactly three bounded attempts and entered `failed` without a
  fabricated fallback.
