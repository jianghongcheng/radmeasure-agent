# RadMeasure

**A bounded agent runtime for reliable tool execution, evaluated on
radiographic measurement and SQL repair.**

RadMeasure combines LLM planning, policy-gated tools, deterministic execution,
verification, replay, and frozen evaluations. Radiographic measurement is the
primary safety-critical workload. The same execution runtime also supports SQL
repair, providing a second tool domain for stress-testing policy enforcement
and replay.

## What it does

RadMeasure turns an authorized measurement request into a durable, auditable
job. An LLM may propose an intent, but it cannot execute arbitrary output. The
runtime validates the plan against registered protocols, applies policy before
tool execution, runs deterministic tools behind a service boundary, verifies
the result, and routes uncertain cases to review.

```text
API / MCP client
       |
       v
FastAPI control plane -- API-key roles -- idempotency
       |
       v
PostgreSQL job queue -- atomic claim -- bounded retry / lease recovery
       |
       v
Worker pool -- registered tools -- isolated inference service
       |
       v
Verifier -- KEEP / REPAIR / STOP -- human review
       |
       v
MinIO artifacts -- audit lineage -- replay -- Prometheus metrics
```

### Operational guarantees in the reference deployment

- **Durable execution:** idempotent submission, PostgreSQL-backed job state,
  atomic concurrent-worker claims, bounded retries, leases, and crash recovery.
- **Constrained action boundary:** schema and registry validation plus
  policy authorization before any tool invocation; unsupported actions fail
  closed.
- **Service isolation:** workers call a separately deployed inference service;
  failures exhaust a bounded retry budget without fabricated fallbacks.
- **Artifact integrity:** content-addressed, deduplicated MinIO storage with
  model version, backend, and artifact-hash provenance.
- **Access and review controls:** API-key roles, internal service
  authentication, and mandatory review paths for uncertain medical outputs.
- **Operations:** correlated structured logs, protected Prometheus metrics,
  trace lineage, parent-linked replay, Docker Compose, and public CI.

> **LLM proposes. Policy authorizes. Deterministic tools execute. Verifier decides.**

> Research prototype only. It is not a medical device and must not be used for
> diagnosis or patient care.

## Quick start

Run the dependency-light offline workflow:

```bash
git clone https://github.com/jianghongcheng/radmeasure-agent.git
cd radmeasure-agent
pip install -e .
radmeasure --question "Measure and verify the hallux valgus angle"
```

The command uses bundled synthetic geometry and labels its provenance
accordingly; no model checkpoint or medical data is required. For the complete
service stack and live-model setup, see [Model serving](docs/MODEL_SERVING.md)
and the [five-minute demo](#five-minute-demo).

Run the complete reference deployment:

```bash
cp .env.example .env
docker compose up --build
```

Compose starts the FastAPI control plane, PostgreSQL job store, concurrent
workers, MinIO artifact storage, isolated inference service, and local review
UI. The credentials in `.env.example` are development-only and must be replaced
before any shared deployment. See
[Security and observability](docs/SECURITY_AND_OBSERVABILITY.md) and
[Engineering architecture](docs/ENGINEERING_ARCHITECTURE.md).

## Safety and execution model

The runtime uses the same bounded execution path for radiographic measurement
and SQL repair: an LLM proposes a registered action, a policy layer authorizes
or rejects it, deterministic tools execute it, and a verifier chooses `KEEP`,
`REPAIR`, or `STOP`. The SQL workload stress-tests this execution and
authorization path outside the radiography stack; it does not imply that every
domain is supported without a registered tool and policy.

## Engineering evidence

| Check | Result |
|---|---:|
| Automated tests | 106 passing locally |
| Controller policy suite | 12/12 expected decisions, 0 unsafe actions |
| Planner safety suite | 24 frozen cases |
| SQL harness v2 | 120 frozen cases, 5 schemas, 24 failure-family clusters |
| SQL confirmatory v3 | 108 frozen cases, 6 unseen schemas, 18 unseen failure families |
| Harbor isolated replay | v3: 98/108 success, 25/25 unsafe proposals blocked |
| Public CI | GitHub Actions |

Model metrics, artifact hashes, split qualifications, and retrieval evaluations
are reported separately in [Evaluation results](docs/PORTFOLIO_RESULTS.md).

## Authorization path

```text
User goal
    ↓
LLM planner
    ↓
Schema + registry validation
    ↓
Policy authorization
    ↓
Deterministic tool execution
    ↓
Verifier
 ┌──┴──────────┐
KEEP      REPAIR / STOP
    ↓
Trace + replay + evals
```

The planner may use an OpenAI-compatible endpoint or local Ollama, but
its JSON output is never executed directly. Every protocol, tool, and repair
action is checked against the registry. Invalid model output and unsupported
requests fail closed with `STOP`. The runtime also supports an explicitly
selected deterministic planner, allowing the complete workflow to run without
a hosted LLM.

```bash
export RADMEASURE_PLANNER_BASE_URL=http://127.0.0.1:8080/v1
export RADMEASURE_PLANNER_MODEL=your-instruct-model
export RADMEASURE_PLANNER_API_KEY=<your-planner-key>
```

For the benchmarked local 8B configuration:

```bash
export RADMEASURE_PLANNER_PROVIDER=ollama
export RADMEASURE_PLANNER_BASE_URL=http://127.0.0.1:11434
export RADMEASURE_PLANNER_MODEL=qwen3:8b
```

Inspect the executable boundary:

```bash
curl http://127.0.0.1:8000/v1/protocols
curl -X POST http://127.0.0.1:8000/v1/plan \
  -H 'content-type: application/json' -H 'x-api-key: <your-viewer-key>' \
  --data '{"request":"Measure hallux valgus angle"}'
```

API keys are read from the environment; see `.env.example` for the variables
the Compose demo expects. No usable credentials are committed to this
repository.

## Measured reliability

The engineering claims above are exercised with frozen, reproducible suites;
they are not production-traffic or clinical-validation claims. For the SQL
ablation, Qwen3-8B generated each proposal once and the evaluator replayed that
identical proposal through every configuration.

| Suite and configuration | Successful tasks | Unsafe actions accepted | Incorrect outputs accepted |
|---|---:|---:|---:|
| v2 development: LLM only | 94/120 | 19/120 | 5/120 |
| v2 development: policy + verifier | **113/120** | **0/120** | **0/120** |
| v3 held-out: LLM only | 73/108 | 25/108 | 6/108 |
| v3 held-out: policy + verifier | **98/108** | **0/108** | **0/108** |

Policy eliminates unsafe execution; verification eliminates acceptance of
incorrect query outputs. Without changing the prompt, policy, verifier, or
evaluation semantics, the separately frozen v3 suite improved by **23.22
points** under an 18-cluster bootstrap (95% CI **+7.41 to +40.74**). Its six
schemas and failure templates do not occur in v2. The complete v3 suite also
runs in [Harbor](https://github.com/harbor-framework/harbor) with an isolated
agent container, hidden database fixtures, a separate verifier, and no network
access; its frozen replay reproduced **98/108** success and **0/25** unsafe
executions. See the
[detailed ablation and measurement protocol](docs/PORTFOLIO_RESULTS.md#cross-domain-agent-reliability),
[audited v2 result](outputs/portfolio/sql_harness_v2_qwen3_8b_audited.json),
[confirmatory v3 result](outputs/portfolio/sql_harness_v3_qwen3_8b_confirmatory.json),
and [Harbor evaluation](docs/HARBOR_EVALUATION.md).

### Planner authorization pilot

On a separate frozen 24-case safety pilot, the registry planner produced 18/24
correct and 4/24 unsafe actions versus 15/24 and 9/24 for Qwen3-8B. This
directional result—not a significance claim—keeps the LLM as an optional intent
proposer rather than an execution authority. Detailed cases and limitations are
reported in [Evaluation results](docs/PORTFOLIO_RESULTS.md#planner-authorization-pilot).

### Medical repair safety gate

An independent HRNet/RepairMLP stack may propose a one-step edit, but the policy
accepts it only when verification and cross-model agreement pass registered
bounds; otherwise the case routes to review. On 243 patient-disjoint cases this
gate reduced the weak proposal model's error while limiting harm, but remained
unsuitable as the final predictor. See the
[detailed medical evaluation](docs/PORTFOLIO_RESULTS.md#medical-repair-safety-gate)
and `outputs/research/hrnet_geometry_repair.json`.

The live workflow also supports:

```text
Orthanc / direct upload
        → DICOM direct-identifier removal
        → content-addressed MinIO storage
        → image quality and OOD gates
        → live HVA/IMA inference
        → mandatory human correction/approval
        → versioned structured measurement report
```

Evaluation jobs remain restricted to locked image IDs. Uploaded JPEG/PNG
radiographs use the live ResNet50 HVA/IMA adapter and always route to human
review. Every response distinguishes replay from live inference under
`provenance`; neither path is approved for clinical use.

## Repository contents

> **Naming note.** `RadMeasure` is the project name. `geomed_copilot` is the
> Python package name, and the service, MCP server, and `GEOMED_*` environment
> variables inherit that prefix. They refer to the same system. Planner
> configuration uses the `RADMEASURE_PLANNER_*` variables; data and artifact
> paths use `GEOMED_*`.

- `artifact_predictor.py`: hash-verified three-seed axis ensemble replay;
- `geometry.py`: independent acute-angle reconstruction;
- `protocols.py`: allow-listed measurement protocols, tools, and repair policy;
- `planner.py`: OpenAI-compatible planner with registry validation and fail-closed fallback;
- `agent_controller.py`: bounded `KEEP / REPAIR / STOP` execution loop;
- `repair_inference.py`: independent HRNet + residual repair + verifier proposal service;
- `retrieval.py`: evidence and measurement-space case retrieval;
- `production.py`: end-to-end application service using real locked artifacts;
- `api.py`: FastAPI planning, protocols, durable analysis, trace lookup, and replay;
- `evaluation.py` and `scripts/evaluate_*`: measurement, retrieval and evidence evals;
- `scripts/prepare_hvangleest.py`: patient-level splitting with identifiers removed;
- `data/evidence/catalog.json`: traceable curated evidence catalog.

## Data integrity

The original HVAngleEst release split has no image overlap but does have patient
overlap (85 train/validation, 42 train/test, and 9 validation/test patients).
Models evaluated on that split can therefore see the same patient in training
and test. The preparation script creates a separate patient-disjoint 1,598-foot
manifest:

| Split | Samples | Patients |
|---|---:|---:|
| Train | 1,121 | 756 |
| Validation | 234 | 162 |
| Test | 243 | 162 |

No patient IDs or image bytes are copied into the project. One out-of-range
source box is clipped and recorded in the audit. HVA/IMA landmark reconstruction
matches all 1,598 released targets within 0.1°.

## Run the verified application

```bash
PYTHONPATH=src python scripts/run_locked_demo.py \
  --predictions /path/to/line_predictions_medimageinsight.csv \
  --annotations /path/to/HVAngleEst/datasets.csv \
  --split-manifest /path/to/hvangle_results.json \
  --evidence-catalog data/evidence/catalog.json \
  --image-id IMG000005.jpg
```

Run tests:

```bash
python -m pytest -q
```

Run the API after installing the optional dependencies:

```bash
pip install -e '.[api]'
export GEOMED_PREDICTIONS=/path/to/line_predictions_medimageinsight.csv
export GEOMED_ANNOTATIONS=/path/to/HVAngleEst/datasets.csv
export GEOMED_SPLIT_MANIFEST=/path/to/hvangle_results.json
export GEOMED_EVIDENCE_CATALOG="$PWD/data/evidence/catalog.json"
uvicorn geomed_copilot.api:create_app --factory
```

Discover the backend contract before calling it:

```bash
curl http://127.0.0.1:8000/v1/capabilities
```

## MCP agent tools

RadMeasure exposes the same honest application boundary through a standard
Python MCP server (packaged as `geomed-mcp`). The current server accepts only
identifiers from the configured, hash-locked artifact; it does not claim live
image inference.

```bash
pip install -e .
export GEOMED_PREDICTIONS=/path/to/line_predictions_medimageinsight.csv
export GEOMED_ANNOTATIONS=/path/to/HVAngleEst/datasets.csv
export GEOMED_SPLIT_MANIFEST=/path/to/hvangle_results.json
export GEOMED_EVIDENCE_CATALOG="$PWD/data/evidence/catalog.json"
geomed-mcp
```

For a minimal synthetic smoke demo instead of the included evaluation replay:

```bash
GEOMED_DEMO_MODE=1 geomed-mcp
```

Use `demo-foot-001` as the image identifier. The response provenance says
`deterministic_synthetic_demo`; this mode never claims live model inference.

Tools:

- `list_geomed_capabilities`: reports supported measurements, backend mode,
  accepted input, and limitations;
- `analyze_radiograph`: runs geometry verification, similar-case retrieval,
  evidence retrieval, citations, and per-tool traces for a locked case ID.

## Five-minute demo

```bash
docker compose up --build
```

Open `http://localhost:8000` and select one of 176 persisted evaluation cases.
These cases come from a legacy prediction artifact produced under an earlier
split state; when reconciled against the current manifest, they map to 122
training, 24 validation, and 30 test records. They are therefore not presented
as a subset of the newer 243-case patient-disjoint test split, and the dashboard
reports the split alignment as unverified. Inspect predictions, targets,
absolute errors, citations, provenance, and per-tool latency. The response
explicitly reports that live inference is off.

Because most of these cases were seen during training, the displayed errors are
optimistically biased and must not be read as held-out performance. They
demonstrate the workflow, traces, and provenance surface—not measurement
accuracy.

The dashboard submits a durable asynchronous job. The API writes an idempotent
queued record, and a separate worker atomically claims it, runs the workflow,
and stores either `completed`, `needs_review`, or `failed`. PostgreSQL persists
job state, while content-addressed uploads live in S3-compatible MinIO.

Workers call a separately deployed, internally authenticated inference service.
Every result records the model ID, version, backend, artifact hash, readiness,
and whether the service output agrees with downstream verification. The included
registry contains both locked replay and live PyTorch image inference. Compose
mounts the local checkpoint read-only and defaults to CPU; set an appropriate
device and GPU-enabled base image before benchmarking GPU serving.

Run a real uploaded-image job:

```bash
curl -X POST http://localhost:8000/v1/uploads \
  -H 'X-API-Key: <your-operator-key>' \
  -H 'Idempotency-Key: example-upload-001' \
  -F 'file=@/path/to/radiograph.jpg;type=image/jpeg'
```

Dashboard and API keys are supplied through the environment and are scoped to
the Compose demo. Protected API calls use `X-API-Key`. See
`docs/SECURITY_AND_OBSERVABILITY.md` before deploying outside localhost.

The local medical-imaging UI is available at:

- Upload/review dashboard: `http://localhost:8000`
- OHIF DICOM viewer: `http://127.0.0.1:3000`
- Orthanc Explorer/DICOMWeb: `http://127.0.0.1:8042`

Orthanc and OHIF bind only to loopback. The development credentials in
`.env.example` and the permissive local assumptions must be replaced before any
shared deployment.

```bash
PYTHONPATH=src python scripts/portfolio_benchmark.py --iterations 100
```

This reports successful runs, tool success rate, citation presence, and p50/p95
workflow latency. It is an engineering reliability check, not clinical validation.

## Limitations

- Live inference currently supports JPEG/PNG only; DICOM is accepted by storage
  but deliberately rejected by the image adapter until modality/windowing and
  de-identification handling are implemented.
- The live model is internally reproduced on one public dataset split only;
  there is no external, prospective, or clinical validation.
- The image encoder's pooled embedding is weak for angle-neighbor retrieval;
  hybrid retrieval only slightly improves over predicted geometry.
- The evidence evaluation contains five transparent, manually labeled questions.
- No prospective or external clinical validation has been performed.
- Dataset redistribution remains disabled pending a separate license review.

See [Evaluation results](docs/PORTFOLIO_RESULTS.md),
[Security and observability](docs/SECURITY_AND_OBSERVABILITY.md), and
[Engineering architecture](docs/ENGINEERING_ARCHITECTURE.md) for detailed
evaluation scope, deployment assumptions, and failure analysis.

## Author

**Hongcheng Jiang** — Ph.D., Electrical & Computer Engineering,
University of Missouri–Kansas City.

[GitHub](https://github.com/jianghongcheng) ·
[Website](https://jianghongcheng.github.io/) ·
[Google Scholar](https://scholar.google.com/citations?user=NPk5cT0AAAAJ)

RadMeasure originated from work at
[NextTier IT Solutions Consultancy](https://www.nexttiertech.com/) and was later
released publicly with permission.

Released under the [MIT License](LICENSE).
