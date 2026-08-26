# RadMeasure

**A bounded agent runtime for reliable tool execution, evaluated on
radiographic measurement and SQL repair.**

RadMeasure combines LLM planning, policy-gated tools, deterministic execution,
verification, replay, and frozen evaluations. Radiographic measurement is the
primary safety-critical environment; SQL repair tests whether the same runtime
and reliability claims transfer beyond medical imaging.

> **Naming note.** `RadMeasure` is the project name. `geomed_copilot` is the
> Python package name, and the service, MCP server, and `GEOMED_*` environment
> variables inherit that prefix. They refer to the same system. Planner
> configuration uses the `RADMEASURE_PLANNER_*` variables; data and artifact
> paths use `GEOMED_*`.

### Highlights

- Deterministic geometry instead of LLM-generated measurements
- LLM planner with registry- and policy-constrained tool execution
- `KEEP / REPAIR / STOP` verification with mandatory human-review paths
- Caught patient-level leakage in the public HVAngleEst release split and
  rebuilt a patient-disjoint manifest ([details](#data-integrity))
- Negative results reported as design evidence, not omitted
  ([details](#planner-reliability-evaluation))
- Trace-based replay, artifact lineage, and per-tool observability
- FastAPI, MCP, PostgreSQL workers, MinIO, Docker Compose, and GitHub Actions

### Measured results

On a frozen 36-case adversarial SQL-repair suite using local Qwen3-8B,
policy-controlled execution increased successful tasks from **19/36 to 30/36**
and blocked **all six unsafe actions proposed by the model**. The six remaining
failures were five unusable repair proposals and one output-contract rejection,
not policy bypasses.

| Configuration | Successful tasks | Unsafe actions executed |
|---|---:|---:|
| LLM only | 19/36 | 6/36 |
| Policy + verifier | **30/36** | **0/36** |

The self-hosted Ollama/Qwen3-8B run measured **443 ms p50** and **594 ms p95
planner-generation latency**. One cold-start request took 16.7 s. The single
planner call averaged 116 prompt and 28 completion tokens. This benchmark uses
a bounded single-action runtime, so these are not multi-turn agent latency or
full-trajectory token-cost claims.
This is a small, frozen agent-reliability benchmark,
not a SQL leaderboard or production-traffic claim. See the
[raw result artifact](outputs/portfolio/sql_harness_ablation_qwen3_8b.json),
[benchmark cases](data/benchmarks/sql_repair_v1.json), and
[evaluation script](scripts/evaluate_sql_harness_ablation.py).

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

## What is reusable

The runtime is evaluated in two deliberately different environments:
radiographic measurement and executable SQL repair. Its reusable systems contribution is a
bounded agent pattern in which an LLM proposes a registered action, a policy
layer authorizes or rejects it, deterministic tools execute it, and a verifier
chooses `KEEP`, `REPAIR`, or `STOP`. The SQL suite provides bounded cross-domain
evidence for the runtime and its safety policy; it does not establish broad
domain independence.

> **LLM proposes. Policy authorizes. Deterministic tools execute. Verifier decides.**

<!-- Keep the design principle and safety disclaimer as separate callouts. -->

> Research prototype only. It is not a medical device and must not be used for
> diagnosis or patient care.

## Engineering evidence

| Check | Result |
|---|---:|
| Automated tests | 89 passing, 1 skipped locally |
| Controller policy suite | 12/12 expected decisions, 0 unsafe actions |
| Planner safety suite | 24 frozen cases |
| SQL harness suite | 36 frozen cases, Qwen3-8B |
| Public CI | GitHub Actions |

Model metrics, artifact hashes, split qualifications, and retrieval evaluations
are reported separately in [Evaluation results](docs/PORTFOLIO_RESULTS.md).

## Workflow

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
requests fail closed with `STOP`. The deterministic fallback keeps the complete
workflow runnable without a hosted model.

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

## Planner reliability evaluation

A frozen 24-case benchmark compares a fixed workflow, the deterministic
registry planner, and local `qwen3:8b` on supported, unsupported, missing-input,
and prompt-injection requests.

| Planner | Action accuracy | Unsafe action rate | Valid JSON |
|---|---:|---:|---:|
| Fixed HVA+IMA workflow | 50.0% | 50.0% | 100% |
| Registry/rule planner | **75.0%** | **16.7%** | 100% |
| Qwen3-8B + schema only | 62.5% | 37.5% | 100% |

The negative result is intentional evidence, not a hidden model claim. Qwen3-8B
produces structured plans but does not beat the rule planner, so the LLM remains
an optional intent proposer behind deterministic authorization. A separate
12-case policy-unit suite verifies all expected controller decisions with zero
unsafe actions and deterministic replay. These are agent-reliability tests, not
clinical performance claims. Raw results live under `outputs/portfolio/`.

### Cross-domain SQL harness

The same bounded runtime also runs against a disposable SQLite environment.

The ablation generates each model response **once**, then replays that identical
response through all six harness configurations. This isolates the contribution
of schema validation, registry checks, policy enforcement, and verification
without confounding the comparison with generation randomness.

| Configuration | Task success | Unsafe action | Invalid action | STOP rate | Avg tool calls / success |
|---|---:|---:|---:|---:|---:|
| LLM only | 52.8% | 16.7% | 0% | 41.7% | 0.47 |
| + Schema | 52.8% | 16.7% | 0% | 41.7% | 0.47 |
| + Registry | 52.8% | 16.7% | 0% | 41.7% | 0.47 |
| + Policy | **83.3%** | **0%** | 0% | 61.1% | 0.47 |
| + Verifier | 66.7% | 16.7% | 0% | 44.4% | 0.58 |
| + Policy + Verifier | **83.3%** | **0%** | 0% | 61.1% | 0.47 |

**Reading the tool-call column.** The average is below 1.0 by design: 16 of the
36 cases are expected-`STOP` tasks that should invoke no tool at all. Reported
by task class, all 8 expected-`KEEP` and all 12 expected-`REPAIR` tasks invoked
the registered SQL tool exactly once, and all 16 expected-`STOP` tasks invoked
none. The policy blocked unsafe actions before execution.

**Why `+ Policy` and `+ Policy + Verifier` are identical.** Once policy gating
is active the verifier changes no task outcome on this suite—the same 30 tasks
succeed either way. It does change how one failure is classified: a contract
rejection rather than a failed execution. Verifier-only, without policy,
improves correctness but leaves the unsafe-action rate unchanged. Policy is the
component that eliminates unsafe execution; the verifier earns its place on the
radiographic path, not on this suite.

Schema and registry validation are still necessary execution boundaries, but all
model outputs happened to satisfy them in this suite.

Qwen3-8B averages 116 prompt tokens and 28 completion tokens per planner call,
with 443 ms p50 and 594 ms p95 planner-generation latency; one cold-start
request took 16.7 s. The harness permits at most one execution action per task
and is not a multi-turn latency benchmark.

After policy gating, the six remaining failures shift away from unsafe action:
five are repair proposals that fail during execution and one is a verifier
contract rejection. No unsafe proposal passes the policy.

This is deliberately a small frozen engineering benchmark, not evidence of SQL SOTA.
It demonstrates that the runtime abstraction and safety result transfer beyond
radiography. See `outputs/portfolio/sql_harness_ablation_qwen3_8b.json`.

Reproduce the ablation with a local Ollama-compatible Qwen3-8B endpoint:

```bash
ollama pull qwen3:8b
python scripts/evaluate_sql_harness_ablation.py \
  --model qwen3:8b \
  --base-url http://127.0.0.1:11434 \
  --output outputs/portfolio/sql_harness_ablation_qwen3_8b.json
```

Full measurement conditions and percentile definitions are recorded in
[Evaluation results](docs/PORTFOLIO_RESULTS.md#measurement-protocol).

### Independent repair proposal

**This proposal stack is deliberately not a strong measurement model, and the
numbers below are reported as a negative result.** They are the reason the
cross-model safety gate exists.

Live uploads keep the supervised ResNet angle model as the primary predictor.
An independently trained HRNet landmark detector, residual RepairMLP, and
learned verifier may propose a one-step geometric edit. The policy accepts that
proposal only when both its verifier passes and its HVA/IMA outputs agree with
the primary model within registered bounds; otherwise it records the attempted
repair and returns `STOP` for review.

On 243 patient-disjoint cases the raw HRNet errors are very large
(HVA 52.61°, IMA 71.96°). Gated one-step repair reduces these to 28.92° and
24.67° with 83.1% coverage and 1.65% any-measurement harm, but the stack remains
unsuitable as the final predictor — which is exactly what the gate is there to
enforce. See `outputs/research/hrnet_geometry_repair.json`.

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
