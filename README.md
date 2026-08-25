# RadMeasure

RadMeasure is a verifiable multimodal execution platform for protocol-driven
radiographic measurements. It connects constrained orchestration and image-model
outputs to deterministic geometry, verification policies, durable jobs,
human review, structured traces, and lineage-aware replay.

The architecture is domain-independent; radiographic measurement is its first
safety-critical evaluation environment. Its reusable systems contribution is a
bounded agent pattern in which an LLM proposes a registered action, a policy
layer authorizes or rejects it, deterministic tools execute it, and a verifier
chooses `KEEP`, `REPAIR`, or `STOP`. The current empirical evidence is medical,
so cross-domain generality is an architectural claim rather than a benchmarked
performance claim.

> Research prototype only. It is not a medical device and must not be used for
> diagnosis or patient care.

## Verified results

All model numbers below come from a hash-locked 176-case evaluation artifact;
they are not synthetic demo scores. The artifact no longer aligns with the
current processed split manifests, so patient-disjoint status is not claimed.

| Evaluation | Result |
|---|---:|
| Locked test cases | 176 |
| HVA MAE / within 5° | **3.56° / 75.6%** |
| IMA MAE / within 3° | **2.13° / 75.6%** |
| Similar-case pool | 829 locked training cases |
| Hybrid case Recall@5 | 6.59% |
| Geometry-only case Recall@5 | 6.25% |
| Image-only case Recall@5 | 0.80% |
| Curated evidence Hit@3 / MRR | 1.00 / 1.00 (5 questions) |
| Automated tests | 86 passing, 1 skipped |

### Agent evaluation

A frozen 24-case protocol-planning benchmark compares a fixed workflow, the
deterministic registry planner, and a local `qwen3:8b` planner on supported,
unsupported, missing-input, and prompt-injection requests.

| Planner | Action accuracy | Unsafe action rate | Valid structured output |
|---|---:|---:|---:|
| Fixed HVA+IMA workflow | 50.0% | 50.0% | 100% |
| Registry/rule planner | **75.0%** | **16.7%** | 100% |
| Qwen3-8B + schema only | 62.5% | 37.5% | 100% |

Qwen3-8B uses Ollama's native JSON mode with thinking disabled and averages
585 ms per request on the local RTX 3090. It does **not** beat the rule planner:
the result justifies treating the LLM as an optional intent proposer behind a
deterministic safety policy, not as an autonomous executor. This is the intended
reliability result: the system measures when an LLM should not be trusted and
fails closed instead of equating model fluency with permission to act. See
`outputs/portfolio/planner_baselines_qwen3_8b.json`.

A separate 12-case deterministic policy-unit benchmark exercises `KEEP`, safe
and rejected `REPAIR`, missing proposals, unsupported inputs, and `STOP`. It
achieves 12/12 expected decisions with zero unsafe actions and deterministic
replay. This verifies controller logic only; it is not evidence of clinical
recovery performance. See `outputs/portfolio/agent_decision_policy_v1.json`.

The live image adapter additionally uses a hash-locked supervised ResNet50
checkpoint evaluated on the 120-image HVAngleEst unilateral test split:

| Live model evaluation | Result |
|---|---:|
| HVA MAE / within 5° | **3.12° / 79.2%** |
| IMA MAE / within 5° | **1.50° / 98.3%** |
| CPU throughput (local verification) | **120 images / 10.4 s** |

Its checkpoint SHA-256 is
`120679da9c6d345461d55343bc164cba7c6a0966170691447e72822ed380144a`.

The prediction artifact SHA-256 is
`ad3fea57c9c8a83c9085220d129b89ea6aaff5483d26abcb64dede92dbc1431f`.
See [PORTFOLIO_RESULTS.md](docs/PORTFOLIO_RESULTS.md) for definitions and
limitations.

## Workflow

```text
natural-language measurement request
              |
 constrained planner → protocol registry / safety policy
              |
       registered tool plan
              |
 image-model output → deterministic geometry executor
              |
       measurement validator
              |
        KEEP / REPAIR / STOP
              |
 evidence + report + auditable trajectory / replay
```

The planner may use an OpenAI-compatible endpoint or local Ollama, but
its JSON output is never executed directly. Every protocol, tool, and repair
action is checked against the registry. Invalid model output and unsupported
requests fail closed with `STOP`. The deterministic fallback keeps the complete
workflow runnable without a hosted model.

```bash
export RADMEASURE_PLANNER_BASE_URL=http://127.0.0.1:8080/v1
export RADMEASURE_PLANNER_MODEL=your-instruct-model
export RADMEASURE_PLANNER_API_KEY=optional-local-or-hosted-key
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
  -H 'content-type: application/json' -H 'x-api-key: viewer-local' \
  --data '{"request":"Measure hallux valgus angle"}'
```

### Independent repair proposal

Live uploads keep the supervised ResNet angle model as the primary predictor.
An independently trained HRNet landmark detector, residual RepairMLP, and
learned verifier may propose a one-step geometric edit. The policy accepts that
proposal only when both its verifier passes and its HVA/IMA outputs agree with
the primary model within registered bounds; otherwise it records the attempted
repair and returns `STOP` for review.

The proposal stack is intentionally not described as a strong measurement
model. On 243 patient-disjoint cases its raw HRNet errors are very large
(HVA 52.61°, IMA 71.96°). Gated one-step repair reduces these to 28.92° and
24.67° with 83.1% coverage and 1.65% any-measurement harm, but remains unsuitable
as the final predictor. This negative result is why the cross-model safety gate
exists. See `outputs/research/hrnet_geometry_repair.json`.

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
The preparation script creates a separate patient-disjoint 1,598-foot manifest:

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

GeoMed exposes the same honest application boundary through a standard Python
MCP server. The current server accepts only identifiers from the configured,
hash-locked artifact; it does not claim live image inference.

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

## Five-minute portfolio demo

```bash
docker compose up --build
```

Open `http://localhost:8000`, select one of 176 persisted evaluation cases, and
inspect predictions, targets, absolute errors, citations, provenance, and
per-tool latency. The response explicitly reports that live inference is off
and that current split alignment is unverified.

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
  -H 'X-API-Key: operator-local' \
  -H 'Idempotency-Key: example-upload-001' \
  -F 'file=@/path/to/radiograph.jpg;type=image/jpeg'
```

Local dashboard credentials are `operator-local`; they are deliberately scoped
to the Compose demo. Protected API calls use `X-API-Key`. See
`docs/SECURITY_AND_OBSERVABILITY.md` before deploying outside localhost.

The local medical-imaging UI is available at:

- GeoMed upload/review dashboard: `http://localhost:8000`
- OHIF DICOM viewer: `http://127.0.0.1:3000`
- Orthanc Explorer/DICOMWeb: `http://127.0.0.1:8042`
- Local reviewer key: `reviewer-local`

Orthanc and OHIF bind only to loopback. The included credentials and permissive
local development assumptions must be replaced before any shared deployment.

```bash
PYTHONPATH=src python scripts/portfolio_benchmark.py --iterations 100
```

This reports successful runs, tool success rate, citation presence, and p50/p95
workflow latency. It is an engineering reliability check, not clinical validation.

## Honest limitations

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
