# Usage

## Local API and dashboard

Install `pip install -e '.[api]'`. Set the same environment in the API and
worker terminals:

```bash
export GEOMED_DEMO_MODE=1
unset GEOMED_EVAL_REPLAY
export GEOMED_JOB_DB=/tmp/radmeasure-demo/jobs.sqlite
export GEOMED_ARTIFACT_ROOT=/tmp/radmeasure-demo/artifacts
export GEOMED_API_KEYS='{"operator-local":{"name":"local operator","role":"operator"},"reviewer-local":{"name":"local reviewer","role":"admin"},"viewer-local":{"name":"local viewer","role":"viewer"}}'
```

Start `uvicorn geomed_copilot.api:app --host 127.0.0.1 --port 8766` in one
terminal and `radmeasure-worker` in the other. Open `http://127.0.0.1:8766`.
These example keys are only for a loopback demo. Use synthetic inputs only.

The offline CLI in the README needs neither terminal service.
MCP direct analysis is separate from the durable API queue.

## Optional LLM planner

Without model configuration, a deterministic registry planner keeps the demo
runnable. To use a separately installed Ollama model:

```bash
export RADMEASURE_PLANNER_PROVIDER=ollama
export RADMEASURE_PLANNER_BASE_URL=http://127.0.0.1:11434
export RADMEASURE_PLANNER_MODEL=qwen3:8b
```

Model output is validated against the registry; it cannot authorize arbitrary
tools or new medical protocols.

## Model service prerequisites

`inference_api.py` exposes model metadata, saved-prediction inference, and
artifact inference. Internal requests require `GEOMED_INFERENCE_TOKEN`.

- `LockedEvaluationAdapter` reads saved predictions, not fresh image inference.
- `ResNet50AngleAdapter` accepts image bytes using compatible weights configured
  through `GEOMED_RESNET_CHECKPOINT`.
- The independent repair adapter requires `GEOMED_LANDMARK_CHECKPOINT` and
  `GEOMED_REPAIR_CHECKPOINT`.

The current registry unconditionally loads
`GEOMED_DATA_ROOT/processed/hvangleest/medimageinsight_locked_test_eval.json`
(`GEOMED_DATA_ROOT` defaults to `data`). This artifact is not distributed.
Configuring live weights alone is therefore insufficient to start this service.
Checkpoint architecture and preprocessing must match the adapter.

`compose.yaml` provides PostgreSQL, object storage, Orthanc, and OHIF integration
configuration. It requires external artifacts and setup, and is not a
one-command public demo. The offline checks do not validate these integrations.
Keep inference private: its artifact loader is not an untrusted public-URL gateway.

Uploaded predictions always require review. DICOM support and metadata removal
are limited safeguards, not PACS certification or complete anonymization.
Do not use patient data in this demo.

## Data and scripts

See [data policy](../data/README.md) for local HVAngleEst preparation and geometry
auditing. Check dataset access and redistribution permissions separately.

Retained scripts cover preparation, geometry auditing, prediction-artifact
evaluation, retrieval, planner evaluation, and synthetic workflow checks.
Artifact-based scripts need compatible local inputs; they do not recreate
missing historical artifacts. Generated results stay under ignored `outputs/`.
