# Local setup

Install dependencies:

```bash
pip install -e '.[api,prod,vision,inference]'
```

Set the same environment in API and worker terminals:

```bash
export RADMEASURE_DEMO_MODE=1
export RADMEASURE_EVAL_REPLAY=0
export RADMEASURE_JOB_DB="$PWD/runtime/jobs.db"
export RADMEASURE_ARTIFACT_ROOT="$PWD/runtime/artifacts"
export RADMEASURE_PROVIDER_CREDENTIAL_DIR="$PWD/runtime/provider-credentials"
export RADMEASURE_API_KEYS='{"operator-local":{"name":"local operator","role":"operator"},"reviewer-local":{"name":"local reviewer","role":"admin"}}'
export RADMEASURE_INFERENCE_URL=http://127.0.0.1:8100
export RADMEASURE_INFERENCE_TOKEN=replace-with-your-service-token
```

These example access keys are for localhost only. Demo mode enables synthetic registered-case tools; it never fabricates uploaded-image predictions.

Initialize SQLite once before launching concurrent processes:

```bash
python -c 'import os; from pathlib import Path; from radmeasure.jobs import SqliteJobRepository; SqliteJobRepository(Path(os.environ["RADMEASURE_JOB_DB"]))'
```

Start the API and worker in separate terminals:

```bash
uvicorn radmeasure.api:create_app --factory --host 127.0.0.1 --port 8000
radmeasure-worker
```

Open **http://127.0.0.1:8000**. Select GPT, Claude, or Gemini and enter an image-capable Model ID and API key. The app access key under Connection settings authenticates to RadMeasure, not to the model provider.

## Enable measurement

In another terminal, configure a compatible HRNet detector checkpoint and use the same inference token:

```bash
export RADMEASURE_LANDMARK_CHECKPOINT=/absolute/path/to/hrnet_lm.pt
export RADMEASURE_INFERENCE_TOKEN=replace-with-your-service-token
export RADMEASURE_MODEL_DEVICE=cpu
uvicorn radmeasure.inference_api:create_app --factory --host 127.0.0.1 --port 8100
```

Set `RADMEASURE_DISCREPANCY_POLICY` in the worker environment before starting it. Supply JSON with `keep_degrees` and `stop_degrees` maps for HVA and IMA, plus `max_repairs`. Thresholds must be chosen using development data; no clinical defaults are supplied.

```bash
export RADMEASURE_DISCREPANCY_POLICY="$(cat /absolute/path/to/your-policy.json)"
```

Without both models and a valid policy, the job goes to human review with a reason. See [workflow configuration](DUAL_PATH_WORKFLOW.md) for policy bounds, provider variables, tool permissions, and credential handling.

## Docker

Compose starts the API, workers, inference service, PostgreSQL and object storage, plus the existing local imaging services. Configure `.env` from `.env.example`, supply a compatible landmark checkpoint and comparison policy, then run:

```bash
docker compose up --build
```

API and workers share a credential volume. The optional custom VLM endpoint must be reachable from the worker container. Model weights and provider credentials are not bundled.

## Existing registered-case tools

The CLI and MCP registered-case interfaces remain separate from uploaded-image inference:

```bash
RADMEASURE_DEMO_MODE=1 RADMEASURE_EVAL_REPLAY=0 radmeasure --question "Measure HVA and IMA"
RADMEASURE_DEMO_MODE=1 RADMEASURE_EVAL_REPLAY=0 radmeasure-mcp
```

`RADMEASURE_MEASUREMENT_WORKFLOW=legacy` selects the previous upload pipeline when needed.
