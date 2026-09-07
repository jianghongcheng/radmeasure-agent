# RadMeasure — Medical Imaging Measurement Agent

Research prototype for **protocol-defined HVA and IMA radiographic measurement**:
constrained planning, deterministic geometry checks, bounded repair,
human review, and persisted execution records.

**Not a medical device. Not validated for diagnosis or patient care.**

Analytical SQL is maintained separately in
[ContractSQL](https://github.com/jianghongcheng/contractsql).

## Medical workflow

Registered request → authorized plan → measurement → contract and geometry
checks → complete or human review → trace and replay.

- Registered hallux valgus angle (HVA) and intermetatarsal angle (IMA) protocols.
- Typed contracts reject missing, duplicate, non-finite, and inconsistent outputs.
- FastAPI job endpoints, API-key roles, SQLite job storage, and a worker.
- Uploaded-image inference adapter; uploaded results require human review.
- MCP medical tools: capabilities, registered cases, and radiograph analysis.
  Direct MCP analysis is distinct from the durable API job workflow.
- PostgreSQL, object storage, Orthanc, and OHIF adapters remain available;
  fresh integration validation of those services is not claimed here.

## Offline quick start

```bash
git clone https://github.com/jianghongcheng/radmeasure-agent.git
cd radmeasure-agent
python -m venv .venv
source .venv/bin/activate
pip install -e .
radmeasure --question "Measure HVA and IMA"
```

Runs one **synthetic** case through the medical pipeline, without a model or
network call. A completed synthetic job is not clinical approval.

## Local API demo

Install `pip install -e '.[api]'`. Set the same environment in two terminals:

```bash
export GEOMED_DEMO_MODE=1
unset GEOMED_EVAL_REPLAY
export GEOMED_JOB_DB=/tmp/radmeasure-demo/jobs.sqlite
export GEOMED_ARTIFACT_ROOT=/tmp/radmeasure-demo/artifacts
export GEOMED_API_KEYS='{"operator-local":{"name":"local operator","role":"operator"},"reviewer-local":{"name":"local reviewer","role":"admin"},"viewer-local":{"name":"local viewer","role":"viewer"}}'
```

Run `uvicorn geomed_copilot.api:app --host 127.0.0.1 --port 8766` in one
terminal and `radmeasure-worker` in the other. Open http://127.0.0.1:8766.
Example keys are only for loopback demos; replace before deployment.

Live uploads need a separately configured model service and compatible weights.
See [model serving](docs/MODEL_SERVING.md) and
[review workflow](docs/DICOM_REVIEW_WORKFLOW.md).
The existing Compose deployment needs separately provisioned artifacts; it is
not the offline demo.

## Evidence modes and limitations

| Mode | Input | Evidence |
| --- | --- | --- |
| Synthetic demo | Bundled fabricated case | Software control flow and guards |
| Locked replay | Saved predictions | Reanalysis of historical artifacts |
| Live inference | Uploaded image + configured model | Actual inference, requiring review |

Do not describe replay as fresh inference or tests as clinical validation.
Historical medical experiments remain in docs and scripts. SPIDER in these
research scripts is the spine-imaging dataset, not a SQL benchmark.

```bash
pip install -e '.[dev]'
python -m pytest -q
python -m compileall -q src
```

See [release validation](docs/MEDICAL_RELEASE_VALIDATION.md) and
[evidence boundaries](docs/PORTFOLIO_RESULTS.md).
No production-user impact, prospective clinical validation, or continuous-load
SLO has been demonstrated. Geometric consistency is not anatomical correctness.
Metadata removal is not complete de-identification; do not upload patient data.
Models and clinical images are not distributed here.

Legacy geomed_copilot imports and command aliases remain for compatibility.
SQL files were removed from the medical working tree; Git history is preserved.
