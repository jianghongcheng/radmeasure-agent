# RadMeasure

A research agent for **protocol-defined radiographic measurement** of hallux
valgus angle (HVA) and intermetatarsal angle (IMA).

RadMeasure separates planning from measurement: a constrained planner selects
registered protocols, measurement tools produce results, and explicit checks
route them to completion, bounded repair, or human review. The LLM does not
directly invent the reported angles.

**Research only. Not a medical device; not validated for diagnosis or patient care.**

## Workflow

```text
Request → registered plan → measurement tools → contract / geometry checks
                                                   ├─ keep
                                                   ├─ bounded repair → recheck
                                                   └─ stop / human review
```

- HVA/IMA protocol registry and validated planner output; unsupported plans stop.
- Deterministic geometry checks and explicit repair limits.
- Uploaded-image inference adapter with mandatory human review.
- FastAPI, a local dashboard, durable jobs, API-key roles, and execution records.
- MCP tools for capabilities, registered cases, and radiograph analysis.

## Run the offline demo

Requires Python 3.10 or newer.

```bash
git clone https://github.com/jianghongcheng/radmeasure-agent.git
cd radmeasure-agent
python -m venv .venv
source .venv/bin/activate
pip install -e .
radmeasure --question "Measure HVA and IMA"
```

This runs a bundled **synthetic** case without a model or network call. It
demonstrates software behavior, not image-model accuracy. Live image inference
requires separately provisioned compatible weights and a model service.

## Documentation

- [Usage](docs/USAGE.md): local API demo, model-service prerequisites, and data.
- [Architecture](docs/ARCHITECTURE.md): source map, checks, and review boundaries.
- [Evaluation](docs/EVALUATION.md): reproducible checks and evidence limitations.

## Development

```bash
pip install -e '.[dev]'
python -m pytest -q
python -m compileall -q src
```

Models, clinical images, runtime state, and generated experiment outputs are not
distributed. Do not upload patient information. Metadata stripping is not
complete de-identification, and geometric consistency is not anatomical validity.

This repository is medical-only. Database analysis is a separate project,
[ContractSQL](https://github.com/jianghongcheng/contractsql), with separate
runtime behavior and evaluations. SQLite/PostgreSQL here store application jobs;
they are not analytical query tools. The `geomed_copilot` package name remains
for compatibility.
