<div align="center">

# RadMeasure

**A medical measurement agent with controlled verification and repair.**

HRNet landmarks · Vision-language models · KEEP / REPAIR / STOP

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-167D7F)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![Status](https://img.shields.io/badge/Status-Research_prototype-64748B)

[Overview](#overview) · [Architecture](#architecture) · [Getting Started](#getting-started) · [Documentation](docs/DUAL_PATH_WORKFLOW.md)

</div>

## Overview

RadMeasure measures hallux valgus angle (HVA) and intermetatarsal angle (IMA) from foot radiographs. An HRNet landmark model and a vision-language model inspect the image independently; a LangGraph controller checks their agreement and decides whether to accept, recheck, or request human review.

- **Simple web interface** — upload an image and choose GPT, Claude, or Gemini.
- **Protocol-constrained actions** — model requests are limited to registered definitions and tools.
- **Bounded repair** — the VLM rechecks the image within an explicit retry budget.
- **Traceable decisions** — retain estimates, verification reasons, and review history.

## Architecture

![RadMeasure architecture: parallel landmark and VLM measurements with a KEEP, REPAIR, STOP controller](docs/assets/workflow.svg)

Accepted measurements come from landmark geometry. Repair revises the VLM estimate, not the landmark coordinates. Agreement alone does not establish accuracy.

## Getting Started

```bash
pip install -e '.[api,prod,vision,inference]'
```

Follow the [setup guide](docs/DUAL_PATH_WORKFLOW.md#configuration-and-local-execution) to configure the landmark checkpoint, comparison policy, and app access key. Start the API and worker in separate terminals:

```bash
uvicorn radmeasure.api:create_app --factory --host 127.0.0.1 --port 8000
radmeasure-worker
```

Open **http://127.0.0.1:8000**, upload an image, and enter your selected provider's Model ID and API key. Checkpoints and credentials are not bundled.

---

[Workflow & configuration](docs/DUAL_PATH_WORKFLOW.md) · [Controller source](src/radmeasure/measurement_graph.py) · [Provider adapters](src/radmeasure/provider_transport.py) · [Contributing](CONTRIBUTING.md)

*Research prototype; not validated for clinical use.*
