# RadMeasure

**Radiographic measurement with geometric verification and human review.**

RadMeasure is a Python research application for measuring hallux valgus angle
(HVA) and intermetatarsal angle (IMA). It combines protocol-guided agent planning,
measurement tools, and a review workflow, with execution records that make each
decision inspectable.

[Quick start](#quick-start) · [Usage](docs/USAGE.md) ·
[Architecture](docs/ARCHITECTURE.md) · [Evaluation](docs/EVALUATION.md)

## Features

- **Protocol-guided planning** — select registered HVA/IMA measurements using a
  rule-based planner or an optional LLM planner.
- **Measurement verification** — check required outputs, numerical validity,
  and consistency with available geometry.
- **Bounded repair** — apply eligible corrections within explicit limits and
  record why a result was retained or stopped.
- **Human review** — review uploaded-image predictions, approve or reject
  results, and record corrected measurements.
- **Application interfaces** — run a CLI demo, use the FastAPI dashboard and
  job endpoints, or call measurement tools through MCP.
- **Execution history** — inspect task status, tool traces, review decisions,
  and replay lineage.

## Quick start

Requires Python 3.10 or newer.

```bash
git clone https://github.com/jianghongcheng/radmeasure-agent.git
cd radmeasure-agent
python -m venv .venv
source .venv/bin/activate
pip install -e .
radmeasure --question "Measure HVA and IMA"
```

The command returns a JSON task record containing measurements, verification
results, and an execution trace. In the bundled synthetic example, the reported
measurements are **HVA 15.0°** and **IMA 8.0°**, and the task completes.

This demo runs without model weights or network access. It uses fabricated
geometry and placeholder evidence to demonstrate the workflow.

For the local web interface and optional model setup, see [Usage](docs/USAGE.md).
Live image inference requires compatible weights and separately provisioned
service artifacts, which are not included in this repository.

## How it works

```text
Request → protocol selection → measurement → verification
                                              ├─ keep result
                                              ├─ eligible repair → verify again
                                              └─ stop for review
```

The planner selects the measurement procedure; measurement tools produce the
angles. Registered-case analysis checks the resulting geometry and output
contract. Uploaded images follow a model-inference path and always require
human review, including after a proposed correction.

See [Architecture](docs/ARCHITECTURE.md) for the source map and execution details.

## Development

```bash
pip install -e '.[dev]'
python -m pytest -q
python -m compileall -q src
```

See [Evaluation](docs/EVALUATION.md) for reproducible checks and artifact
requirements, and [Contributing](CONTRIBUTING.md) for development guidelines.

## Research use

RadMeasure is a research prototype, not a medical device, and is not validated
for diagnosis or patient care. Geometric consistency alone does not establish
anatomical correctness. Use synthetic or appropriately authorized research data;
do not upload patient information to the demo. See the [data policy](data/README.md).

## License

[MIT](LICENSE).
