# Architecture

![Image-first measurement workflow](assets/workflow.svg)

## Measurement graph

`measurement_graph.py` implements a LangGraph workflow with validation, parallel landmark and VLM branches, a joining controller, bounded VLM repair, and finalization or human-review routing.

The HRNet detector returns anatomical coordinates in original-image pixels. Geometry code computes HVA and IMA. The VLM sees the same image and protocol definitions independently. The controller checks finite coordinates, axis validity, image quality, required measurements, and disagreement thresholds. KEEP retains the geometry-derived measurement; REPAIR revises only the VLM estimate; STOP creates a reviewable job.

`vision_measurement.py` allows protocol lookup and image cropping. `provider_transport.py` implements native image requests for GPT, Claude and Gemini. The model cannot invoke arbitrary executable code. Agreement and geometry checks do not prove anatomical correctness.

## Service boundaries

- **Browser / FastAPI:** uploads, provider selection, status and review endpoints.
- **Worker / LangGraph:** model orchestration, policy checks and execution records.
- **Inference service:** separately authenticated landmark prediction.
- **Storage:** SQLite or PostgreSQL jobs; local or S3-compatible image artifacts.
- **Credentials:** separate private files, shared by API and worker; no raw provider keys in job payloads or traces.

Intermediate graph checkpoints are not persisted by the default worker integration. Job recovery may rerun inference. Completed jobs retain the initial estimate, repair attempts, policy, routing reason and execution trace.

Registered-case evaluation and MCP tools keep their separate execution path. The legacy upload path is available through an explicit environment setting. This repository remains focused on radiographic measurement.

See [Usage](USAGE.md) and [workflow details](DUAL_PATH_WORKFLOW.md) for configuration and limits.
