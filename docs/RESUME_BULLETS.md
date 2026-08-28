# Resume-ready project entry

## RadMeasure — Verifiable Multimodal Agent System

**Python, PyTorch, FastAPI, MCP, PostgreSQL, MinIO, Docker, Ollama**

- Built a bounded tool-using agent runtime combining LLM planning,
  schema/registry validation, policy-gated execution, verification, repair,
  deterministic replay, and MCP tools across radiography and SQL environments.
- Designed deterministic geometry execution and a bounded `KEEP / REPAIR / STOP`
  controller with policy validation, human-review routing, artifact lineage,
  and reproducible trajectory replay.
- Productionized PyTorch inference behind authenticated FastAPI services with
  durable PostgreSQL workers, idempotent jobs, content-addressed MinIO storage,
  retries, circuit breaking, structured traces, Docker Compose, and CI.
- Without retuning, evaluated a separately frozen 108-case confirmatory suite
  across six schemas absent from development: raised task success from 73/108
  to 98/108, blocked 25/25 unsafe proposals, and rejected all six incorrect
  outputs; the paired cluster-bootstrap 95% interval was +7.4 to +40.7 points.

## One-line version

Built RadMeasure, a verifiable multimodal agent platform that constrains LLM
planning through registered protocols, deterministic geometry tools,
`KEEP / REPAIR / STOP` safety policies, and end-to-end auditable replay.

## Harbor evaluation version

Packaged the reliability suite as a **Harbor-compatible evaluation environment**
(the substrate behind Terminal-Bench) with separate agent/verifier containers,
hidden labels, network-isolated execution, and hidden database fixtures; the
frozen v3 replay reproduced **98/108** successful tasks and blocked **25/25**
unsafe proposals, with an oracle verifier check of **108/108** and **106 tests**
locally.

## Claim guardrails

Do not describe RadMeasure as an autonomous radiology agent, a clinically
validated system, or a fully automated measurement product. The current
evidence supports a production-style research prototype with explicit human
review and safety boundaries.
