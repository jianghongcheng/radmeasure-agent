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
- Reduced unsafe SQL actions from 16.7% to 0% while improving task success from
  52.8% to 83.3% through policy-gated execution on 36 frozen cases; attributed
  all remaining failures to repair proposal execution or contract rejection.

## One-line version

Built RadMeasure, a verifiable multimodal agent platform that constrains LLM
planning through registered protocols, deterministic geometry tools,
`KEEP / REPAIR / STOP` safety policies, and end-to-end auditable replay.

## Harbor evaluation version

Packaged the reliability suite as a **Harbor-compatible evaluation environment**
(the substrate behind Terminal-Bench) with separate agent/verifier containers,
hidden labels, and network-isolated execution; on the frozen **36-case**
SQL-repair suite, policy gating raised task success from **19/36 to 30/36** and
blocked **6/6 unsafe proposals**, with **92 tests** in public CI.

## Claim guardrails

Do not describe RadMeasure as an autonomous radiology agent, a clinically
validated system, or a fully automated measurement product. The current
evidence supports a production-style research prototype with explicit human
review and safety boundaries.
