# Resume-ready project entry

## RadMeasure — Verifiable Multimodal Agent System

**Python, PyTorch, FastAPI, MCP, PostgreSQL, MinIO, Docker, Ollama**

- Built a verifiable multimodal agent system that converts natural-language
  requests into registered radiographic measurement workflows while preventing
  LLM outputs from directly controlling executable tools.
- Designed deterministic geometry execution and a bounded `KEEP / REPAIR / STOP`
  controller with policy validation, human-review routing, artifact lineage,
  and reproducible trajectory replay.
- Productionized PyTorch inference behind authenticated FastAPI services with
  durable PostgreSQL workers, idempotent jobs, content-addressed MinIO storage,
  retries, circuit breaking, structured traces, Docker Compose, and CI.
- Evaluated planner reliability on a frozen 24-case safety suite: a local
  Qwen3-8B planner produced valid JSON but underperformed the rule planner,
  demonstrating why LLM proposals remain behind deterministic authorization;
  separately verified all 12 controller-policy cases with zero unsafe actions.

## One-line version

Built RadMeasure, a verifiable multimodal agent platform that constrains LLM
planning through registered protocols, deterministic geometry tools,
`KEEP / REPAIR / STOP` safety policies, and end-to-end auditable replay.

## Claim guardrails

Do not describe RadMeasure as an autonomous radiology agent, a clinically
validated system, or a fully automated measurement product. The current
evidence supports a production-style research prototype with explicit human
review and safety boundaries.
