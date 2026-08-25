# Five-minute interview demo

## Setup

```bash
docker compose up --build
```

Open `http://localhost:8000`.

## English talk track

**0:00–0:40 — Problem**

“GeoMed Agent Platform turns model measurements into an auditable workflow.
Instead of returning one opaque number, it verifies geometry, retrieves similar
research cases, grounds the response in cited evidence, and records every tool
call with latency and failure state.”

**0:40–1:30 — Architecture**

“REST and MCP clients use the same typed Python tool boundary. The MCP server
lets an external agent discover cases and call radiograph analysis, while the
FastAPI service creates durable idempotent jobs. A separate worker atomically
claims each job with a time-bounded lease, applies bounded retries, and records an
append-only state history. The orchestrator remains deterministic, so evaluation
does not depend on an LLM judge.”

**1:30–2:30 — Live workflow**

Select a case and click Analyze.

“This response contains the persisted model prediction, an independent
analytical reconstruction, the evaluation target and absolute error, retrieved
cases, evidence citations, provenance, and tool-level latency. A large
prediction-versus-geometry discrepancy produces `review_required` rather than
hiding uncertainty.”

**2:30–3:30 — Evaluation**

“The included artifact contains 176 evaluations with HVA and IMA MAE of 3.56
and 2.13 degrees. I also run engineering checks for tool success, citation
presence, latency, API behavior, and geometry reconstruction. The test suite has
30 tests, and CI runs it on Python 3.11.”

**3:30–4:30 — Failure analysis**

“During integration I found two important limitations. First, the persisted
artifact maps to a different version of the processed splits, so the service
sets `split_alignment_verified` to false instead of making a patient-disjoint
claim. Second, normalized landmark coordinates initially ignored image aspect
ratio. After restoring pixel scale, all 176 reconstructed targets agree within
0.0005 degrees. Both findings became regression checks and documented claim
boundaries.”

**4:30–5:00 — Production direction**

“The service currently replays persisted predictions and never claims live
clinical inference. The next adapter can load a versioned model artifact without
changing the REST, MCP, retrieval, evaluation, or observability layers. That
separation is the main production design decision.”

## Likely follow-up questions

- Why MCP? It provides tool discovery and a standard boundary for agent clients.
- Why deterministic orchestration? It makes regression testing and failure
  attribution straightforward.
- What would you ship next? A versioned live-inference adapter, external
  validation set, persistent traces, authentication, and human approval gates.
- What failed? Image embeddings were weak for angle-neighbor retrieval, and the
  checked-in split manifests do not align with the persisted evaluation artifact.
