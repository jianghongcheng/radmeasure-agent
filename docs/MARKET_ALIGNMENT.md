# Market alignment

Checked on 2026-08-23. Job pages can expire; links are retained as evidence of
the requirements used to scope this project.

| Hiring signal | Project evidence now | Next production milestone |
|---|---|---|
| Typed tool calling and multi-step orchestration | Deterministic tool boundary, typed state, execution traces | LangGraph backend with retry policy and human approval node |
| Hybrid retrieval and reranking | Lexical + metadata baseline; geometry-distance case retrieval | Dense/sparse Qdrant retrieval, cross-encoder reranking, MRR/nDCG |
| Multimodal retrieval | Case schema accepts measurement geometry | Image embeddings and matched-vs-shuffled retrieval evaluation |
| Evaluation and regression testing | Versioned smoke set; MAE, Recall@K, citation and tool metrics | Public-data golden set, error taxonomy, CI quality gates |
| Reliability and observability | Partial-result fallback, per-tool status/latency, idempotent API request IDs | OpenTelemetry traces, persistent jobs, retry/backoff and cost metrics |
| Research-to-production | Geometry verification and swappable model/retrieval boundaries | Stable GeoMed checkpoint adapter, container and deployed API |

Representative current role evidence:

- [TwelveLabs, ML Research Engineer — Video Cognition](https://jobs.ashbyhq.com/twelve-labs/7ccdb7ae-1d16-4f87-939b-a50a8873465d): multimodal retrieval, tool-based workflows, scientific rigor, and reliable user-facing systems.
- [TwelveLabs, Senior AI Engineer — Tools & Agents](https://jobs.ashbyhq.com/twelve-labs/8ad28030-3654-4793-a7e1-25611e29fbd0): knowledge stores, retrieval primitives, grounded citations, and trustworthy agent infrastructure.
- [Titan AI, AI LLM Retrieval Engineer](https://jobs.ashbyhq.com/titan-ai/2bd78c4f-6cea-44a2-977b-32b4147a2e8d): dense/sparse/hybrid retrieval and Recall@K, MRR, nDCG evaluation.
- [Drata, Applied AI Engineer](https://jobs.ashbyhq.com/drata/51a418d1-c371-4f9f-b248-2c3b542bec42): golden datasets, regression detection, reranking, statistical testing, and failure taxonomies.
- [Sarvam, Applied AI Engineer — Agents](https://jobs.ashbyhq.com/sarvam/30259734-50c3-4f1c-81cd-8bff07e585e7): tools, memory, deployment, retries, audit trails, latency/cost, observability, and integration tests.
- [Twin Health, Senior AI Engineer](https://boards.greenhouse.io/embed/job_app?token=5655780004): healthcare domain, RAG, iterative evals, deep learning, and multimodal models.

## Positioning conclusion

The project is optimized for **Applied AI Engineer — Multimodal RAG and Agent
Systems**, not for a pure prompt-engineering role. GeoMed supplies differentiated
vision, geometry, and experimental depth; the copilot layer demonstrates the
retrieval, evaluation, backend, and reliability work repeatedly requested by
current employers.

