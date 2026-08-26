# Portfolio result record

Generated from local reproducible evaluations on 2026-08-23.

## Product surface

- Browser dashboard and REST endpoints share one typed boundary with MCP tools.
- The dependency-free stdio MCP server supports initialization, discovery,
  calls, protocol errors, and structured results.
- Docker Compose provides a one-command persisted-evaluation replay demo.
- GitHub Actions runs core tests and compilation on Python 3.11.

## Cross-domain agent reliability

A frozen 36-case adversarial SQL-repair suite evaluates the same bounded action
contract outside radiographic measurement. Qwen3-8B generated each proposal
once; the evaluator replayed the identical proposal through every harness
configuration so generation randomness could not confound the component
ablation.

| Configuration | Task success | Unsafe action | STOP rate |
|---|---:|---:|---:|
| LLM only | 19/36 (52.8%) | 6/36 (16.7%) | 41.7% |
| Schema validation | 19/36 (52.8%) | 6/36 (16.7%) | 41.7% |
| Registry validation | 19/36 (52.8%) | 6/36 (16.7%) | 41.7% |
| Policy enforcement | 30/36 (83.3%) | 0/36 (0%) | 61.1% |
| Verifier only | 24/36 (66.7%) | 6/36 (16.7%) | 44.4% |
| Policy + verifier | **30/36 (83.3%)** | **0/36 (0%)** | 61.1% |

Policy enforcement blocked all six unsafe model proposals. The six residual
failures under the full harness were five repair-proposal execution failures and
one output-contract rejection. On the self-hosted Ollama/Qwen3-8B evaluation
using one RTX 3090, planner generation measured 458.1 ms p50 and 600.7 ms p95.
Each planner call averaged 116.2 prompt and 28.3 completion tokens; these token
counts do not include deterministic tool execution or represent a full
multi-turn agent trajectory.

Evidence:

- cases: `data/benchmarks/sql_repair_v1.json`;
- evaluator: `scripts/evaluate_sql_harness_ablation.py`;
- raw result: `outputs/portfolio/sql_harness_ablation_qwen3_8b.json`.

This is a small frozen engineering benchmark. It is not evidence of SQL SOTA,
production traffic, or statistical generalization beyond the suite.

## Measurement

The persisted benchmark artifact contains 176 unilateral HVAngleEst evaluations.
GeoMed Copilot averaged undirected anatomical axes from three persisted
MedImageInsight spatial readouts (seeds 17, 42 and 73), then reconstructed
angles analytically. The included artifact and current processed manifests come
from different split states; therefore this repository does not currently claim
patient-disjoint evaluation status.

| Metric | HVA | IMA |
|---|---:|---:|
| MAE | 3.563° | 2.130° |
| Median absolute error | 2.632° | 1.800° |
| P95 absolute error | 9.069° | 5.471° |
| Within tolerance | 75.57% at 5° | 75.57% at 3° |

Evaluation artifact:
`data/processed/hvangleest/medimageinsight_locked_test_eval.json` (local,
ignored because it contains per-image records).

## Similar-case retrieval

For each test image, relevance is defined as the oracle five nearest locked
training cases in normalized HVA/IMA target space. Retrieval uses predicted
angles, MedImageInsight pooled image embeddings, or an equal-weight hybrid.
Weights were fixed before test evaluation and were not tuned on the test set.

| Method | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|
| Predicted geometry | 0.0625 | 0.1243 | 0.0648 |
| Image embedding | 0.0080 | 0.0360 | 0.0081 |
| Hybrid | **0.0659** | **0.1339** | **0.0686** |

The low absolute retrieval scores are retained as an explicit failure analysis.
They show that a general medical image embedding does not reliably represent
measurement similarity.

## Reproducibility findings

- The 176-case prediction artifact currently maps to 122 train, 24 validation,
  and 30 test records in the checked-in processed manifests. This indicates a
  later split regeneration, so the API exposes `split_alignment_verified=false`
  and the project does not claim patient-disjoint status for this artifact.
- An integration test initially exposed angle reconstruction in normalized
  coordinates without restoring image aspect ratio. The replay service now
  converts endpoints to pixel space; across all 176 cases, the maximum
  analytical-to-target delta is below 0.0005 degrees.

## Evidence retrieval

A four-source curated catalog was evaluated on five manually labeled questions.
The lexical/metadata baseline achieved Hit@3 1.00 and MRR 1.00. This is a smoke-
scale transparent golden set, not a production benchmark.

## System verification

- FastAPI health: 200
- known locked analysis: 200 / complete
- repeated request ID: byte-equivalent idempotent response
- result lookup: 200
- unknown image: 422 with explicit live-inference limitation
- automated test suite: 89 passed, 1 skipped (local run, 2026-08-25)

## Claim boundary

Resume-safe: the frozen 36-case SQL result and counts reported above; 176
persisted medical evaluations; measurement metrics; tool workflow;
case/evidence retrieval; artifact hashes; API behavior; evaluation harness; and
the disclosed split-alignment limitation.

Not resume-safe: clinical deployment, diagnosis, production traffic, external
clinical validity, or retrieval quality beyond the reported small evaluations.
