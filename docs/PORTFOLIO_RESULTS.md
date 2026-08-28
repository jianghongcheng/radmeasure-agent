# Portfolio result record

Generated from local reproducible evaluations on 2026-08-23.

## Product surface

- Browser dashboard and REST endpoints share one typed boundary with MCP tools.
- The dependency-free stdio MCP server supports initialization, discovery,
  calls, protocol errors, and structured results.
- Docker Compose provides a one-command persisted-evaluation replay demo.
- GitHub Actions runs core tests and compilation on Python 3.11.

## Cross-domain agent reliability

A frozen 120-case adversarial SQL-repair suite evaluates the same bounded action
contract across five SQLite schemas (workforce, commerce, support, research,
and logistics). It contains 40 `KEEP`, 40 `REPAIR`, and 40 `STOP` cases grouped
into 24 failure-family clusters. Qwen3-8B generated each proposal once; the
evaluator replayed the identical proposal through every harness configuration
so generation randomness could not confound the component ablation. The suite
SHA-256 is `ab4275f4b4348945e60a8e63b0bd8768933c09e83f54392e9958e94f2dc9a52b`.

| Configuration | Task success | Unsafe accepted | Incorrect output accepted |
|---|---:|---:|---:|
| LLM only | 94/120 (78.3%) | 19/120 | 5/120 |
| Schema validation | 94/120 (78.3%) | 19/120 | 5/120 |
| Registry validation | 94/120 (78.3%) | 19/120 | 5/120 |
| Policy enforcement | 113/120 (94.2%) | 0/120 | 5/120 |
| Verifier only | 94/120 (78.3%) | 19/120 | 0/120 |
| Policy + verifier | **113/120 (94.2%)** | **0/120** | **0/120** |

Policy enforcement blocked all 19 unsafe model proposals. The verifier rejected
all five semantically incorrect outputs that policy alone would accept. Under
the full harness, the remaining seven unsuccessful tasks were two proposal
execution failures and five contract rejections; neither unsafe actions nor
incorrect outputs were accepted. Task success improved by 15.85 points under a
paired 2,000-sample bootstrap over 24 failure-family clusters (95% CI 4.17 to
31.67 points). The full-system Wilson intervals are 88.45%--97.15% for task
success and 0%--3.10% for unsafe-action rate.

Evidence:

- cases: `data/benchmarks/sql_repair_v2.json`;
- frozen generations: `data/benchmarks/sql_repair_v2_qwen3_8b_generations.json`;
- evaluator: `scripts/evaluate_sql_harness_ablation.py`;
- raw result: `outputs/portfolio/sql_harness_v2_qwen3_8b_audited.json`.

This is a small frozen engineering benchmark. It is not evidence of SQL SOTA,
production traffic, or statistical generalization beyond the suite.

### Held-out confirmatory suite

Before generation, v3 was frozen with SHA-256
`ed0c1416a92e27ec93363359af280b47af6b9b0aecc580aed5bb36e4fd1a7f00`.
It contains 108 balanced cases across six schemas and 18 failure families that
do not occur in v2. The planner prompt, policy, verifier, and evaluation
semantics were held fixed, and the result is reported without outcome-driven
retuning.

| Configuration | Task success | Unsafe accepted | Incorrect output accepted |
|---|---:|---:|---:|
| LLM only | 73/108 (67.6%) | 25/108 | 6/108 |
| Policy enforcement | 98/108 (90.7%) | 0/108 | 6/108 |
| Verifier only | 73/108 (67.6%) | 25/108 | 0/108 |
| Policy + verifier | **98/108 (90.7%)** | **0/108** | **0/108** |

The paired 2,000-sample bootstrap over 18 failure-family clusters estimates a
**+23.22-point** improvement over LLM-only (95% CI **+7.41 to +40.74**).
Policy blocked all 25 unsafe proposals, while verification rejected all six
incorrect outputs. Ten cases remain unsuccessful because the frozen planner
does not supply an acceptable fallback—not because policy is bypassed.

Evidence:

- preregistration: `data/benchmarks/sql_repair_v3_preregistration.json`;
- cases: `data/benchmarks/sql_repair_v3_confirmatory.json`;
- frozen generations: `data/benchmarks/sql_repair_v3_qwen3_8b_generations.json`;
- result: `outputs/portfolio/sql_harness_v3_qwen3_8b_confirmatory.json`.

### Tool execution by task class

| Expected action | Tasks | Registered tool calls | Successful tasks |
|---|---:|---:|---:|
| KEEP | 40 | 40 | 40 |
| REPAIR | 40 | 40 | 33 |
| STOP | 40 | 0 | 40 |

The aggregate 80/120 call rate is not used as an agent-complexity claim. It
reflects the benchmark design: every execution-eligible KEEP/REPAIR task invokes
the SQL tool exactly once, while STOP-policy tasks terminate before execution.

### Measurement protocol

- frozen suite: 120 cases from `data/benchmarks/sql_repair_v2.json`;
- execution date: 2026-08-27 (America/Chicago);
- model endpoint: self-hosted Ollama `qwen3:8b`;
- scheduling: sequential requests, batch size 1, one planner generation per case;
- warm-up: none; the reported distribution includes the first cold request;
- prompt length: 133.2 tokens on average;
- completion length: 30.7 tokens on average;
- timing source: Ollama `total_duration_ns`, measuring planner generation rather
  than multi-turn workflow latency;
- aggregation: median for p50 and nearest-rank order statistic for p95 over all
  120 requests;
- result: 451.0 ms p50, 646.2 ms p95, and a 16.8 s cold-start maximum;
- raw artifact: `outputs/portfolio/sql_harness_v2_qwen3_8b_audited.json`.

The six harness layers reuse the same 120 generated responses, so their
reliability comparison is not confounded by repeated stochastic generation.

### Component interpretation

- **Tool-call rate:** all 40 expected-`KEEP` and 40 expected-`REPAIR` cases call
  the registered SQL tool exactly once; all 40 expected-`STOP` cases call no
  tool. The aggregate rate below 1.0 therefore reflects intended refusal, not a
  missing execution loop.
- **Policy versus verifier:** policy blocks all 19 unsafe proposals but would
  accept five incorrect outputs. Verifier-only rejects those five outputs but
  does not block the 19 unsafe proposals. The full runtime eliminates both
  failure classes; task success remains 113/120 because rejected wrong outputs
  are still unsuccessful tasks.
- **Schema and registry:** these remain required execution boundaries, but the
  frozen model outputs happened to satisfy both in every case, so neither layer
  changes the aggregate result.
- **Residual failures:** two are proposal execution failures and five are
  output-contract rejections; no unsafe proposal bypasses policy and no
  incorrect output is returned.

## Planner authorization pilot

A separate frozen 24-case suite covers supported, unsupported, missing-input,
and prompt-injection requests.

| Planner | Correct actions | Unsafe actions | Valid JSON |
|---|---:|---:|---:|
| Fixed HVA+IMA workflow | 12/24 | 12/24 | 24/24 |
| Registry/rule planner | **18/24** | **4/24** | 24/24 |
| Qwen3-8B + schema only | 15/24 | 9/24 | 24/24 |

The small sample is directional pilot evidence, not a statistical-significance
claim. It motivated keeping the LLM behind deterministic authorization.

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

## Medical repair safety gate

Live uploads keep the supervised ResNet angle model as the primary predictor.
An independently trained HRNet landmark detector, residual RepairMLP, and
learned verifier may propose a one-step geometric edit. Policy accepts the edit
only when its verifier passes and its HVA/IMA outputs agree with the primary
model within registered bounds; otherwise it records the attempt and returns
`STOP` for review.

On 243 patient-disjoint cases, raw HRNet error was 52.61° HVA and 71.96° IMA.
Gated one-step repair reduced error to 28.92° and 24.67° with 83.1% coverage and
1.65% any-measurement harm. The proposal stack remains unsuitable as the final
predictor; this result motivates the cross-model gate and mandatory review path.
Raw evidence: `outputs/research/hrnet_geometry_repair.json`.

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
- automated test suite: 106 passed (local run, 2026-08-27)

## Claim boundary

Resume-safe: the frozen 120-case SQL result and uncertainty reported above; 176
persisted medical evaluations; measurement metrics; tool workflow;
case/evidence retrieval; artifact hashes; API behavior; evaluation harness; and
the disclosed split-alignment limitation.

Not resume-safe: clinical deployment, diagnosis, production traffic, external
clinical validity, or retrieval quality beyond the reported small evaluations.
