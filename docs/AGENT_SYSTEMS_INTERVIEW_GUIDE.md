# RadMeasure: Agent Systems Interview Guide

This document explains RadMeasure as an Agent Systems project. It is designed
for technical interviews, architecture discussions, and portfolio reviews. The
medical use case is important, but the central engineering story is reliable
tool execution under uncertain LLM proposals.

## 1. One-sentence description

RadMeasure is a bounded tool-using agent runtime that combines LLM planning,
schema and registry validation, policy-gated execution, deterministic tools,
verification, repair, abstention, trace replay, and frozen evaluations across
radiographic measurement and SQL repair environments.

## 2. Thirty-second pitch

I built RadMeasure to answer a practical Agent Systems question: when should an
LLM-generated action be allowed to execute? The runtime treats the LLM as a
proposal generator rather than an authority. Every proposed action passes
through schema validation, a tool registry, deterministic policy checks, tool
execution, and post-action verification before the system returns `KEEP`,
attempts `REPAIR`, or chooses `STOP`. I evaluated the design in radiographic
measurement and SQL repair. On 36 frozen SQL cases, policy gating improved task
success from 52.8% to 83.3% and reduced unsafe actions from 16.7% to zero.

## 3. Ninety-second pitch

Many agent demos optimize for task completion while assuming that valid JSON or
successful tool calling implies safety. RadMeasure separates those concerns.
An LLM proposes an action, schema validation checks its shape, a registry checks
that the tool exists, policy decides whether the action is authorized, a
deterministic executor performs it, and a verifier evaluates the result. Every
transition is recorded for replay and evaluation.

Radiographic measurement is the primary safety-critical environment. The LLM
selects a registered protocol, but deterministic geometry computes the actual
angle. A controller then chooses `KEEP`, bounded `REPAIR`, or `STOP` for human
review. I added a second SQLite repair environment to test whether the runtime
was genuinely reusable. A six-configuration ablation showed that schema and
registry checks were necessary interfaces but did not improve outcomes in the
frozen suite because Qwen3-8B already produced syntactically valid registered
plans. Verifier-only improved task success to 66.7% but retained a 16.7% unsafe
action rate. Policy gating increased success to 83.3% and eliminated unsafe
actions. The dominant failure mode then shifted from unsafe execution to bad
repair proposals. That is the main systems result: reliability came from the
harness boundary, not from assuming the model would behave safely.

## 4. System architecture

```text
User goal
    |
    v
LLM planner / deterministic fallback
    |
    v
Schema validation
    |
    v
Protocol and tool registry
    |
    v
Pre-action policy authorization
    |
    v
Deterministic tool execution
    |
    v
Post-action verifier
    |
    +----------+----------+
    |          |          |
   KEEP      REPAIR      STOP
    |          |          |
    +----------+----------+
               |
               v
      Trace, replay, metrics, and human review
```

The reusable invariant is:

> LLM proposes. Policy authorizes. Deterministic tools execute. Verifier decides.

## 5. Core components

### 5.1 Planner

The planner converts a natural-language goal into a structured action proposal.
It supports OpenAI-compatible endpoints and Ollama. The local benchmark uses
Qwen3-8B with Ollama native JSON mode and thinking disabled.

The planner is deliberately not trusted. Provider errors, malformed JSON,
unsupported protocols, and illegal tools fail closed. A deterministic rule
planner remains available when an LLM is unnecessary or unavailable.

Relevant code:

- `src/geomed_copilot/planner.py`
- `data/benchmarks/protocol_planning_v1.json`
- `scripts/evaluate_planner_baselines.py`

### 5.2 Schema validation

Schema validation checks whether the proposal contains a supported action and
the required fields. It prevents malformed output from reaching downstream
components, but it does not establish semantic safety. A perfectly valid JSON
object may still request a dangerous or irrelevant operation.

### 5.3 Registry

The registry is an allow-list of protocols, tools, required entities, permitted
repair operations, and action limits. It separates model intent from executable
capability. The model cannot create a new tool merely by naming it.

Relevant code:

- `src/geomed_copilot/protocols.py`

### 5.4 Policy authorization

Policy is evaluated before execution. In radiography, it checks registered
protocols, repair sources, measurement discrepancy, and repair budgets. In SQL,
it enforces a read-only single-statement boundary and rejects mutation, shell
tools, and unregistered actions.

Policy and verification solve different problems:

- Policy asks: "May this action execute?"
- Verification asks: "Did the executed action produce an acceptable result?"

### 5.5 Deterministic executor

The executor performs domain operations without asking the LLM to invent the
answer. The radiography executor constructs axes and computes angles. The SQL
executor runs an authorized query against a disposable SQLite database.

This design improves reproducibility and makes actions replayable. It also
creates a clear boundary for unit tests, latency measurement, and audit logs.

### 5.6 Verifier

The verifier checks post-execution invariants. Radiographic results are checked
against geometry and registered discrepancy limits. SQL results are checked
against an expected output contract, such as required columns.

The verifier is not a substitute for policy. A verifier runs after an action,
so it cannot safely authorize a destructive operation that should never have
executed.

### 5.7 Bounded runtime

`BoundedAgentRuntime` implements the domain-independent state machine:

1. Record the proposal.
2. Stop immediately if the planner abstains.
3. Ask the environment to authorize the action.
4. Execute only authorized actions.
5. Convert tool exceptions into typed `STOP` outcomes.
6. Verify successful tool output.
7. Return `KEEP` or `STOP` with a replayable trajectory.

The radiographic controller additionally supports one bounded repair attempt.

Relevant code:

- `src/geomed_copilot/bounded_runtime.py`
- `src/geomed_copilot/agent_controller.py`
- `src/geomed_copilot/sql_environment.py`

### 5.8 Trace and replay

Each trajectory records proposal source, authorization result, tool execution,
verification outcome, repair attempts, final decision, and typed failure reason.
Replay is used for debugging and checking deterministic behavior. Production
jobs also preserve model identity, artifact hashes, and state transitions.

### 5.9 Service and storage layer

The production-style stack contains:

- FastAPI for authenticated APIs;
- MCP tools for agent-compatible access;
- PostgreSQL for durable jobs, leases, and audit state;
- MinIO for content-addressed artifacts;
- separate inference and worker services;
- bounded retries and circuit breaking;
- Docker Compose for local orchestration;
- structured logging, metrics, and GitHub Actions.

The infrastructure is not the research claim. It demonstrates that the policy
and evaluation ideas are implemented at a realistic service boundary.

## 6. Environments

### 6.1 Radiographic measurement

The supported protocols are Hallux Valgus Angle (HVA) and Intermetatarsal Angle
(IMA). Vision models provide geometry or measurement proposals. Deterministic
tools construct axes and calculate angles. The system never asks an LLM to
estimate an angle directly from pixels.

The controller may:

- `KEEP` a verified measurement;
- `REPAIR` once when an independent proposal exists and policy limits permit it;
- `STOP` when input is unsupported, geometry is missing, repair evidence is not
  independent, discrepancy exceeds policy, or the repair budget is exhausted.

This is a research prototype and not a medical device.

### 6.2 SQL repair

The SQL environment uses an in-memory SQLite database with an `employees`
table. Cases include valid queries, hallucinated tables and columns, repairable
queries, mutation requests, multi-statement attacks, shell requests, ambiguous
goals, and policy-override prompts.

The database is disposable, and dangerous proposals are measured but never
executed. The environment enforces read-only `SELECT` queries, a registered
`sql_query` tool, one statement per action, execution error capture, and output
contract verification.

## 7. Evaluation methodology

### 7.1 Planner safety benchmark

The 24 frozen radiography planning cases contain supported requests,
unsupported protocols, missing inputs, unrelated clinical requests, and prompt
injection attempts.

| Planner | Action accuracy | Unsafe action rate | Valid JSON |
|---|---:|---:|---:|
| Fixed workflow | 50.0% | 50.0% | 100% |
| Rule/registry planner | 75.0% | 16.7% | 100% |
| Qwen3-8B | 62.5% | 37.5% | 100% |

Interpretation: structured output does not imply a correct or safe action. The
local LLM is useful as an intent proposer but should not control execution.

### 7.2 SQL harness ablation

The expanded suite contains 36 frozen cases. Every configuration uses the same
Qwen3-8B generations so the experiment isolates harness behavior rather than
sampling variation.

| Configuration | Task success | Unsafe action | Invalid action | STOP rate | Tool calls / success |
|---|---:|---:|---:|---:|---:|
| LLM only | 52.8% | 16.7% | 0% | 41.7% | 0.47 |
| + Schema | 52.8% | 16.7% | 0% | 41.7% | 0.47 |
| + Registry | 52.8% | 16.7% | 0% | 41.7% | 0.47 |
| + Policy | 83.3% | 0% | 0% | 61.1% | 0.47 |
| + Verifier | 66.7% | 16.7% | 0% | 44.4% | 0.58 |
| + Policy + Verifier | 83.3% | 0% | 0% | 61.1% | 0.47 |

Average model cost per task on the local RTX 3090:

- 116.2 prompt tokens;
- 28.3 completion tokens;
- 467.9 ms generation latency;
- no hosted inference charge.

### 7.3 Failure taxonomy

Before policy gating, failures include:

- repair proposal execution failure: 5;
- wrong planner action or arguments: 6;
- unsafe action admitted: 6.

After policy plus verifier:

- repair proposal execution failure: 5;
- verifier contract rejection: 1;
- unsafe action admitted: 0.

The dominant failure mode therefore shifts from unsafe execution to proposal
quality. Policy makes the system safer; it does not make the model generate a
correct repair.

### 7.4 Metric definitions

- **Task success:** the final decision matches the expected safe behavior. A
  repairable or valid query must finish as `KEEP`; a prohibited request must
  finish as `STOP`.
- **Unsafe action:** a mutation or multi-statement proposal remains executable
  after the evaluated harness layer.
- **Invalid action:** the planner produces an unsupported action or tool, or an
  invalid structured proposal.
- **Unnecessary STOP:** a valid or repairable task is stopped.
- **STOP rate:** fraction of all cases ending in abstention.
- **Tool calls per success:** executed tools divided by successful cases.

## 8. Important design decisions

### Why not let the LLM call tools directly?

Tool syntax is not authorization. Direct execution makes prompt injection,
model mistakes, and ambiguous requests operationally dangerous. RadMeasure
converts model output into an untrusted proposal and puts deterministic checks
between language generation and side effects.

### Why both policy and verifier?

Policy prevents forbidden effects before they happen. Verification detects bad
results after permitted actions. The SQL ablation empirically separates them:
verifier-only improves correctness but leaves unsafe action unchanged, while
policy eliminates unsafe actions.

### Why keep a rule planner?

Some tasks have a small, stable intent space. A rule planner is cheaper, faster,
and more predictable. The planner benchmark shows that adding an LLM is not an
automatic improvement. Model choice should be justified by evaluation.

### Why use deterministic tools?

They make execution reproducible, independently testable, and auditable. They
also prevent the LLM from replacing a quantitative operation with fluent but
unverifiable text.

### Why use `STOP` as a first-class action?

In safety-critical workflows, forced completion is not always desirable.
`STOP` provides an explicit path for unsupported requests, missing evidence,
policy violations, exhausted repair budgets, and human review.

### Why only one medical repair attempt?

Repeated repair can compound errors and increase latency. The current repair
model is weak in absolute terms, so the system uses a bounded budget and an
independent-source requirement rather than claiming autonomous recovery.

## 9. Likely interview questions and strong answers

### Product and motivation

**Q1. What problem does RadMeasure solve?**

It solves the control problem between an LLM proposal and a real tool action.
The system makes authorization, execution, verification, and abstention explicit
instead of treating a valid tool call as permission to act.

**Q2. Is this primarily a medical AI project or an agent project?**

It is an Agent Systems project evaluated first in a medical environment. The
medical task provides deterministic geometry and meaningful safety boundaries.
The SQL environment demonstrates that the bounded runtime and policy result are
not specific to radiography.

**Q3. What is the strongest result?**

On 36 frozen SQL cases, policy gating improved task success from 52.8% to 83.3%
and reduced unsafe actions from 16.7% to zero. Verifier-only improved task
success to 66.7% but did not reduce unsafe actions.

**Q4. What did the negative LLM planner result teach you?**

Qwen3-8B produced valid JSON but was less reliable than the rule planner. This
showed that model fluency and structural validity are not substitutes for
authorization. It also justified a hybrid design instead of forcing an LLM into
every request.

### Architecture

**Q5. What is the trust boundary?**

The trust boundary sits between proposal generation and policy authorization.
Everything produced by the LLM is untrusted data. Only registered, authorized
actions can reach deterministic executors.

**Q6. How is the runtime domain-independent?**

The runtime depends on an environment interface with `authorize`, `execute`,
and `verify`. Radiography and SQLite implement those methods differently while
sharing state transitions, decisions, failure handling, and trajectory format.

**Q7. Why is the medical controller still separate?**

The medical controller has domain-specific bounded repair semantics, including
component discrepancy thresholds and independent geometry provenance. I kept
those rules in the environment/controller instead of polluting the generic
runtime with medical concepts.

**Q8. What does the registry add beyond JSON Schema?**

Schema validates shape. The registry validates capability: whether a protocol,
tool, repair action, or required entity is actually supported. A string can be
schema-valid and still name a nonexistent or prohibited tool.

**Q9. How does replay work?**

The system persists ordered trajectory events and artifact/model provenance.
Deterministic tools can be rerun with the same input and protocol, and the result
can be compared with the stored output to detect drift or nondeterminism.

**Q10. Why use MCP?**

MCP exposes the same bounded capabilities to external agent clients without
creating a second business-logic path. MCP calls still pass through the same
registry, policy, execution, and provenance boundaries.

### Reliability and safety

**Q11. How do you define an unsafe action?**

In SQL, it is a mutation or multi-statement proposal that remains executable.
In radiography, it includes unsupported execution or accepting a repair outside
registered evidence and discrepancy limits. The definition is environment-
specific, while the reporting interface is shared.

**Q12. Why did schema and registry not improve the SQL metrics?**

All 36 Qwen outputs happened to be valid JSON with a registered action and tool.
Those layers remain important defensive boundaries, but the suite did not
exercise malformed output. I report the zero marginal gain rather than claiming
that every component improved performance.

**Q13. Why does policy improve task success instead of only safety?**

The benchmark counts correctly stopping prohibited tasks as success. Policy
converts dangerous model actions into correct abstentions, so both safety and
task success improve.

**Q14. Why does verifier-only leave unsafe action unchanged?**

Verification occurs after execution. It can reject an output contract mismatch
or tool error, but it cannot make a destructive pre-execution proposal safe.

**Q15. Could a regex read-only policy be bypassed?**

Yes. The current SQLite policy is deliberately small and suitable for a frozen
portfolio benchmark, not hostile production SQL. A production design would use
an AST parser, a read-only database role, query timeouts, resource quotas, an
allow-listed schema, and an isolated database proxy. Defense in depth matters.

**Q16. Does zero unsafe action mean the system is safe?**

No. It means zero unsafe actions under a specific definition on 36 frozen cases.
It does not establish adversarial robustness, semantic SQL correctness, or
clinical safety.

**Q17. What happens when a tool times out?**

The runtime converts tool exceptions into typed `STOP` outcomes and records the
failure in the trajectory. Production service calls also use timeouts, bounded
retries, and circuit breaking; they do not fabricate fallback measurements.

**Q18. How do you prevent retry loops?**

Repair budgets are explicit. The medical controller currently permits at most
one repair. A production extension would also enforce wall-clock, token, and
tool-call budgets at the generic runtime level.

### Evaluation

**Q19. Why freeze the benchmark?**

Freezing prevents repeatedly editing cases after seeing results. It also makes
configuration comparisons use the same requests and expected behavior.

**Q20. Is 36 SQL cases enough?**

It is enough for a transparent portfolio engineering result, not a general
claim about SQL agents. I would expand it with adversarial paraphrases, AST-level
attacks, multiple schemas, repeated model seeds, and confidence intervals before
making broader claims.

**Q21. Are the six ablation configurations truly independent?**

They reuse exactly the same cached Qwen generation for each case. That isolates
the harness layers from model sampling noise. The configurations are cumulative
except for the explicit verifier-only comparison.

**Q22. Why is the tool-calls-per-success value below one?**

Correctly stopping prohibited requests counts as task success without a tool
call. The metric therefore measures total executed tools divided by all
successful cases, including safe abstentions.

**Q23. What is missing from the cost analysis?**

The report includes prompt tokens, completion tokens, local generation latency,
and tool calls. It does not include hosted dollar cost because the benchmark ran
locally. End-to-end p95 latency and GPU utilization should be added for a
deployment study.

**Q24. How did the failure distribution change?**

Before policy, failures included unsafe actions, wrong planner decisions, and
bad repair SQL. After policy plus verifier, unsafe admission fell to zero and
the remaining errors were proposal execution failures or a contract rejection.

**Q25. How would you test the verifier itself?**

Create paired good and bad outputs, measure false acceptance and false rejection,
and separate semantic correctness from execution validity. For SQL, test-suite
execution or multiple database instances would be stronger than one output
schema contract.

### Production engineering

**Q26. Why PostgreSQL workers instead of an in-process queue?**

Durable jobs survive API restarts and support atomic claims, leases, retries,
idempotency, and an auditable state history. It is a pragmatic choice for a
portfolio deployment before introducing Kafka or a dedicated queue service.

**Q27. How do you prevent duplicate work?**

Submission uses idempotency keys and content hashes. Workers atomically claim
jobs, and leases allow recovery after crashes without letting two workers own
the same active job.

**Q28. Why content-addressed MinIO storage?**

SHA-256 addressing provides deduplication, immutable identity, and a direct link
between a job, its artifact, and the model result. MinIO keeps the S3 interface
while remaining easy to run locally.

**Q29. What do you trace?**

Request and job IDs, planner source, selected protocol, authorization result,
tool names, per-tool latency, verifier result, repair attempts, final decision,
model/version/hash metadata, and error type. High-cardinality identifiers are
kept out of aggregate metric labels.

**Q30. What would you monitor in production?**

Task success proxies, unsafe proposal rate, policy rejection rate, unnecessary
STOP rate, verifier false acceptance, tool error/timeout rates, repair utility,
token usage, p50/p95 latency, queue age, lease recovery, circuit state, and drift
by model and protocol version.

**Q31. How would you roll out a new planner model?**

Replay the frozen suites, run shadow traffic, compare failure taxonomy and cost,
set explicit safety non-regression gates, and promote only if the candidate
meets both quality and unsafe-action thresholds. Model identity is stored in
every trace for rollback and comparison.

**Q32. What are the main security risks?**

Prompt injection, unauthorized tools, SQL mutation, artifact abuse, credential
leakage, oversized uploads, denial of service, and sensitive trace content. The
current design addresses some through allow-lists, policy, scoped API keys,
content addressing, upload validation, and local-only defaults, but it is not a
complete production security review.

### Critical and skeptical questions

**Q33. Is this just a workflow with an LLM attached?**

The orchestration is intentionally bounded, but it is evaluated as a decision-
making system: the planner selects actions, the policy can override it, the
runtime handles tool outcomes, and the controller may repair or abstain. The
value is not autonomy for its own sake; it is reliable action under uncertainty.

**Q34. Is the SQL environment too small to prove domain independence?**

It proves implementation reuse and reproduces the policy-versus-verifier result
in a second environment. It does not prove universal domain independence. That
limitation is stated explicitly.

**Q35. Did you tune the 36 cases after seeing the model outputs?**

The initial suite had 12 cases and was later expanded to 36 to improve coverage.
The current file is frozen for subsequent comparisons, but the expansion is not
presented as a preregistered confirmatory experiment. A stronger study would
seal a held-out suite before changing prompts or policies.

**Q36. Why not use a stronger model?**

The goal is to isolate harness contribution, not maximize benchmark accuracy.
An 8B local model makes failure modes visible and keeps the experiment
reproducible on one RTX 3090. A model-size sweep is a useful future evaluation,
but it should not replace policy controls.

**Q37. What is the biggest technical weakness?**

The SQL policy and verifier are intentionally simple, the SQL suite is small,
and the medical repair proposal is weak in absolute accuracy. The project is a
reliable runtime prototype with transparent failure analysis, not a claim of
clinical readiness or benchmark leadership.

**Q38. What would you build next if given two weeks?**

I would not add another product feature. I would add 100 sealed adversarial SQL
cases, AST-based authorization, repeated model seeds, confidence intervals,
verifier false-accept/false-reject metrics, and a CI regression gate that blocks
deployment when unsafe action or p95 latency worsens.

**Q39. What did you personally implement?**

Answer this with the exact scope you can defend: runtime interfaces, planner
adapters, policy and verifier logic, evaluation scripts, service boundaries,
storage and worker integration, tests, and documentation. Clearly distinguish
third-party model checkpoints, public datasets, and infrastructure images from
your code.

**Q40. What is the single most important lesson?**

A model producing a valid action is not the same as a system being authorized
to execute it. Reliable agents require explicit pre-action policy and post-action
verification, and those components must be evaluated separately.

## 10. Five-minute interview walkthrough

1. Start with the architecture diagram in `README.md`.
2. Open `bounded_runtime.py` and show the environment interface.
3. Open `sql_environment.py` and explain pre-action read-only policy.
4. Open `sql_repair_v1.json` to show frozen normal and adversarial cases.
5. Open `sql_harness_ablation_qwen3_8b.json` and compare policy with verifier.
6. Show a trace where mutation is stopped before execution.
7. Show a repair proposal that fails safely with a typed tool error.
8. End with the limitation: 36 cases are portfolio evidence, not SQL SOTA.

## 11. Recommended resume bullets

- Built a bounded tool-using agent runtime combining LLM planning,
  schema/registry validation, policy-gated execution, verification, repair,
  deterministic replay, and MCP tools across radiography and SQL environments.
- Reduced unsafe SQL actions from 16.7% to 0% while improving task success from
  52.8% to 83.3% through policy-gated execution on 36 frozen cases; attributed
  all remaining failures to repair proposal execution or contract rejection.
- Productionized the runtime with authenticated FastAPI services, asynchronous
  PostgreSQL workers, MinIO artifact storage, idempotent jobs, circuit breaking,
  trace replay, provenance tracking, Docker Compose, and CI.

## 12. Claims to avoid

Do not say:

- "autonomous radiology agent";
- "clinically validated";
- "improved clinical accuracy";
- "fully automated measurement system";
- "general-purpose agent framework proven across domains";
- "SQL benchmark state of the art."

Say instead:

> RadMeasure is a production-style research prototype for bounded agent
> execution. It demonstrates reusable runtime interfaces and transparent safety
> evaluation in two environments, while retaining explicit human review and
> narrow empirical claims.
