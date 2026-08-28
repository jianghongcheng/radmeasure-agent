# RadMeasure: Applied Scientist Interview Guide

This guide prepares the project owner to discuss RadMeasure in Applied
Scientist, Applied AI, and Agent Reliability interviews. It emphasizes
hypotheses, controlled evaluation, failure analysis, scientific limitations,
and coding—not only system architecture.

## 1. Thirty-second introduction

> I built RadMeasure to study when an LLM-proposed action should be allowed to
> execute. The LLM is an untrusted proposal generator: schema and registry
> checks validate the action, a deterministic policy authorizes it, registered
> tools execute it, and a verifier evaluates the result. After a 120-case
> development ablation, I froze a separate 108-case confirmatory suite with six
> new schemas before generation. Without retuning, policy and verification
> increased task success from 73 to 98 cases, blocked all 25 unsafe proposals,
> and rejected all six incorrect outputs. A separate 24-case planning
> benchmark showed that a deterministic rule planner outperformed Qwen3-8B,
> which motivated keeping the LLM behind deterministic authorization.

## 2. What is the research question?

**Question:** What problem does RadMeasure study?

**Answer:**

RadMeasure studies whether an LLM-generated tool action should be executed,
repaired, or rejected. The scientific question is not merely whether the model
can produce valid JSON. It is whether independently testable system boundaries
can improve task completion while preventing prohibited actions.

The core hypothesis is:

> Separating proposal generation, pre-action authorization, deterministic
> execution, and post-action verification produces more reliable behavior than
> giving an LLM direct execution authority.

## 3. Architecture questions

### Q1. Why not let the LLM call tools directly?

An LLM can produce syntactically valid but semantically unsafe actions. It may
select an unsupported protocol, hallucinate a tool, attempt a database
mutation, or follow a prompt-injection request. RadMeasure therefore treats the
LLM output as a proposal, not permission to act.

### Q2. What is the difference between a proposal and a plan?

A plan describes intended steps. A proposal is a concrete candidate action
submitted for authorization, such as calling `sql_query` with a specific SQL
statement. A proposal can be rejected without executing it.

### Q3. What does schema validation do?

Schema validation checks whether the model output has the required structure,
field types, and action names. It answers, “Is this request well formed?” It
does not answer, “Is this action safe?”

### Q4. What does the registry do?

The registry is an allow-list of tools and measurement protocols. It prevents
the model from creating capabilities simply by naming them. It answers, “Does
this supported capability exist?”

### Q5. What is the difference between policy and verifier?

- **Policy is pre-action:** may the proposed action execute?
- **Verifier is post-action:** is the resulting output acceptable?

A verifier cannot replace policy because destructive or unauthorized actions
must be blocked before execution.

### Q6. Why use deterministic tools?

Deterministic tools make execution reproducible, testable, and replayable. In
radiography, geometry code—not an LLM—computes the angle. In SQL repair, an
authorized query executes in a disposable SQLite environment.

### Q7. What do KEEP, REPAIR, and STOP mean?

- `KEEP`: accept a verified result.
- `REPAIR`: attempt one bounded correction when an eligible proposal exists.
- `STOP`: abstain because the request or result is unsupported, unsafe, or
  insufficiently reliable.

`STOP` is not necessarily a failure. For a prohibited request, stopping is the
correct task outcome.

### Q8. What makes this a bounded agent?

The system limits available tools, permissions, repair attempts, execution
time, and accepted output contracts. When uncertain, it fails closed instead
of expanding its authority.

### Q9. Why are trace and replay important?

A trace records the proposal, authorization decision, tool execution,
verification result, and final decision. Replay allows the same saved proposal
to be evaluated under another harness configuration without resampling the
LLM. This supports debugging, component attribution, and regression testing.

## 4. Experimental-design questions

### Q10. Why freeze the evaluation suite and model proposals?

LLM sampling can change between runs. If every configuration receives a new
generation, differences may come from sampling rather than the component being
tested. RadMeasure reuses the same frozen proposals across configurations to
isolate the effects of schema, registry, policy, and verification.

### Q11. What are the principal SQL results?

| Configuration | Successful tasks | Unsafe actions executed |
|---|---:|---:|
| LLM only | 94/120 | 19/120 |
| Schema validation | 94/120 | 19/120 |
| Registry validation | 94/120 | 19/120 |
| Verifier only | 94/120 | 19/120 |
| Policy only | 113/120 | 0/120 |
| Policy + verifier | 113/120 | 0/120 |

The key interpretation is that policy and verification protect different
boundaries. Policy blocked unsafe execution. Verification rejected five
incorrect outputs that policy alone admitted, although those rejections do not
increase task success because the current proposal generator has no valid
fallback for those cases.

### Q12. Why did schema and registry checks not improve success?

The frozen Qwen3-8B proposals were already syntactically valid and used
registered action names. Therefore, these checks were necessary interface
boundaries but were not the active bottleneck in this particular suite. Their
value is defensive, not demonstrated as an accuracy improvement here.

### Q13. Why are policy and policy-plus-verifier identical on this suite?

Both configurations produce 113 successful outcomes, but they are not
behaviorally identical. Policy alone accepts five incorrect outputs; the
verifier rejects them. Those cases remain unsuccessful because the bounded
runtime has no second valid proposal, so the task-success count is unchanged.

### Q14. What caused the seven residual failures?

They are proposal-quality failures: after the unsafe actions are blocked and
wrong outputs are rejected, no valid fallback proposal remains for seven
cases. None is a policy bypass. After authorization was introduced, the
dominant bottleneck shifted from unsafe execution to proposal quality.

### Q15. Why did the deterministic planner beat Qwen3-8B?

The benchmark has a small, explicit protocol registry and clear safety rules.
A rule planner directly represents those constraints, while the LLM must infer
them from text and can be distracted by unsupported or adversarial requests.
The result does not prove that rules always beat LLMs. It shows that an LLM
should not replace a simpler reliable component when the action space is small
and well specified.

**Likely follow-up: Why not put all of those constraints in the system prompt?**

The constraints should also appear in the prompt because that can reduce bad
proposals, but a prompt is not an authorization boundary. Following a prompt is
probabilistic: the model may ignore it, misunderstand it, or follow an
adversarial instruction. In this benchmark, Qwen3-8B still produced a 37.5%
unsafe-action rate despite receiving structured instructions. A policy rule is
structural and runs outside the model: a prohibited action cannot execute even
when the model proposes it. Prompting improves proposal quality; policy limits
execution authority. They are complementary rather than interchangeable.

### Q16. Is 120 cases enough to claim general safety?

No. The 120 frozen adversarial cases provide stronger component-level evidence
and reproducible failure attribution, not population safety or clinical
validity. The honest claim is “zero unsafe executions among 19 unsafe proposals
in this frozen suite.” Its Wilson 95% upper bound is 3.1% over all 120 cases, so
zero observed failures must not be described as zero true risk.

### Q17. How uncertain is the 94/120 to 113/120 improvement?

The comparison is paired because every configuration uses the same cases and
proposals. We bootstrap the 24 failure-family clusters rather than treating 120
correlated cases as independent. The estimated gain is +15.85 percentage
points, with a 2,000-resample cluster-bootstrap 95% interval of +4.17 to +31.67
points. This supports the component ablation while retaining substantial
uncertainty about deployment populations.

### Q18. How would you scale the evaluation?

The current version includes 120 development cases plus 108 separately frozen
confirmatory cases over six schemas absent from development. Next I would add
repeated model generations, larger independently authored suites, distribution
shift, false-block cost, token cost, and tool-failure injection.

### Q18a. Why is v3 stronger evidence than simply adding more v2 cases?

V3 was frozen before model generation, uses six schemas absent from v2, and
introduces different query and failure templates. The prompt, policy, verifier,
and scoring code were not tuned after observing its outcomes. Its 73/108 to
98/108 improvement has a paired 18-cluster bootstrap interval of +7.41 to
+40.74 points. This remains an engineering benchmark, but it tests transfer
more directly than paraphrasing development cases.

### Q19. How did you prevent data leakage?

The evaluation labels are hidden from the agent-facing environment, frozen
proposals are separated from verifier logic, and the Harbor package separates
the agent and verifier containers. In the imaging data, patient overlap in the
released HVAngleEst split was detected and replaced with a patient-disjoint
manifest. These controls reduce leakage, although they do not prove external
clinical generalization.

### Q20. Why use Harbor?

Harbor packages the task as isolated agent and verifier containers with hidden
labels and network-controlled execution. It makes the reliability evaluation
reproducible outside the application code and uses the same evaluation
substrate associated with Terminal-Bench-style agent tasks.

## 5. Applied Scientist reasoning questions

### Q21. What was the most important negative result?

The rule/registry planner achieved 75.0% action accuracy with a 16.7% unsafe
action rate, while Qwen3-8B achieved 62.5% accuracy with a 37.5% unsafe rate on
the frozen 24-case suite. This caused a design change: the LLM became an
optional intent proposer, while deterministic policy retained execution
authority.

### Q22. What did you learn from the ablation?

Reliability did not come uniformly from every added component. Schema and
registry checks were inactive on the frozen proposals; the verifier improved
correctness but did not prevent unsafe actions; policy produced the largest
change. The scientific value is identifying which boundary affects which
failure mode.

### Q23. What is the causal claim you can defend?

Because proposals and cases are held fixed, the within-suite outcome difference
can be attributed to changing the harness configuration. I cannot claim that
the measured magnitude generalizes to other models, databases, or clinical
settings.

### Q24. What metric would you optimize?

I would not optimize task success alone. A deployment objective should constrain
unsafe execution and false authorization while maximizing success among allowed
requests. Relevant metrics include task success, unsafe-action rate, false-block
rate, STOP precision and recall, repair utility, latency, token cost, and tool
calls per completed task.

### Q25. What is the next research question?

Once unsafe actions are blocked, proposal quality becomes the bottleneck. The
next question is whether a proposal generator can improve repair coverage
without increasing unsafe proposals or unnecessary interventions. This should
be tested with a larger frozen suite and matched inference budgets.

### Q26. How is this architecture domain independent?

The reusable state machine—propose, authorize, execute, verify, and decide—is
domain independent. The policy rules, tools, and verification contracts are
domain specific. Evidence currently comes from radiographic measurement and SQL
repair; broader generality remains an architectural hypothesis until evaluated
in additional environments.

### Q27. What would make you reject your own approach?

I would reject the added harness complexity if a simpler deterministic workflow
matched task success and safety, if policy created excessive false blocks, or
if the verifier could not detect meaningful failures. The project already shows
one case where the deterministic planner is preferable to the LLM.

## 6. Ownership and limitation questions

### Q28. How much of this system did you personally implement?

Do not memorize a generic answer. Before interviewing, make a truthful inventory
of four categories:

1. components you designed and implemented directly;
2. components implemented with AI coding assistance but reviewed and tested by
   you;
3. third-party components you integrated;
4. components you can explain experimentally but cannot yet explain line by
   line.

A safe answer, only if it is accurate, is:

> I owned the problem formulation, frozen evaluation design, ablation protocol,
> failure taxonomy, and interpretation of the negative results. I used AI coding
> assistance for parts of the implementation, then reviewed them through tests
> and controlled runs. I can explain the runtime boundaries and evaluation
> behavior, but I will distinguish those from implementation details I need to
> inspect rather than pretend I wrote every line unaided.

Never claim sole authorship of code you did not inspect. Be ready to point to
specific commits, tests, and design decisions that demonstrate your ownership.

### Q29. Is RadMeasure really an agent if most workflows contain one action?

RadMeasure is a bounded single-action execution runtime with one optional repair
step, not a general autonomous long-horizon agent. It still evaluates important
agent-system boundaries—model planning, tool authorization, execution,
verification, abstention, and replay—but it does not demonstrate open-ended
multi-step planning.

The next extension would represent an explicit state machine with a bounded
action budget, observation-dependent replanning, loop detection, idempotent
tools, accumulated risk, and trajectory-level evaluation. Calling the current
system a long-horizon or autonomous agent would overstate the evidence.

### Q30. Why does the verifier matter if policy and policy-plus-verifier have
the same SQL success rate?

On the SQL suite, it does not improve the final success count; it changes how
one failure is classified through an output-contract rejection. That is an
honest negative result. On the radiographic path, the verifier has a different
domain-specific role: it checks measurement geometry, confidence, discrepancy,
and repair eligibility before returning `KEEP`, `REPAIR`, or `STOP`.

However, architecture alone is not evidence of benefit. To claim that the
radiographic verifier improves outcomes, I must show a matched ablation with and
without it on held-out cases, including false rejection and unsafe acceptance.
Until then, I describe its implemented role rather than claiming a measured
benefit.

## 7. Coding questions likely to appear

### Coding task 1: Implement a policy gate

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Proposal:
    tool: str
    arguments: dict


def authorize(proposal: Proposal, registry: dict[str, dict]) -> str:
    """Return EXECUTE or STOP without running the proposed tool."""
    if proposal.tool not in registry:
        return "STOP"

    spec = registry[proposal.tool]
    if not spec["validate"](proposal.arguments):
        return "STOP"

    if spec.get("read_only") and proposal.arguments.get("mutates_state", False):
        return "STOP"

    return "EXECUTE"
```

Be ready to discuss fail-closed behavior, exception handling, immutable audit
records, and why validation must precede execution.

### Coding task 2: Compute evaluation metrics

```python
def evaluate(records: list[dict]) -> dict[str, float]:
    n = len(records)
    if n == 0:
        raise ValueError("records must not be empty")

    successes = sum(r["task_success"] for r in records)
    unsafe_executions = sum(
        r["proposal_unsafe"] and r["executed"] for r in records
    )
    unsafe_proposals = sum(r["proposal_unsafe"] for r in records)

    return {
        "task_success_rate": successes / n,
        "unsafe_execution_rate_per_task": unsafe_executions / n,
        "unsafe_admission_rate": (
            unsafe_executions / unsafe_proposals if unsafe_proposals else 0.0
        ),
    }
```

The interviewer may ask why the denominator matters. Per-task risk and
conditional unsafe-admission risk answer different questions and should both be
reported with raw counts.

### Coding task 3: Build a replayable state machine

Be prepared to implement:

1. parse a proposal;
2. authorize it without side effects;
3. execute only when authorized;
4. convert exceptions into typed failures;
5. verify the output;
6. append every transition to an immutable trace.

Discuss idempotency, timeouts, retry budgets, deterministic inputs, and artifact
hashes.

### Coding task 4: SQL safety validation

The interviewer may ask you to reject mutation, multiple statements, comments,
or unregistered tools. Explain why string matching alone is insufficient and
why a parser, read-only database role, transaction boundary, and sandbox should
provide defense in depth.

### Coding task 5: Paired statistical comparison

Given outputs from two configurations on the same cases, construct the 2x2
discordance table for McNemar's test. Explain why an unpaired proportions test
would discard the matched-case structure.

## 8. Questions to ask the interviewer

- How does your team separate model capability from execution authorization?
- Which agent failures are currently most expensive: planning, tool execution,
  verification, or recovery?
- Are evaluations replayed on frozen trajectories or regenerated on every run?
- How are unsafe-action and false-block trade-offs measured?
- What evidence is required before a new model or tool policy is deployed?

## 9. Claims to avoid

Do not say:

- “RadMeasure is an autonomous radiology agent.”
- “The system is clinically validated.”
- “Zero unsafe actions proves safety.”
- “The LLM improved clinical accuracy.”
- “The verifier prevents dangerous actions.”
- “The architecture is proven to generalize to every domain.”

Say instead:

- “RadMeasure is a bounded tool-using agent runtime.”
- “It is a research prototype, not a medical device.”
- “After a 120-case development ablation, policy blocked 25 of 25 unsafe
  proposals in a separately frozen 108-case confirmatory suite over six new
  schemas; Harbor reproduced 98/108 outcomes in isolated containers.”
- “The verifier checks post-action contracts; policy controls authorization.”
- “The runtime was evaluated in radiographic measurement and SQL repair.”

## 10. Three-day hands-on preparation plan

This guide is not a substitute for operating the project. Complete these steps
before memorizing answers.

### Day 1: Trace one request through the code

Read the README, then follow one request through:

- `src/geomed_copilot/planner.py`;
- `src/geomed_copilot/protocols.py`;
- `src/geomed_copilot/bounded_runtime.py`;
- `src/geomed_copilot/sql_environment.py`;
- `src/geomed_copilot/agent_controller.py`.

For each file, write one sentence answering: what enters, what decision is made,
what leaves, and what can fail?

### Day 2: Run and perturb the system

1. Run the relevant tests.
2. Run the frozen SQL ablation.
3. Run the Harbor evaluation.
4. Change one policy rule locally—for example, temporarily weaken or strengthen
   the read-only SQL authorization rule.
5. Predict which cases should change before running the evaluation.
6. Run it, record the changed counts, explain every difference, and revert the
   local experiment.

The temporary modification is a learning exercise, not a result to commit.

### Day 3: Explain without notes

Give a three-minute explanation covering the problem, architecture, ablation,
negative result, limitation, and next experiment. Record yourself. Mark every
place where you use a term you cannot explain with an example, then return to
the code or test that implements it.

Only after these three days should this question bank be used for mock
interviews.

## 11. Final answer framework

For almost every project question, use this order:

1. **Problem:** what failure could occur?
2. **Hypothesis:** why should the proposed boundary help?
3. **Implementation:** what component did you build?
4. **Evaluation:** what was held fixed and what changed?
5. **Result:** report counts before percentages.
6. **Limitation:** what does the result not prove?
7. **Next step:** what experiment would reduce the remaining uncertainty?

This structure demonstrates the combination expected from an Applied Scientist:
implementation ability, experimental discipline, quantitative reasoning, and
honest interpretation.
