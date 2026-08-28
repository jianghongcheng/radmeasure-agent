# RadMeasure: 90-Minute Interview Practice Session

Use this session to prepare for Applied AI, Agent Systems, ML Systems, and
Applied Scientist interviews. Do not memorize sentences. Run the code, explain
the evidence, and defend the claim boundary.

## Evidence card

- Development suite: 120 cases, five schemas, 24 failure-family clusters.
- Confirmatory suite: 108 separately frozen cases, six schemas absent from
  development, 18 failure-family clusters.
- Confirmatory result: 73/108 LLM-only versus 98/108 policy plus verifier.
- Safety boundary: 25/25 unsafe proposals blocked; zero observed unsafe
  executions does not imply zero population risk.
- Output boundary: six incorrect outputs rejected by the verifier.
- Paired cluster-bootstrap improvement: +23.22 points, 95% CI +7.41 to +40.74.
- Harbor replay: oracle 108/108; frozen Qwen plus runtime 98/108 in separate,
  network-isolated agent and verifier containers.
- Test suite: 106 passing tests.

## 0–10 minutes: two-level project pitch

Give both answers without notes.

### 30 seconds

Explain the problem, the bounded execution architecture, and one confirmatory
result. Avoid implementation details.

### 3 minutes

Cover:

1. why the LLM is a proposal generator rather than an authority;
2. schema validation versus registry validation versus policy;
3. deterministic execution and post-action verification;
4. why `STOP` can be the correct outcome;
5. v2 development versus v3 confirmatory evidence;
6. one limitation and the next experiment.

Pass condition: a listener can draw the system and state the defended claim.

## 10–30 minutes: adversarial interviewer round

Answer each in under 90 seconds, then accept one follow-up.

1. Why is this an agent rather than a fixed workflow?
2. Why can the verifier not replace policy?
3. Why did schema and registry checks change no aggregate result?
4. Why did policy plus verifier have the same success count as policy alone?
5. Are 25 blocked mutations a meaningful safety result or a constructed test?
6. What exactly was frozen before v3 generation?
7. Why bootstrap failure-family clusters instead of 108 individual cases?
8. What does Harbor add beyond Docker Compose?
9. How could hidden labels leak into the agent container?
10. Which part was your design judgment, and which implementation was
    AI-assisted?

Pass condition: every answer distinguishes observed evidence from inference.

## 30–55 minutes: coding exercise

Implement a pure authorization function:

```python
def authorize(proposal: dict, registered_tools: set[str]) -> tuple[bool, str]:
    """Authorize one bounded SQL proposal without executing it."""
```

Requirements:

- `STOP` succeeds without a tool call;
- only `KEEP` and `REPAIR` may request execution;
- the tool must be registered;
- only one read-only `SELECT` is eligible;
- mutation and multiple statements fail closed;
- malformed proposals return typed reasons instead of raising.

Write tests first for:

1. valid `KEEP`;
2. valid `REPAIR`;
3. expected `STOP`;
4. unregistered tool;
5. `DROP`, `UPDATE`, and `INSERT` hidden by whitespace or casing;
6. `SELECT ...; DROP ...`;
7. missing and incorrectly typed fields.

Then explain why production defense also needs parsing, a read-only database
role, timeouts, sandboxing, and audit logs. Regex is not the security boundary.

## 55–70 minutes: debugging exercise

Scenario: after a policy change, task success remains 98/108, but the verifier
reports three unsafe executions.

Work through this sequence:

1. reproduce with one frozen generation artifact;
2. identify the smallest failing case;
3. compare proposal, authorization reason, executed SQL, and final decision;
4. state competing hypotheses before editing code;
5. instrument the policy-to-executor boundary;
6. fix the narrow cause;
7. add a regression test;
8. replay all 108 cases and compare the failure taxonomy.

Pass condition: do not patch the metric or verifier to hide the failure.

## 70–82 minutes: system-design extension

Design a multi-tenant version. Address:

- API authentication and role-scoped tool permissions;
- idempotency keys and atomic worker claims;
- lease expiry and crash recovery;
- PostgreSQL state and content-addressed MinIO artifacts;
- per-tenant model and policy versions;
- trace correlation and replay;
- false-block monitoring;
- safe rollout and rollback of a policy change.

State which guarantees are deterministic and which remain statistical.

## 82–90 minutes: self-review rubric

Score each dimension from 0 to 2:

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| Problem framing | feature list | partial motivation | clear failure and user impact |
| Architecture | names components | explains flow | explains distinct trust boundaries |
| Evidence | percentages only | counts and suite | counts, uncertainty, and limitations |
| Coding | incomplete | works on happy path | tested edge cases and typed failures |
| Debugging | guesses | reproduces | minimizes, instruments, fixes, regresses |
| Ownership | vague | names tasks | separates judgment, implementation, and AI assistance |

A score below 9/12 means repeat the session. For missed questions, consult
`APPLIED_SCIENTIST_INTERVIEW_GUIDE.md`, then rerun the relevant test or
evaluation rather than adding another memorized answer.
