# Harbor evaluation

## Confirmatory v3 task

The primary Harbor task now packages all 108 cases from the separately frozen
v3 confirmatory suite. The agent container sees requests, current SQL, and
schema declarations. Database seed rows, expected actions, gold queries, and
scoring code remain in a separate verifier container; both phases run without
network access.

Local Docker execution on 2026-08-27 reproduced the non-containerized result:

| Run | Reward | Passed | Unsafe proposals | Unsafe executions |
|---|---:|---:|---:|---:|
| Oracle | 1.000 | 108/108 | 0 | 0 |
| Frozen Qwen3-8B + policy/verifier | 0.907 | 98/108 | 25 | 0 |

Run it with:

```bash
./scripts/run_harbor_sql_suite.sh v3-oracle
./scripts/run_harbor_sql_suite.sh v3-frozen
```

The task lives at `harbor/tasks/radmeasure_sql_repair_v3/`. Its frozen replay
uses the preregistered generations rather than sampling the model again.

## Historical v1 task

RadMeasure also retains its historical frozen 36-case SQL-repair evaluation as a
versioned native Harbor v1
task. The goal is reproducibility and verifier isolation, not a new SQL
leaderboard claim.

## Why Harbor

The original ablation already generated each Qwen3-8B proposal once and replayed
it through multiple runtime configurations. Harbor adds an independent execution
boundary:

```text
public cases in agent container
        ↓
agent writes /app/submission.json
        ↓ artifact transfer
separate verifier container
        ↓
hidden expected actions and outputs
        ↓
reward.json + metrics.json
```

The agent image contains requests and current SQL only. Expected actions, oracle
SQL, and scoring code are built into the separate verifier image. Both phases
run without network access.

## Install

Harbor requires Python 3.12 or newer. The tested version is 0.22.0.

```bash
uv tool install 'harbor>=0.22,<1'
```

Docker must be running and accessible to the current user.

## Run

Validate the task with the perfect reference solution:

```bash
./scripts/run_harbor_sql_suite.sh oracle
```

Replay the exact Qwen3-8B proposals used by the published component ablation:

```bash
./scripts/run_harbor_sql_suite.sh frozen
```

The frozen replay is intentionally model-free: it measures the same proposals
without generation randomness. It is not presented as a new inference run.

## Verified result

Local Docker execution on 2026-08-27 produced:

| Run | Reward | Passed | Unsafe proposals | Unsafe executions |
|---|---:|---:|---:|---:|
| Oracle | 1.000 | 36/36 | 0 | 0 |
| Frozen Qwen3-8B + policy/verifier | 0.833 | 30/36 | 6 | 0 |

The six residual failures remain five SQL execution failures and one exact
output-contract mismatch. This reproduces the existing benchmark result while
moving the expected outputs out of the agent-visible environment.

## Task contents

```text
harbor/tasks/radmeasure_sql_repair_v1/
├── task.toml
├── instruction.md
├── environment/
│   ├── Dockerfile
│   └── cases.json
├── solution/
│   ├── solve.sh
│   └── oracle_submission.json
└── tests/
    ├── Dockerfile
    ├── expected.json
    ├── test.sh
    └── verify.py
```

`scripts/export_harbor_sql_suite.py` deterministically regenerates public cases,
hidden expectations, and the oracle submission from the frozen source suite.
CI regenerates these files and fails if committed task artifacts drift.

## Claim boundary

This task demonstrates container isolation, hidden-label verification,
standardized artifacts, and exact replay of the existing safety result. It does
not establish SQL state of the art, clinical validity, or broad domain
generality. Live model comparisons should be reported separately from the
frozen replay.
