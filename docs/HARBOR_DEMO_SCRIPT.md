# Three-minute Harbor demo script

## 0:00–0:20 — What the system is

Show the README architecture and say:

> RadMeasure is a bounded tool-using agent runtime. The LLM proposes, policy
> authorizes, deterministic tools execute, and a verifier decides. I packaged
> its frozen reliability suite as a Harbor-compatible evaluation environment.

## 0:20–1:30 — Run Harbor without acceleration

Show the command and let the terminal output remain visible:

```bash
./scripts/run_harbor_sql_suite.sh v3-frozen
```

While it builds/runs, point out:

- the Agent and verifier use separate Docker containers;
- both phases have network access disabled;
- expected labels exist only in the verifier image;
- `submission.json` crosses the boundary as a declared Harbor artifact.

## 1:30–2:30 — Results and negative evidence

Open the generated `verifier/metrics.json` and show:

- the confirmatory Harbor v3 replay: 98/108 successful tasks after policy and
  verification;
- 25 unsafe proposals and zero unsafe executions;
- six output-contract rejections and four wrong-action outcomes;
- the separate 24-case result where the deterministic rule planner beats
  Qwen3-8B (75.0% vs. 62.5% action accuracy).

Say explicitly:

> I use the negative result to constrain the architecture: the LLM is an intent
> proposer, not the authorization boundary.

## 2:30–3:00 — Claim boundary

Say:

> This is a frozen agent-reliability benchmark, not a SQL leaderboard.
> Radiography is the safety-critical application environment. RadMeasure is a
> research prototype, not a medical device and not approved for patient care.

End on the GitHub repository URL. Upload the recording as an unlisted YouTube
video, then add the real URL to the README and resume. Do not add a placeholder
or claim a demo is available before the upload succeeds.
