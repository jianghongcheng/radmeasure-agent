# Evaluation

## Reproduce software checks

```bash
pip install -e '.[dev]'
python -m pytest -q
python -m compileall -q src
radmeasure --question "Measure HVA and IMA"
python scripts/evaluate_agent_decisions.py
```

The CLI and policy check use synthetic inputs. The policy script writes local
results under ignored `outputs/`; its success rate is not clinical accuracy.
Demo citations and similar cases are synthetic placeholders, not clinical sources.
Test counts depend on optional dependencies and artifacts. Inspect skip reasons
rather than treating skipped tests as validation.

Tests cover unsupported requests, contract failures, bounded repair, mandatory
upload review, persistence, and execution lineage.

## Evidence modes

| Mode | Demonstrates | Does not demonstrate |
| --- | --- | --- |
| Synthetic demo | Control flow, checks, and review routing | Image-model accuracy |
| Locked replay | Behavior on an identified saved artifact | Fresh inference or new patient-disjoint evaluation |
| Configured live inference | Execution with particular weights | Clinical utility or validated deployment |

Model checkpoints and historical per-image artifacts are not included. This
public release therefore does not claim a reproducible live-model accuracy score.
Earlier artifacts and preparation manifests used different split states;
historical metrics are not current release results.

For a new medical accuracy evaluation, identify dataset version, patient-level
split, model hash, preprocessing, eligible cases, and per-angle errors. Report
exclusions, failures, and review coverage alongside accuracy. Keep sensitive
inputs and per-patient outputs out of Git.

## Limits

- Unit tests and mocks do not validate a live model or external service stack.
- Geometry consistency does not establish correct anatomical localization.
- Human review is a mechanism, not evidence of clinical effectiveness.
- No prospective clinical, production-user, or continuous-load SLO evidence is
  demonstrated by these checks.
