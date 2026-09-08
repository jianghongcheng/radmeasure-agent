# Evaluation

## Historical selective-repair study

The archived `learned_expected_gain` selector was evaluated with a 20% intervention
budget and five patient-grouped folds (split seed 2027). The source loader pools
predictions from seeded base-model runs: 176 image identifiers contribute three
records each, for 528 case-records. These are repeated predictions of the same
cases, not 528 independent patients or new runtime-agent requests.

The learned selector ranks candidate axis corrections built from ensemble
geometry. It is an offline research policy, distinct from the rule-based
KEEP / REPAIR / STOP controller in this release.

| Metric | Value |
| --- | ---: |
| Mean HVA/IMA error before repair | 2.677705° |
| Mean HVA/IMA error after repair | 2.539686° |
| Intervened case-records | 106/528 (20.1%) |
| Mean reduction among intervened records | 0.687493° |
| Records with increased error | 43/528 |
| Records with error increase greater than 0.5° | 32/528 |

Each record's error is the mean absolute error of HVA and IMA. Values above pool
all case-records, rather than averaging fold means with equal weight. The gain on
intervened records is a net average, not an improvement on every selected case.
The archived paired bootstrap, clustered by image identifier, estimates the
overall reduction at 0.1380° (95% interval 0.0438° to 0.2441°). This interval does
not establish patient-level clinical generalization.

### Provenance and recalculation

[Aggregate result](results/selective_repair_summary.json) contains only counts,
metrics, and the source artifact hash. No images, patient records, or model weights
are included. Recalculate from an authorized local copy of the original artifact:

```bash
python scripts/summarize_repair_study.py /path/to/selector_baselines_fixed20.json
```

The source is `selector_baselines_fixed20.json`; the accompanying historical
`reconciled_policy_metrics.json` has SHA-256
`35b976ad14b853d78e965e6dd5d39f99cd45528b72e4f0c61489c87aa5d53ef9`.
The script reproduces aggregate arithmetic, not model training or fresh inference.
The full historical study depends on separately retained checkpoints, data, and
research scripts. Current model accuracy must be measured with a newly identified
dataset split and checkpoint; it cannot be inferred from these historical results.

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
