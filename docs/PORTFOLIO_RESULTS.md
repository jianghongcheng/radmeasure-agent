# Medical evidence boundaries

SQL-repair benchmarks belong to ContractSQL, not medical evidence.
See [release validation](MEDICAL_RELEASE_VALIDATION.md) for current software checks.

## Historical artifact — not rerun for this release

Earlier analysis reported 176 unilateral HVAngleEst evaluations:
HVA MAE 3.563 degrees and IMA MAE 2.130 degrees. It averaged undirected axes
from three persisted MedImageInsight spatial readouts (seeds 17, 42, 73).

The per-image artifact is local and not distributed:
`data/processed/hvangleest/medimageinsight_locked_test_eval.json`.
That artifact and current manifests came from different split states.
No patient-disjoint, prospective clinical, or fresh live-inference accuracy
claim is supported by this release. Historical numbers are not current
dashboard performance.

Retained research reports describe their own protocols and limitations.
They do not establish validation of the current deployed model.
Synthetic tests establish software behavior, not medical accuracy.
