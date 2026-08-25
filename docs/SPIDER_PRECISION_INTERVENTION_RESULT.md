# Preregistered Spider Precision Intervention

## Design qualification

Direct sign-stratified subsampling was rejected before preregistration because
beneficial candidates occur only on base-wrong cases and harmful candidates only
on base-correct cases. No question contained both signs. An auxiliary harmful
injection was also rejected before outcome evaluation: a database-grouped
original-versus-injected classifier achieved AUC 0.806 (SD 0.046), exceeding the
preregistered 0.60 qualification ceiling.

The retained design uses within-question, same-operator switches between an
existing non-neutral candidate and an existing neutral candidate. It fixes 1,080
questions, 108 database clusters, one candidate per question, and 612
non-neutral proposals. The target precisions are 0.15, 0.25, 0.35, 0.45, and
0.50. The preregistration was hashed before selector outcomes:

SHA-256: af442cc336d18710e00c14a212feb112c079935d79f2931409e23481e804ff78

## Result

| Precision | Frozen predicted gain | Observed gain | Reliably beats no-op and apply-all |
|---:|---:|---:|:---:|
| 0.15 | +0.08 pt | -2.22 pt | No |
| 0.25 | +2.07 pt | -5.56 pt | No |
| 0.35 | +6.10 pt | -5.56 pt | No |
| 0.45 | +12.85 pt | -2.50 pt | No |
| 0.50 | +17.08 pt | +2.59 pt | No |

The preregistered curve-agreement test fails (Spearman rho 0.359, p=0.553;
required rho >0.80). The phase predicted the first reliable condition at
precision 0.25; no observed condition reliably beats both global actions.

## Interpretation

The Spider base shift remains an association, not a causal identification of
proposal precision. Marginal precision alone does not recover learnability even
when questions, base predictions, candidate count, operator type, split, and
non-neutral mass are controlled. Candidate content and feature--advantage
fidelity are missing variables. The synthetic three-axis phase is therefore a
first-order diagnostic that is falsified as a sufficient predictive model by
this preregistered intervention.
