# Phase robustness and cross-base transfer

## Frozen protocols

The robustness grid was frozen before execution in
`protocols/SYNTHETIC_PHASE_ROBUSTNESS_PREREGISTRATION_V1.json` (SHA-256
`16345da72142c07fbd8c86d4109f4bb27dd282c10b9eaf274749541c3bc4b960`).
The first two requested seed cells proved to be interior rather than boundary
cells. A second audit therefore selected existing 80% boundary cells by a
mechanical rule recorded before new seeds were run in
`protocols/SYNTHETIC_BOUNDARY_SEED_AUDIT_V1.json` (SHA-256
`dde3c9053e22ace5a3d22ec0fab3e37c091bc47248794f9286f544ffd8dbc322`).

## Seed repeatability

At the preregistered interior condition (precision 0.5, noise 0.2), the mean
probability that selectivity beats the best global action is 99.75% (between-seed
SD 0.79%) for 30 clusters and 100% for 100 clusters. All ten seeds satisfy the
registered reliable-win rule in both conditions.

The mechanically selected boundaries are less absolute, as expected. On the
low-precision/no-op side (precision 0.2, noise 0.1, 100 clusters), the new-seed
win probability is 80.5% +/- 7.9%; seven of ten seeds individually meet the
80% rule. On the high-precision/apply-all side (precision 0.8, noise 0.2, 176
clusters), it is 86.5% +/- 4.4%, and all ten seeds meet the rule. Thus the broad
region is stable, but a binary boundary label on the low-precision side moves
by roughly one grid cell across seeds.

## Advantage-definition sensitivity

The magnitude-aware simulator preserves continuous evaluation utility while
changing the positive supervision definition: strict (`Delta > 0`), meaningful
(`Delta > 0.5` synthetic units), or rank-based (top 20%). Across 30 registered
phase cells, the sign of the learned margin agrees in 100% of cells for every
pair. Reliable-region Jaccard similarity is 0.964 for strict versus meaningful,
0.893 for strict versus rank, and 0.926 for meaningful versus rank. The phase
structure is therefore not created solely by the zero threshold, although the
exact reliable boundary is definition-dependent.

## Candidate count and dependence

Precision alone is not sufficient. The preregistered candidate-set experiment
holds marginal precision fixed and varies 2, 5, 10, or 20 candidates per case,
with independent candidates or a shared case effect. At precision 0.5, noise
0.2, and 100 clusters, increasing the independent candidate count from 2 to 20
raises learned gain over no-op from 26.58 to 60.38 points, because the best
candidate improves. At the same time, margin over always applying the
highest-scored candidate falls from +7.50 to -0.25 points; the probability of
beating apply-all falls from 100% to 30%. With shared-case dependence, the same
pattern holds (+6.09 to -0.31 points; 100% to 32.5%). Candidate count and
within-case dependence are therefore missing phase variables, not harmless
nuisance choices.

## Spider cross-base operator transfer

No additional model was trained. Existing database-grouped CodeS experiments
already provide a stricter transfer test than a second weak base: train the
selector on four prediction sources and evaluate on a held-out fifth base.
The frozen operator family retains 4.64--5.51 points of candidate-oracle
headroom on CodeS 1B/3B/7B/15B, but leave-one-base-out selector gains are
-0.39, -0.68, -0.68, and -0.97 points. Within-base retraining is also negative.
This rules out the strong claim that the weak-base +11.51-point result reflects
a generally transferable operator/selector. It supports the narrower phase
claim: exact labels reveal headroom, while useful-proposal density and
predictability remain base-distribution dependent.

## Zero-learning execution-only baseline

The policy edits if and only if the base SQL fails to execute, then takes the
first executable candidate in the frozen operator order. It uses neither a
learned score nor gold information at decision time. On Spider clean it gains
+2.06 execution-accuracy points (136 benefits, zero harms; database-clustered
95% CI 0.52--3.74), compared with +0.061 points for learned selection. It gains
exactly zero on Spider weak and on CodeS 1B/3B/7B/15B. The clean-base
recoverable invalid-query regime is therefore better handled by execution
filtering; it is not evidence of learned-selector superiority.

The prespecified cascade applies execution-only repair first and retains the
existing cross-fitted learned-selector decision on all remaining cases. It
gains +2.121 points (144 benefits, 4 harms; database-clustered 95% CI
0.58--3.79). The learned stage therefore adds only +0.061 points beyond
execution-only repair, or four net cases among 6,602. The execution stage has
structural zero harm under execution accuracy because it triggers only on an
already-invalid base query; conditional on nonzero advantage, this subset has
proposal precision one.

## Finite-sample risk-control audit

For 20 predeclared policies and familywise level 0.05, zero observed harms can
certify risk epsilon only when `(1-epsilon)^n <= 0.05/20`. A 5% harm guarantee
requires 117 independent calibration identifiers, while 10% requires 57. The
five X-ray folds contain only 20--33 calibration patients, yielding best-case
zero-harm bounds of 16.6--25.9%. Independent cluster count therefore affects
not only fitting stability but whether low-risk behavior can be certified at
all.

## Paper-safe conclusion

The new evidence strengthens the phase hypothesis but also refines it. Proposal
precision, label noise, and cluster count explain the first-order transition;
candidate-set cardinality and dependence move the boundary. The phase should be
presented as a controlled diagnostic, not a complete law or post-hoc universal
predictor.
