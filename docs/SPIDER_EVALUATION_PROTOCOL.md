# Spider evaluation protocol and selection-bias audit

## Status of the existing development-set result

The reported Spider development-set result is **exploratory**, not a preregistered
confirmatory test. Within its single recorded cross-validation run, every outer test
database is isolated correctly:

- TF-IDF vocabulary and both SGD classifiers are fitted on training databases only.
- `risk_weight` and the intervention threshold are selected on the calibration
  databases only.
- The selected policy is applied once to the corresponding test databases.
- Train, calibration, and test database sets are pairwise disjoint in every fold.

However, candidate operators and the initial model specification were developed in
the same project before and around the first inspection of Spider development-set
results. Consequently, the 33.17% to 44.68% result must not be described as an
untouched final test, even though there is no per-example gold leakage.

The workspace contains one recorded selector result from this development cycle. Its
timestamp follows the candidate artifact by approximately two minutes, and the
selector source predates that result by approximately one minute. This supports, but
cannot cryptographically prove, that no broad post-result hyperparameter search was
performed. The result remains exploratory regardless.

## Frozen confirmatory protocol

### Scope of confirmation

The current candidate generator is **development-set-designed**. Every current edit
family—question-grounded literals, aggregation substitution, ordering, DISTINCT,
LIMIT, comparator substitution, Boolean-connective substitution, and column
substitution—was implemented during the Spider development cycle. Candidate ordering
and the limit of 40 are development choices as well. A new database split cannot
remove this method-level selection bias.

There is no credible pre-development implementation in this workspace, so a
retrospective "introduced before versus during dev" comparison is not identifiable.
All operator families must be tagged `dev-developed`; inventing a clean subgroup
after seeing outcomes would create another selection bias. Frozen family-wise oracle
coverage may be reported diagnostically, but it cannot establish absence of dev
specialization.

Consequently, confirmation applies only to the complete, frozen pipeline:

- **Candidate generator:** dev-developed; its confirmatory result tests transfer of a
  dev-designed edit space and does not validate an independently specified generator.
- **Selector, threshold, and risk-control procedure:** developed without confirmatory
  labels and eligible for confirmatory evaluation once frozen.
- **Interpretation:** low confirmatory oracle headroom may reflect dev specialization,
  intrinsically harder residual errors, or both. The present design cannot identify
  these causes without a separately designed generator or external domain.

Before any confirmatory strong-base experiment, freeze the following choices:

1. Candidate operators and their ordering.
2. Selector features and model family.
3. Hyperparameter grid and calibration objective.
4. Database-grouped split assignment and random seeds.
5. Primary metrics: base execution accuracy, candidate-oracle headroom, learned
   absolute gain, learned/oracle recovery ratio, benefit count, harm count, and net.
6. Subgroups: Spider easy, medium, hard, and extra-hard.

### Confirmatory decision rule

The primary estimand is the database-clustered absolute execution-accuracy change of
the frozen learned policy relative to no repair. Confirmatory support requires both:

1. net gain **at least 3.00 percentage points**, and
2. the lower bound of a two-sided **database-clustered 95% bootstrap confidence
   interval greater than 0**.

Both conditions are required. The 3-point threshold is fixed as the minimum effect
considered practically meaningful for deployment. Candidate-oracle headroom,
learned/oracle recovery, benefit, harm, coverage, test-suite accuracy, and difficulty
strata are secondary diagnostics and cannot substitute for failure of the primary
rule. If oracle headroom is below 3 points, the result is interpreted as an edit-space
coverage boundary rather than selector failure.

### One-look stopping rule

The confirmatory gold is evaluated exactly once after every required artifact below
has been frozen. After that evaluation, no candidate operator, candidate ordering,
candidate limit, feature, model, hyperparameter grid, threshold rule, decoding choice,
or primary analysis may be changed and re-evaluated on the same confirmatory split.
The study stops regardless of effect size or confidence interval. Any later method
change is exploratory and requires a newly sealed split or external test set.

Spider train databases are the development population for all further feature,
candidate, model, threshold, and LTT work. A hash-selected subset of train databases
must be sealed before its labels are evaluated and used as the genuinely unseen
confirmatory split. Spider dev may be used as an external replication set only with
an explicit disclosure that its weak-base results were previously inspected. It is
not an untouched final test.

### Mandatory artifact order

The following order is mandatory and must be recorded in a version-2 seal:

1. Select the base model for reproducibility and training-data compatibility, not
   because it has the highest Spider-dev leaderboard score. Record model/checkpoint
   identity, training-data provenance, decoding configuration, prompt/template hash,
   and software version. A model fine-tuned on the complete Spider train set is
   ineligible for testing on the sealed Spider-train databases.
2. Generate SQL without confirmatory gold execution; freeze the ordered outputs and
   their SHA-256.
3. Generate candidates without confirmatory gold execution; freeze candidate actions,
   SQL strings, ordering, limit, generator-code hash, and corpus SHA-256.
4. Freeze selector checkpoint, threshold/LTT policy, bootstrap implementation, random
   seeds, and analysis code.
5. Only then unlock gold execution and perform the single confirmatory evaluation.

Public CodeS dev predictions used in the exploratory stress test are not eligible as
confirmatory outputs for the sealed Spider-train split because the released Spider
fine-tuned models were trained on the full Spider training set. Any subsequent method
change creates a new exploratory version and requires a new seal and new independent
test population.

### Base-pipeline redevelopment and version 3

The post-run audit found that the clean T5 control fails primarily through hallucinated
schema identifiers. Repairing schema serialization, adding database values or
demonstrations, increasing context length, or adding schema-valid constrained
generation changes the definition of the base pipeline and therefore the development
error distribution. Version 2 is invalidated before creation and before any
confirmatory gold execution. The next eligible preregistration is version 3.

During base-pipeline selection, only the 116 development databases may be used. The
allowed metrics are base execution accuracy, SQL executability, schema-identifier
validity, and base failure taxonomy. Candidate-oracle headroom, proposal performance,
selector performance, selector thresholds, and every sealed outcome are forbidden.
Once the base pipeline is fixed, candidate operators must be redesigned for that
base's errors, and proposal/selector training and calibration must restart from
scratch. Only the fully frozen rebuilt system may enter a version-3 seal.

## Accuracy-gradient analysis

Cross-model points and within-model difficulty strata answer different questions and
must not be collapsed into one causal curve:

- Cross-model results test external validity but confound accuracy with error type.
- Difficulty-stratified results within a fixed strong model probe how edit-space
  coverage and realized gain change as errors become harder.

For every model and stratum, report separately:

`oracle headroom = candidate oracle accuracy - base accuracy`

and

`selector recovery = learned gain / oracle headroom`.

This distinguishes failure of the candidate edit space from failure of advantage
estimation.
