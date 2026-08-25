# Selective Structured Correction: Reproducible Experiment Report

## Claim supported by current evidence

Unconditional correction frequently damages adequate measurement geometry. Across
three-axis HVA/IMA, two-axis Cobb, and discrete Cobb candidate-pair correction,
component-level detector disagreement supports selective intervention under a
deterministic executor. The contribution is selective structured correction, not a
DSL or a generic medical agent.

The executable measurement operator is more than an interpretability device. For
every candidate edit \(a\), deterministic execution gives an exact counterfactual
advantage label
\(\Delta(s,a)=L(G(s),y)-L(G(T(s,a)),y)\). The selector therefore learns edit
advantage from exact executor supervision rather than a noisy preference, proxy
reward, or learned critic target.

## Frozen results

### HVA/IMA — patient-grouped five-fold CV

| Method | Mean MAE | Harm | Coverage |
|---|---:|---:|---:|
| No repair | 2.801° | 0% | 0% |
| Unconditional learned repair | 2.890° | 55.6% | 100% |
| Conservative ensemble gain | **2.660°** | **5.8%** | 17.5% |
| Balanced ensemble gain | **2.644°** | 8.9% | 24.1% |
| HGB effect-oriented | **2.623°** | 8.1% | 22.3% |
| Higher-coverage ExtraTrees | 2.655° | 12.3% | 30.3% |
| Fixed 15% intervention budget | **2.678°** | **5.6%** | 15.4% |
| Fixed 20% intervention budget | **2.660°** | 7.1% | 20.1% |
| Oracle selector + consensus proposal | 2.035° | 0% | 91.0% |

The conservative point improves every test fold and keeps per-fold harm at or below
7.8%. It is the primary safety operating point; other rows form the coverage-risk
frontier.

The primary fixed-20% result is an absolute 0.142° and relative 5.1% reduction from
the 2.801° no-repair baseline. This modest mean effect should never be omitted or
described as a large clinical improvement.

The fixed-budget policies improve all five test folds. Their coverage standard
deviation is below one percentage point by construction, distinguishing stable score
ranking from the less stable absolute calibration threshold. The 20% operating point
has an identifier-clustered mean improvement of 0.138° (95% bootstrap CI
0.043–0.244°, two-sided paired sign-flip p=0.0046); the 15% point improves by 0.119°
(95% CI 0.029–0.218°, p=0.0112). A deployment batch or rolling ranking window is
required for fixed-budget operation.

Across three independent grouped-CV partition seeds, the fixed-20% policy reduces
fold-mean MAE by 0.142°, 0.238°, and 0.223°, with harm 7.1%, 6.1%, and 6.7%. It wins
14 of 15 held-out folds; the single losing fold changes by -0.047°. This is evidence
that the ranking result is not tied to one favorable patient partition.

### HVA/IMA feature ablation — same proposal, split, regressor, and 5% risk limit

| Selector features | Mean MAE | Harm | Coverage | Opportunity recall |
|---|---:|---:|---:|---:|
| Full | **2.660°** | 5.8% | 17.5% | 17.7% |
| No ensemble uncertainty | 2.746° | 4.0% | 10.6% | 11.0% |
| No analytic sensitivity | 2.658° | 5.2% | 15.7% | 15.8% |
| Geometry only | 2.729° | 2.1% | 7.8% | 8.5% |
| Uncertainty + context only | 2.729° | 3.9% | 12.3% | 12.0% |

The ablation supports detector disagreement as an important selection signal and
shows that uncertainty and proposal geometry are complementary. It does **not** show
an independent benefit from the current finite-difference sensitivity scalar, so the
paper must not claim sensitivity as the dominant empirical contribution.
Sensitivity being the strongest one-variable heuristic is not contradictory: it has
useful marginal value in isolation, but no measurable conditional contribution after
uncertainty and proposal geometry enter the flexible expected-gain regressor.

### Matched selector comparison — identical proposal and 20% intervention budget

| Selector | Mean MAE | Harm | Opportunity recall |
|---|---:|---:|---:|
| Random (100 repeats/fold) | 2.818° | 9.4% | 20.2% |
| Largest uncertainty | 3.204° | 12.5% | 20.0% |
| Largest proposal displacement | 3.197° | 11.9% | 20.0% |
| Largest analytic sensitivity | 2.796° | 8.7% | 19.6% |
| **Learned expected gain** | **2.660°** | **7.1%** | **20.2%** |

All methods use the same ensemble-consensus candidates and edit exactly 20% of test
cases. Thus disagreement identifies candidate changes but is not itself a safe repair
criterion. At the identifier-cluster level, learned expected gain beats largest
uncertainty by 0.486° (95% CI 0.230–0.782°, p=0.00015), displacement by 0.479°
(0.224–0.770°, p=0.00035), and sensitivity by 0.122° (0.017–0.237°, p=0.0265).
The learned-over-sensitivity advantage repeats under two additional grouped split
seeds: 0.208° (p=0.0116) and 0.151° (p=0.0081).

### Soft/shrinkage attack

All shrinkage coefficients are either fixed in advance or selected on the held-out
calibration partition:

| Update | Mean MAE | Harm | Trigger rate |
|---|---:|---:|---:|
| No repair | 2.801° | 0% | 0% |
| Geometry shrinkage, α=0.1 | 2.750° | 36.8% | 100% |
| Geometry shrinkage, α=0.3 | 2.699° | 38.5% | 100% |
| Output averaging, α=0.3 | 2.627° | 37.4% | 100% |
| Calibration-tuned output averaging | **2.598°** | 38.5% | 100% |
| Shrinkage calibrated under ≤10% harm | 2.801° | 0% | 0% |
| Selective expected gain, 20% budget | 2.660° | **7.1%** | 20% |

The zero-threshold harm column is structurally unfavorable to continuous updates:
even an arbitrarily small negative change counts as harm. It is retained only as a
diagnostic and must not support the safety claim. Magnitude-aware results overturn the
earlier interpretation.

| Method | Harm@0.5° | Harm@1° | Harm@2° | E[worsening \| worse] | P95 worsening \| worse |
|---|---:|---:|---:|---:|---:|
| Learned binary gate | 2.8% | 0.6% | 0.6% | 0.402° | 0.949° |
| Geometry shrink α=0.1 | 0.6% | 0.6% | 0% | 0.054° | 0.354° |
| Geometry shrink α=0.3 | 2.3% | 2.3% | 0.6% | 0.205° | 1.730° |
| Output average α=0.3 | **1.1%** | **0%** | **0%** | 0.048° | 0.402° |

Consequently, binary selective correction does **not** dominate clinically meaningful
harm thresholds. Output averaging is a stronger competitor, while executable
geometry shrinkage occupies different portions of the frontier.

### Unified Pareto frontier and gated shrinkage

![Magnitude-aware Pareto frontier](../outputs/research/pareto_frontier.png)

The deployment objective is now stated explicitly as
\(\min \mathrm{MAE}\;\text{s.t.}\;\Pr(E_{after}-E_{before}>\tau)\leq\epsilon\).
Sweeping gate coverage and update magnitude shows that learned gate + shrinkage
dominates the original binary 20% policy over substantial low-harm regions. On the
descriptive test frontier, for example, learned executable geometry reaches 2.535° at
harm@1°=0.95% using 80% coverage and α=0.5; the corresponding output interpolation is
2.519°. These are frontier diagnostics, not calibration-selected headline numbers.

When coverage and α are selected only on a calibration fold, empirical risk does not
transfer exactly: a calibration target harm@0.5°≤1% yields 2.699° test MAE and 2.36%
test harm for learned geometry. Small-sample empirical calibration therefore cannot
be presented as a risk guarantee; an upper-confidence or conformal risk-control
procedure is required for such a claim.

### Exact advantage over joint component–magnitude actions

The action space is expanded from three full component replacements to STOP plus
18 executable actions: three components × α∈{0.1,0.2,0.3,0.5,0.7,1.0}. Every action
receives an exact counterfactual advantage label from the executor.

| Policy | Mean MAE | P95 | Coverage | Harm@0.5° | Harm@1° | Harm@2° |
|---|---:|---:|---:|---:|---:|---:|
| No repair | 2.801° | 10.177° | 0% | 0% | 0% | 0% |
| Full-edit-only regression | 2.569° | 8.862° | 90.9% | 11.2% | 6.3% | 2.7% |
| Adaptive-α regression | **2.499°** | **7.795°** | 95.1% | 10.8% | 5.7% | 2.6% |
| Risk-targeted adaptive (τ=1°, ε=1%) | 2.661° | 9.252° | 36.4% | 0.6% | 0.6% | 0.4% |
| Oracle full edit | 2.030° | 7.202° | 90.0% | 0% | 0% | 0% |
| Oracle adaptive α | **1.864°** | **5.565°** | 95.9% | 0% | 0% | 0% |

The expanded oracle improves by 0.166° over the binary oracle, but the binary action
set is a strict subset of the adaptive set, so this inequality is guaranteed and is
not evidence by itself. A block-permutation max-over-actions null produces a much
larger nominal gap (1.066°); because it breaks the strong correlation among α actions,
it is an upper stress test rather than an unbiased null correction, but it invalidates
any claim based on the raw oracle gap alone. More importantly, selecting an action on
two detector seeds and evaluating it on the held-out third seed yields adaptive minus
binary gain −0.073° (95% CI −0.161–0.006°). Thus case-specific magnitude headroom does
not replicate across detector noise in the current data.

The global-α oracle selects α=1, so a single alternative global geometry step does not
explain the nested-oracle gap. Nevertheless, the learned adaptive regressor improves
over its full-edit restriction by only 0.061°, with clustered CI −0.014–0.147°
(p=0.143); this learned difference is inconclusive.
The risk-targeted adaptive policy improves over no repair by 0.137° (95% CI
0.064–0.224°), while empirical calibration still must not be interpreted as a formal
risk guarantee. Direct benefit classification reduces large-harm frequency but loses
mean improvement, exposing the expected asymmetric-loss trade-off rather than solving
it automatically.

### Reconciled protocol and policy behavior

All headline tail values are now pooled across held-out predictions, averaged within
the 176 identifiers, and then summarized once. Under this single protocol:

| Policy | Mean | P90 | P95 | Harm@0.5° | Harm@1° | Harm@2° |
|---|---:|---:|---:|---:|---:|---:|
| No repair | 2.678° | 5.297° | 8.055° | 0% | 0% | 0% |
| Fixed output α=0.3 | **2.510°** | 5.061° | 7.912° | 1.1% | **0%** | **0%** |
| Learned binary 20% | 2.540° | 4.947° | **7.576°** | 2.8% | 0.6% | 0.6% |
| Risk-targeted adaptive | 2.541° | **4.889°** | 7.901° | **0%** | **0%** | **0%** |

Fixed α=0.3 beats risk-targeted adaptive by 0.031° in mean error, but the paired
cluster bootstrap CI is −0.073–0.135°, so neither is demonstrably superior at n=176.
Their P95 bootstrap intervals are also very wide and strongly overlapping. The honest
conclusion is that no learned deployable policy has surpassed the tuned constant at
this sample size.

Raw adaptive regression is not an abstaining system: STOP occurs in only 5.3% of
outputs and α=1 accounts for 76.2% of edits. The risk-targeted variant does abstain
(65.5% STOP) and uses a broader α distribution, but its 2.541° clustered MAE is
essentially identical to the original binary policy's 2.540°. These behavioral labels
must replace the earlier generic “risk-aware abstention” description wherever the raw
regressor is discussed.

### Learn-then-Test feasibility audit

With 20 predeclared policies and familywise δ=0.05, the five calibration folds contain
only 20–33 independent identifiers. Even with zero observed harms, a Bonferroni-valid
binomial test can certify only 16.6%–25.9% risk. Certification at 10%, 5%, 2%, and 1%
requires at least 57, 117, 297, and 597 independent calibration identifiers,
respectively. Formal low-risk control is therefore underpowered in this dataset and
must not be sold as a solved contribution. This power result is a direct motivation
for a second, larger executable domain.

## Second domain: executable Text-to-SQL edits

The official Spider 1.0 development questions, SQLite databases, and repository
example predictions define a second domain with an exact observed-database reward:
whether a candidate SQL edit returns the same result as the gold query. Generic edit
operators modify literals grounded from the question, aggregations, columns,
comparators, Boolean connectives, ordering, DISTINCT, and LIMIT. Candidate generation
does not inspect the gold SQL or gold execution label. Databases, rather than queries,
are assigned to five folds, so every test fold contains unseen schemas.

This separation is now enforced by a reproducible gold-leakage audit. All 28,107
stored candidates were regenerated using only predicted SQL, question text, public
schema column names, and the candidate limit. Their ordered action/SQL corpus exactly
matched the stored corpus (SHA-256
`ba6f4a65f027081562d15bc75f5dbbbd4e9aad5ac2d7826a1cab9f78f983c3c4`). Removing or
deliberately mutating gold SQL, base correctness, candidate correctness, and advantage
labels leaves every inference feature unchanged. The audit also verifies the label
identity `advantage = candidate_correct - base_correct` and pairwise-disjoint
train/calibration/test databases in every fold. Gold execution is therefore used only
after candidate generation to construct supervised counterfactual labels and to score
held-out outcomes; it is not an input to candidate generation or selector inference.

| Policy | Execution accuracy | Coverage | Benefit | Harm | Net |
|---|---:|---:|---:|---:|---:|
| No repair | 33.17% | 0% | 0% | 0% | 0% |
| First executable edit | 32.69% | 97.49% | 16.54% | 17.02% | −0.48% |
| Learned exact-advantage selector | **44.68%** | 43.52% | **13.73%** | **2.22%** | **+11.51%** |
| Candidate oracle | 52.51% | 19.34% | 19.34% | 0% | 19.34% |

The learned selector improves all five held-out database folds. It repairs 142 cases
and harms 23, for an absolute 11.51-point accuracy gain over no repair. The
database-clustered bootstrap CI is 5.37–17.45 points; the case-level paired exact test
is p=3.96e−22. Within this exploratory development cycle, the result shows that exact
counterfactual edit advantage can be fitted in a second executable domain; it is not
independent confirmatory evidence.

This is not yet a standard Spider leaderboard result. The prediction file is the
official repository example rather than a modern named model, the candidate set is
hand-designed and recovers only 19.34% of cases, and reward is equality on the
observed SQLite instance rather than official multi-instance test-suite accuracy.
Those constraints must remain explicit. This development-set result is exploratory:
although model fitting, policy calibration, and outer-fold testing are isolated by
database within the recorded run, the same Spider development set was used during
initial method development. It is therefore not an untouched confirmatory test. The
frozen follow-up protocol is specified in `docs/SPIDER_EVALUATION_PROTOCOL.md`.

All current candidate families, their ordering, and the 40-candidate limit were
developed during the Spider dev cycle. They must therefore be described as
dev-developed. The sealed database split can confirm transfer of the complete frozen
pipeline, but it cannot erase method-level candidate-generator selection bias or
separate dev specialization from intrinsically hard residual errors. No credible
pre-dev operator implementation exists for a retrospective provenance comparison.

The preregistered success rule is a net gain of at least 3.00 percentage points with
a database-clustered 95% bootstrap lower confidence bound above zero. Confirmatory
gold is inspected once, only after base outputs, candidates, selector policy, and
analysis hashes are frozen. The experiment stops after that single look regardless of
the result; subsequent modifications are exploratory and require new test data.

### Strong-base stress test

We additionally evaluated the frozen edit operators on the four official Spider dev
prediction files committed with CodeS (repository commit
`11a9f1ceb292d6e9887990e7b108204d94c82cb0`). This is a cross-model diagnostic on the
previously used development set, not a new confirmatory test.

| Base | Base EX | Candidate-oracle headroom | Learned net gain | Benefit / harm |
|---|---:|---:|---:|---:|
| Repository weak example | 33.17% | +19.34 pt | +11.51 pt | 142 / 23 |
| CodeS-1B | 72.63% | +4.64 pt | −0.48 pt | 0 / 5 |
| CodeS-3B | 76.11% | +5.51 pt | −0.29 pt | 0 / 3 |
| CodeS-7B | 78.14% | +5.51 pt | −0.68 pt | 2 / 9 |
| CodeS-15B | 79.01% | +4.64 pt | −0.48 pt | 0 / 5 |

The weak-base positive result does not survive stronger bases. Importantly, the edit
space does not become empty: it retains 48–57 exactly beneficial cases, or 4.64–5.51
accuracy points. Every oracle benefit on all four CodeS bases starts from an executable
but semantically incorrect SQL query; none is merely invalid-to-valid syntax repair.
The immediate failure is therefore selection under sparse, model-dependent positive
advantage rather than complete loss of candidate coverage.

A leave-one-base-model-out selector trained on the other four prediction sources also
fails: net changes are −0.39, −0.68, −0.68, and −0.97 points for CodeS 1B, 3B, 7B,
and 15B. Merely pooling more counterfactual labels from other base models does not
solve transfer. The defensible conclusion is now conditional: exact rewards expose a
nontrivial repair ceiling, but the learned advantage policy is not robust to stronger
base-model error distributions.

### Clean retraining versus pretrained-model transfer

To separate transfer from retraining, we initialized five independent models from the
original `t5-base` checkpoint and fine-tuned each for three fixed epochs using only
four fifths of the 116 development databases. Each model generated SQL exclusively
for its held-out database fold. The resulting 6,602 predictions are therefore
database-level out-of-fold predictions; the 30 sealed databases and their gold labels
were excluded by construction. Training loss was finite in every fold and each source
example appears exactly once in the merged output.

The clean base obtains 26.40% observed-instance execution accuracy. Its lower score is
not presented as a modern competitive baseline; its purpose is a data-provenance-clean
retraining control. Despite its low accuracy, the frozen edit space has only 5.83
points of candidate-oracle headroom, demonstrating that base accuracy alone does not
determine repairability.

**Post-run pipeline audit.** This control is too weak to answer the intended
strong-base question and the table below must be treated as a pipeline diagnostic,
not evidence about a modern strong parser. Among all 4,859 base failures, 2,731
(56.2%) reference a nonexistent column, 468 (9.6%) have SQL syntax errors, 132 (2.7%)
reference a nonexistent table, and 1,453 (29.9%) execute but return the wrong result.
A fixed-seed manual review of 50 failures gives 26 nonexistent-column, 4 syntax, 3
nonexistent-table, and 17 executable-but-wrong cases. There are no markdown, empty
output, explanatory-prefix, or multi-statement parsing failures in that sample or the
full heuristic audit. Only 8.9% of prompts exceed the 768-token input limit, so
truncation contributes but cannot explain the dominant schema-grounding failure.

Accordingly, no proposal/selector conclusion in this subsection is promoted to a
strong-base claim until a repaired schema-grounding pipeline reaches a prespecified
modern-baseline range and the entire quadruple is rerun. CodeS and BIRD follow-ups are
paused to prevent accumulating additional negative results from the same upstream
defect.

| Quadruple component | EX | Absolute gain | Benefit / harm | Clustered 95% CI |
|---|---:|---:|---:|---:|
| No repair | 26.40% | 0 | 0 / 0 | – |
| Learned proposal applied to all | 17.71% | −8.69 pt | 119 / 693 | [−11.24, −6.32] pt |
| Learned selector + learned proposal | 26.46% | +0.061 pt | 8 / 4 | [−0.047, 0.181] pt |
| Oracle selector + learned proposal | 28.20% | +1.80 pt | 119 / 0 | [1.37, 2.22] pt |
| Candidate oracle | 32.23% | +5.83 pt | 385 / 0 | [4.54, 7.10] pt |

As a diagnostic of this failed pipeline, the learned proposal recovers 30.9% of
candidate-oracle headroom. The learned selector recovers only 3.36% of the learned
proposal's oracle-gated headroom, and end-to-end recovery is 1.04%. The primary
preregistered Go condition fails: +0.061 points is below the 3-point minimum and its
database-clustered confidence interval crosses zero. The sealed outcome therefore
remains locked.

### Sealed-set power audit without outcomes

The 30 sealed databases contain 2,057 questions, counted using database identifiers
only. Power cannot be estimated from the observed near-always-STOP policy because its
near-zero variance would make the design look artificially certain. Using the more
heterogeneous candidate-oracle between-database SD of 5.02 points as a planning proxy,
30 databases provide approximately 90.6% normal-approximation power for a 3-point
effect at two-sided α=0.05; the corresponding estimate is 22 databases. This result is
assumption-sensitive: power is 59.1% at a 7.5-point database SD and 37.6% at a
10-point SD. We therefore call the current sealed size conditionally adequate, not
universally powered. This does not affect the present decision because the selector
already fails the development Go criterion.

### Asymmetric and tail outcomes (identifier-clustered)

| Selector | P90 | P95 | Worst | Benefit | Harm | Benefit−harm |
|---|---:|---:|---:|---:|---:|---:|
| No repair | 5.297° | 8.055° | 30.708° | 0% | 0% | 0% |
| Largest sensitivity | 5.543° | 8.320° | 30.708° | 14.2% | 12.5% | 1.7% |
| Learned expected gain | **4.947°** | **7.576°** | 30.708° | **25.0%** | **10.2%** | **14.8%** |

The learned policy improves P90 by 0.350° and P95 by 0.479° relative to no repair,
but does not improve the worst case. Its benefit is therefore broader than the mean
alone suggests, while catastrophic-tail elimination remains unsupported.

### Ensemble size and measured CPU cost

| Detectors | Pairing | Mean MAE | Harm | Oracle MAE | Parameters | Forward latency |
|---:|---|---:|---:|---:|---:|---:|
| 1 | – | 2.801° | 0% | 2.801° | 7.76M | 79 ms |
| 2 | companion A | **2.651°** | 7.4% | 2.107° | 15.53M | 157 ms |
| 2 | companion B | 2.782° | 8.2% | 2.069° | 15.53M | 157 ms |
| 3 | all | **2.660°** | **7.1%** | **2.035°** | 23.29M | 243 ms |

Latency is sequential batch-one forward time at 256×256 on the available six-thread
CPU and excludes weighted-PCA post-processing. K=1 has no independent proposal and
therefore reduces to STOP; its nominal top-20% selections are geometric no-ops.
Two-detector performance depends on which independent detector is paired with the
base. Three detectors give the best oracle ceiling and the most consistent five-fold
result, while the favorable two-detector pairing exposes a useful lower-cost operating
point rather than a universally equivalent replacement.

### Unseen detector-pairing and severity robustness

For each of the three unordered detector pairs, the selector is trained only on the
other two pairings and tested on both unseen patients and the held-out pairing. Across
15 pairing×fold cells:

| Method | Mean MAE | Harm | Coverage | Improved cells |
|---|---:|---:|---:|---:|
| No repair | 2.801° | 0% | 0% | – |
| Expected gain on unseen pairing | **2.716°** | **6.8%** | 19.9% | 11/15 |

Identifier-clustered improvement is 0.089° (95% CI 0.032–0.152°, p=0.0024).
All three held-out pairings improve on average, but pairing-specific uncertainty is
heterogeneous: 17–73 is significant (p=0.0045), 42–73 is borderline (p=0.0567), and
17–42 is not significant alone (p=0.184). This supports **partial cross-pairing
generalization**, not detector invariance.

Severity thresholds are determined from each training partition and then frozen:

| Baseline severity | Before MAE | After MAE | Change | Coverage | Within-bin score–gain ρ |
|---|---:|---:|---:|---:|---:|
| Low | 0.691° | 0.731° | **−0.040°** | 8.8% | −0.208 |
| Medium | 1.561° | 1.557° | +0.004° | 14.9% | 0.196 |
| High | 6.013° | **5.696°** | **+0.317°** | 37.2% | 0.349 |

The transferable benefit is concentrated in high-error cases, where score ranking
remains meaningful within the severity stratum. Low-error cross-pairing cases expose a
remaining failure mode: even sparse edits can be net harmful. This result strengthens
the mechanism claim but also motivates a conservative adequacy/STOP gate; it must not
be hidden by reporting only the global average.

### Cobb — cross-protocol validation

Continuous endplate-axis repair, image-grouped five-fold CV:

| Method | Cobb MAE | Harm | Coverage |
|---|---:|---:|---:|
| No repair | 12.207° | 0% | 0% |
| Selective consensus-axis repair | **11.501°** | **6.4%** | 21.9% |
| Oracle selector | 7.956° | 0% | 69.9% |

Discrete candidate-pair repair (`RESELECT_STRUCTURE`):

| Method | Cobb MAE | Harm | Effective coverage |
|---|---:|---:|---:|
| No repair | 4.099° | 0% | 0% |
| Selective structure repair | **3.961°** | **4.5%** | 12.0% |
| Oracle selector | 3.805° | 0% | 8.5% |

Effective coverage counts only cases whose candidate pair actually changes; accepted
no-ops are excluded.

## Synthetic phase diagram: when selectivity is learnable

A cluster-disjoint synthetic study independently varies proposal precision,
advantage-label noise, and the number of independent clusters. Across 4,800
runs, learned selection does not reliably beat the calibration-tuned global
apply-all/no-op action with 10 clusters and 40% label noise. The reliable-win
region expands substantially at 100 clusters and covers the full evaluated
grid at 300 clusters. The largest advantage appears at intermediate proposal
precision, where neither global action is already near-optimal. This supports a
conditional claim: exact counterfactual labels are not sufficient; selective
correction additionally requires actionable proposal heterogeneity, a usable
advantage signal, and adequate effective sample size. Full protocol and
limitations are in `docs/SYNTHETIC_SELECTIVE_CORRECTION_PHASE.md`.

## Schema-grounded Spider base audit

On development fold 0 only, replacing the compact schema list with `CREATE
TABLE` DDL, primary/foreign keys, three sample rows, and question-conditioned
schema selection increased execution accuracy from 23.10% to 29.07% and SQL
executability from 46.68% to 55.28%. Among the remaining 1,047 failures, 49.95%
still contain nonexistent columns and 37.15% execute but return the wrong
result. Because accuracy remains below the preregistered 40% capacity gate, the
remaining T5-base folds were not run and no candidate or selector metric was
inspected. These numbers diagnose base-model capacity; they are not a method
result.

## Reproduction

From the project root:

```bash
python scripts/ensemble_gain_model_cv.py --risk-limit .05 \
  --output outputs/research/ensemble_gain_model_cv_risk05.json
python scripts/ensemble_gain_model_cv.py --risk-limit .10 \
  --output outputs/research/ensemble_gain_model_cv.json
python scripts/ensemble_gain_model_cv.py --risk-limit .15 \
  --output outputs/research/ensemble_gain_model_cv_risk15.json
python scripts/ensemble_gain_model_cv.py --risk-limit .20 \
  --output outputs/research/ensemble_gain_model_cv_risk20.json
python scripts/ensemble_gain_model_cv.py --target-coverage .15 \
  --output outputs/research/fixed_coverage_15.json
python scripts/ensemble_gain_model_cv.py --target-coverage .20 \
  --output outputs/research/fixed_coverage_20.json
python scripts/ensemble_gain_model_cv.py --seed 2028 --target-coverage .20 \
  --output outputs/research/fixed_coverage_20_seed2028.json
python scripts/ensemble_gain_model_cv.py --seed 2029 --target-coverage .20 \
  --output outputs/research/fixed_coverage_20_seed2029.json
# Repeat with --feature-mode no_uncertainty, no_sensitivity, geometry_only,
# and uncertainty_only for the matched feature ablation.
python scripts/cobb_ensemble_gain_cv.py
python scripts/cobb_discrete_structure_repair_cv.py
python scripts/paired_cluster_statistics.py
python scripts/paired_cluster_statistics.py --input outputs/research/fixed_coverage_20.json \
  --output outputs/research/paired_cluster_statistics_fixed20.json
python scripts/selector_baselines_cv.py
python scripts/paired_selector_statistics.py
python scripts/ensemble_gain_model_cv.py --ensemble-size 1 --target-coverage .20 \
  --output outputs/research/ensemble_size_1_fixed20.json
python scripts/ensemble_gain_model_cv.py --ensemble-size 2 --target-coverage .20 \
  --output outputs/research/ensemble_size_2_fixed20.json
python scripts/ensemble_gain_model_cv.py --ensemble-size 2 --ensemble-companion-offset 2 \
  --target-coverage .20 --output outputs/research/ensemble_size_2_offset2_fixed20.json
python scripts/ensemble_gain_model_cv.py --ensemble-size 3 --target-coverage .20 \
  --output outputs/research/ensemble_size_3_fixed20.json
python scripts/benchmark_ensemble_cost.py --repeats 10
python scripts/pairing_generalization_cv.py
python scripts/paired_pairing_generalization_statistics.py
python scripts/shrinkage_baseline_cv.py
python scripts/selector_tail_metrics.py
python scripts/pareto_frontier_cv.py
MPLCONFIGDIR=/tmp/geomed-mpl python scripts/plot_pareto_frontier.py
python scripts/calibrated_pareto_policies_cv.py
python scripts/adaptive_action_advantage_cv.py
python scripts/paired_adaptive_action_statistics.py
python scripts/adaptive_oracle_diagnostics.py
python scripts/reconciled_policy_metrics.py
python scripts/ltt_risk_control_feasibility.py
python scripts/build_spider_executable_edits.py
python scripts/spider_advantage_selector_cv.py
python scripts/spider_selector_statistics.py
python scripts/summarize_selective_correction_results.py
python scripts/synthetic_selective_correction_phase.py
pytest -q
```

## Limitations that must remain explicit

- HVAngleEst is patient-grouped. AASCE provides no patient identifier, so its CV is
  image-grouped and must not be called patient-disjoint.
- The continuous-axis and discrete-structure Cobb experiments use different upstream
  models and should not be presented as one end-to-end pipeline.
- The current selector relies on three independently trained detectors. Compute cost
  and the single-model alternative require an ablation.
- Overall harm is low, but conditional harm among selected edits remains nontrivial.
- Fixed-budget policies require a deployment batch or rolling score-ranking window.
- External institutional validation remains necessary for clinical generalization;
  AASCE is a cross-protocol benchmark, not an institutional HVA/IMA replication.
- Spider reward is equality on one observed SQLite database instance, not official
  multi-instance test-suite accuracy; execution-equivalent false positives can occur.
- Spider uses an official repository example prediction file with unclear model
  provenance and must not be reported as a modern leaderboard baseline.
