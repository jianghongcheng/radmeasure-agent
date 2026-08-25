# Synthetic selective-correction phase diagram

## Question

When does a learned selective policy outperform both no correction and a
calibration-tuned global action?  This synthetic experiment independently
controls three quantities implicated by the medical and Text-to-SQL results:

- proposal precision: the fraction of proposed edits with positive true advantage;
- advantage-label noise: the probability that a training advantage sign is flipped;
- effective sample size: the number of independent clusters.

The experiment is explanatory rather than a claim about a particular real-world
data-generating process.

## Protocol

Each proposal has true counterfactual advantage `+1` (beneficial) or `-1`
(harmful). A stable observable signal predicts the sign, while cluster effects
and nuisance features induce dependence and distribution variation. Clusters,
not examples, are split 60/20/20 into train/calibration/test sets.

The learned policy is logistic advantage-sign estimation trained with the
specified label-flip probability. Its intervention threshold is selected only
on calibration clusters. The tuned constant uses the same calibration set to
choose between applying every proposal and applying none. All reported gains
are evaluated on unseen test clusters.

The grid contains 120 conditions: proposal precision 0.30--0.80, label noise
0.00--0.40, and 10/30/100/300 clusters. Each condition has 40 independent
repeats and 40 examples per cluster (4,800 runs total). A reliable win is
defined descriptively as positive mean test gain over the best tuned constant
and wins in at least 80% of repeats. This is an empirical boundary, not a
formal guarantee.

## Results

The phase diagram supports three conclusions.

1. Effective cluster count controls reliability. With 10 clusters, no reliable
   win is observed at 40% label noise. With 30 clusters, reliable wins remain
   only in the middle-precision region. With 100 clusters, almost the entire
   0.30--0.70 range is reliable even at 40% label noise. With 300 clusters, all
   evaluated cells meet the descriptive reliability rule.
2. The largest benefit occurs near 50% proposal precision. At 100 clusters and
   zero label noise, learned-minus-constant gain is +16.0 points at precision
   0.50, versus +3.7 at 0.30 and +0.6 at 0.80. This is expected: a global
   apply-all/no-op action is already strong at extreme precision, leaving less
   selective headroom.
3. Noise and cluster count interact. At 10 clusters and 40% noise, the learned
   policy is not reliably better anywhere. At 30 clusters under the same noise,
   precision 0.50 yields +9.2 points and wins 90% of repeats. At 100 clusters,
   that cell yields +13.3 points and wins every repeat.

The synthetic result therefore does not support “exact labels are sufficient.”
It predicts that selective correction succeeds when there is actionable
heterogeneity (neither almost-all-good nor almost-all-bad), an observable signal
for edit advantage, and enough independent clusters relative to label noise.
Proposal precision alone is not monotonic evidence for learnability.

## Reproduction and artifacts

Run:

```bash
python scripts/synthetic_selective_correction_phase.py
```

Artifacts:

- `outputs/research/synthetic_selective_phase.json`: design and aggregated results;
- `outputs/research/synthetic_selective_phase_raw.csv`: all 4,800 runs;
- `outputs/research/synthetic_selective_phase_summary.csv`: condition summaries;
- `outputs/research/synthetic_selective_phase_boundaries.csv`: empirical reliable-win regions;
- `outputs/research/synthetic_selective_phase.png`: four-panel phase diagram.

Limitations: the reward magnitude is symmetric, the feature signal strength is
fixed, and label noise is independent sign flipping. Real proposal mechanisms
may have asymmetric harm, covariate-dependent noise, and base-model-specific
distribution shifts. The phase diagram isolates the requested variables; it
does not replace validation in those richer settings.
