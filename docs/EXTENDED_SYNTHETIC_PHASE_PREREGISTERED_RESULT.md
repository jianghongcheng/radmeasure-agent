# Preregistered extended phase result

## Protocol integrity

The extension was frozen before it was run in
`protocols/SYNTHETIC_PHASE_EXTENSION_PREREGISTRATION_V1.json` (SHA-256
`b2a40c418b4e04e1bb2118134eec6591f624977f41324d66a625be6044b01065`).
The simulator's data-generating process and learner were unchanged. Only the
registered precision, noise, cluster-count grid and output prefix were added.

The 630-condition, 25,200-run simulation was completed before real outcomes
were joined. Predictions were written to a file containing no observed gains
and frozen with SHA-256
`8e8f3b43c9cfb4f99c66cb1443ad8c0e04927a7f945ac6e012e4711f89bafe84`.
Only after that hash was recorded was the outcome comparison executed.

This controls new analysis freedom but is not perfect temporal blinding: the
historical outcomes were already known before preregistration. The protocol
must be described as a sealed analysis extension, not a prospective external
validation.

## Frozen predictions and reveal

The registered definition calls selectivity valuable only when mean gain beats
both no-op and apply-all and both repeat-level win probabilities are at least
80%.

| Domain | Frozen phase prediction | Observed result | Direction match |
|---|---|---|---|
| X-ray | No reliable selective value | -0.040° MAE reduction | Yes, but not noise-mapping robust |
| Spider weak | Reliable selective value | +11.51 execution-accuracy points | Yes |
| Spider clean | No reliable selective value | +0.061 points; clustered CI crosses zero | Yes |

Numerically, Spider weak was predicted to gain 13.93 points with 99.64% and
97.68% probabilities of beating no-op and apply-all. Spider clean was predicted
to have essentially zero gain (0.0058 points) and only a 51.64% probability of
beating no-op. X-ray at the direct disagreement coordinate was predicted to
have only a 45.05% probability of beating no-op, so it failed the frozen
reliability criterion despite a positive mean.

## X-ray noise is a proxy, not the simulator parameter

The real coordinate is pairwise cross-detector advantage-sign disagreement:

\[
D = \Pr[\operatorname{sign}(\Delta_i) \ne \operatorname{sign}(\Delta_j)] = 0.47598.
\]

The simulator parameter is the probability `q` that a supervision label is
flipped relative to latent truth. Under the additional assumptions of two
independent, symmetric label errors,

\[
D = 2q(1-q), \qquad
q = \frac{1-\sqrt{1-2D}}{2} = 0.39042.
\]

Those assumptions are not established for detector proposals. Moreover, at the
preregistered `noise=0.4` sensitivity point the simulator predicts reliable
selective value, whereas at direct `noise=0.476`, `0.6`, and `0.622` it predicts
no reliable value. Therefore the X-ray directional match is sensitive to the
mapping and cannot be claimed as robust explanatory validation. The 0.5°
threshold disagreement of 62.20% is also above the 50% maximum of the simple
independent symmetric-flip model, directly demonstrating that this model is
insufficient for the robust proxy.

## Controlled Spider base shift

With the same dataset and dev-developed operator family, proposal precision
falls from 49.28% on Spider weak to 14.66% on Spider clean, a **70.26% relative
drop**. The observed learned gain simultaneously falls from +11.51 points to
+0.061 points with a confidence interval crossing zero. The frozen expanded
phase predicts the same qualitative transition: reliable selective value near
49% precision and no reliable advantage over no-op near 15% precision.

This is the cleanest real-domain support for the phase hypothesis. It is a
controlled association under a base-model shift, not a randomized causal
effect: changing the base also changes its error distribution.

## Verdict

The extension produces three primary directional matches, but only the two
Spider points share an exact deterministic-noise coordinate with the simulator.
The strongest defensible claim is therefore:

> The expanded synthetic phase correctly predicts the collapse of selective
> correction under a controlled Spider base-model shift as proposal precision
> falls by 70%, while the X-ray comparison remains qualitatively consistent but
> is not robust to the mapping between detector disagreement and label-flip
> noise.

This is useful evidence for the paper, but it is not yet proof that a
three-variable simulator fully explains all real domains. Cluster-internal
sample size and a generative model for correlated detector noise remain missing
variables.

## Artifacts

- Extended Figure 1: `outputs/research/figure1_extended_real_domains_on_synthetic_phase.png`
- Extended Figure 2: `outputs/research/figure2_extended_selective_strategy_boundaries.png`
- Frozen predictions: `outputs/research/synthetic_phase_extended_v1_predictions_only.json`
- Revealed comparison: `outputs/research/synthetic_phase_extended_v1_revealed_comparison.json`
- Full simulation: `outputs/research/synthetic_selective_phase_extended_v1.json`
- Raw 25,200 runs: `outputs/research/synthetic_selective_phase_extended_v1_raw.csv`
