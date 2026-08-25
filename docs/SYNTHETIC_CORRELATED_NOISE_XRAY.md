# Correlated detector-noise extension around the X-ray point

## Purpose

The frozen independent-flip phase model cannot represent the robust X-ray
pairwise disagreement of 0.622: for two independent equal-error detectors,
`D = 2q(1-q) <= 0.5`. This independent extension asks whether an explicit
paired-detector error model can represent that observation and whether the
resulting X-ray neighborhood supports learned selective correction.

It does not modify or overwrite any frozen phase artifact. It uses no Spider
data, sealed or otherwise, and trains no real model.

## Correlated error model

Let `Y` be the latent true advantage sign and let detector labels be
`Y_i = Y xor E_i`. Both error indicators have marginal error probability `q`,
and their Pearson correlation is `rho`. Their joint probability is

`P(E1=1,E2=1) = q^2 + rho q(1-q)`.

Consequently,

`D = P(Y1 != Y2) = P(E1 != E2) = 2q(1-q)(1-rho)`.

Positive correlation describes shared, same-direction failures and *reduces*
disagreement; it cannot explain `D>0.5`. Negative correlation describes
case-dependent opposing errors---one detector is preferentially wrong on cases
where the other is right---and permits `D>0.5`, subject to the Frechet bounds
of the Bernoulli joint distribution. For example, `q=0.35, rho=-0.35` gives
`D=0.614`, close to the robust X-ray estimate.

Detector 1 supplies the noisy training advantage label. Detector 2 is used
only to estimate disagreement. This distinction is important: at fixed `q`,
`rho` changes the relation between observable pairwise disagreement and latent
label error, but does not by itself change the marginal noise seen by a learner
trained only on detector 1. Thus the extension repairs the proxy mapping; it
does not claim that correlation is an additional causal learnability axis in
this single-label training design.

## Design

- 176 independent clusters, one example per cluster, matching the X-ray
  effective cluster structure;
- proposal precision 0.40;
- `q` in `{0.35, 0.375, 0.40, 0.425, 0.45, 0.475, 0.50}`;
- `rho` in `{-0.50, -0.35, -0.20, 0, 0.20}`;
- 30 repeats for every feasible cell: 35 conditions and 1,050 runs;
- cluster-disjoint 60/20/20 train/calibration/test splits;
- learned logistic selector compared against the better calibration-tuned
  constant action (apply-all or no-op).

## Results

The correlated model successfully spans both observed X-ray disagreements.
Near the primary value `D=0.476`, the closest cell (`q=0.40, rho=0`, `D=0.480`)
has learned-minus-constant gain `+0.10` points on average, a repeat interval of
`[-15.86, +14.57]` points, and wins in only 33.3% of repeats. Near the robust
value `D=0.622`, the closest cell (`q=0.35, rho=-0.35`, `D=0.614`) has `+0.67`
points, interval `[-18.71, +18.71]`, and wins in 40.0% of repeats. Neither is a
reliable selective-win region.

The conclusion is therefore narrower but more defensible than treating raw
disagreement as independent label-flip probability:

> A feasible negatively correlated detector-error model explains disagreement
> above 0.5, and the X-ray operating neighborhood remains statistically
> unreliable for learned selective correction at 176 one-sample clusters.

This upgrades the simulator's support for the observed disagreement but does
not identify `q` from disagreement alone: many `(q,rho)` pairs produce the same
`D`. Repeated labels or an independently estimated detector error correlation
would be required for point identification.

## Artifacts

- `scripts/synthetic_correlated_noise_extension.py`
- `outputs/research/synthetic_correlated_noise_xray.json`
- `outputs/research/synthetic_correlated_noise_xray_raw.csv`
- `outputs/research/synthetic_correlated_noise_xray_summary.csv`
- `outputs/research/synthetic_correlated_noise_xray.png`
