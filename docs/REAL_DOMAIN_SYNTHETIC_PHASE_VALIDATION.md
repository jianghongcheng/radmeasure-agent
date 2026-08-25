# Can the synthetic phase diagram explain the real domains?

## Result

Not yet as a three-domain validation. The one real coordinate inside the frozen
synthetic grid, Spider weak, is predicted correctly. X-ray and Spider clean lie
outside the grid, so assigning them a phase prediction would be extrapolation.
This is a useful falsifiability result: the current synthetic study has the
right qualitative axis for Spider weak, but its support is too narrow to claim
that it explains all three real outcomes.

## Coordinate definitions

- **Clusters:** total independent clusters available to cross-validation. X-ray
  contains 176 patients; its five train/validation/test patient splits are
  113/28/35, 118/30/28, 111/35/30, 93/48/35, and 93/35/48. Spider weak has 20
  databases and Spider clean has 116 development databases.
- **Proposal precision:** beneficial / (beneficial + harmful) for the concrete
  proposal used by the apply-all policy. The robustness value is beneficial /
  all cases and therefore treats zero-advantage proposals as non-beneficial.
- **Advantage noise:** pairwise disagreement of the realized advantage sign
  across detector predictions for the same X-ray patient. Spider execution
  rewards are deterministic, so their label-noise coordinate is zero.
- **Mapping:** trilinear interpolation inside the synthetic grid and log-linear
  interpolation over cluster count. Coordinates outside the grid are not given
  a prediction. A clipped-grid value is retained in JSON only as sensitivity
  analysis and must not be presented as a validated prediction.

## Table 1 — real-domain phase coordinates

| Domain | Clusters | Proposal precision | Robust precision | Noise | Phase prediction | Observed selective gain |
|---|---:|---:|---:|---:|---|---:|
| X-ray | 176 patients | 40.06% | 37.14% | 47.60% | Out of support: noise >40% | **-0.040°** MAE reduction |
| Spider weak | 20 databases | 49.28% | 16.54% | 0% | Selective beats no-op and apply-all | **+11.51 pt** execution accuracy |
| Spider clean | 116 databases | 14.66% | 1.80% | 0% | Out of support: precision <30% | **+0.061 pt**, CI includes zero |

For X-ray, a stricter 0.5° sign threshold leaves 82 detector pairs and produces
62.20% disagreement, so the conclusion that its advantage labels are noisier
than the synthetic grid is robust. The observed X-ray number uses the exact
stored fold aggregate: 2.8013° no-repair versus 2.8416° learned selective
repair. It is slightly more negative than the earlier rounded shorthand
`-0.03°`.

## Figure 1 — real points on the frozen phase surface

`outputs/research/figure1_real_domains_on_synthetic_phase.png`

The colored background is learned gain minus the best calibration-tuned global
action. Stars are plotted at their real coordinates. White space marks regions
that were never simulated; the X-ray and Spider-clean stars deliberately remain
outside the colored support.

## Figure 2 — strategy boundaries

`outputs/research/figure2_selective_strategy_boundaries.png`

The upper row marks cells where mean learned selective gain exceeds no-op. The
lower row marks cells where it exceeds apply-all. Their intersection is the
region where selectivity has positive value over both alternatives. These are
mean-gain boundaries from the existing 4,800 runs, not formal confidence or
reliability guarantees.

## Interpretation

The frozen phase diagram correctly anticipates the Spider-weak positive result:
its precision is near 0.5, rewards are noiseless, and selectivity has substantial
room over both global actions even with only 20 database clusters. It cannot
honestly predict the other two domains. X-ray exceeds the simulated noise range,
while Spider clean has much lower proposal precision than anything simulated.

Therefore tonight's test does **not** prove that the current synthetic phase
diagram explains the real world. It proves a narrower statement: the only
in-support real point is consistent with the phase prediction, and the two
failures occur along axes that the synthetic design identifies as relevant but
did not cover far enough. Extending the frozen grid would be a separate future
experiment, not part of this analysis.

## Reproduction

```bash
MPLCONFIGDIR=/tmp/geomed-mpl python scripts/map_real_domains_to_synthetic_phase.py
```

Machine-readable table: `outputs/research/real_domain_phase_coordinates.csv`.
Full definitions and sensitivity values:
`outputs/research/real_domain_phase_coordinates.json`.
