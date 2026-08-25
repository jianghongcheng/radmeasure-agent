#!/usr/bin/env python3
"""Plot preregistered phase predictions against the revealed intervention."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "research"
prereg = json.loads((OUT / "spider_precision_intervention_preregistered.json").read_text())
revealed = json.loads((OUT / "spider_precision_intervention_revealed.json").read_text())
targets = prereg["design"]["target_precisions"]
predicted = np.asarray([
    prereg["phase_predictions"][str(p)]["learned_gain_mean"] * 100 for p in targets
])
observed = np.asarray([
    revealed["results"][str(p)]["pooled"]["learned"]["gain"] * 100 for p in targets
])
low = np.asarray([
    revealed["results"][str(p)]["learned_minus_no_op"]["ci_low"] * 100 for p in targets
])
high = np.asarray([
    revealed["results"][str(p)]["learned_minus_no_op"]["ci_high"] * 100 for p in targets
])

fig, ax = plt.subplots(figsize=(6.7, 4.2))
ax.plot(targets, predicted, "o--", color="#4477AA", label="Frozen phase prediction")
ax.errorbar(
    targets, observed, yerr=np.vstack([observed-low, high-observed]),
    fmt="s-", capsize=4, color="#AA3377", label="Matched-switch observed"
)
ax.axhline(0, color="black", linewidth=.8)
ax.axvline(.25, color="#4477AA", linestyle=":", linewidth=1,
           label="Predicted reliable transition")
ax.set_xlabel("Manipulated proposal precision")
ax.set_ylabel("Learned gain over no-op (accuracy points)")
ax.set_title("Preregistered precision intervention falsifies the first-order phase")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "spider_precision_intervention.png", dpi=240)
plt.close(fig)
