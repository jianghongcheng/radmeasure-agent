# Data policy

No radiographs or patient identifiers are committed to this repository.

## HVAngleEst

The local preparation command creates patient-disjoint JSONL manifests while
leaving images in their original directory:

```bash
PYTHONPATH=src python scripts/prepare_hvangleest.py \
  --source-csv /path/to/HVAngleEst/datasets.csv \
  --image-dir /path/to/HVAngleEst/images \
  --output-dir data/processed/hvangleest
```

Generated records exclude `patient_id` and replace dataset row identifiers with
opaque deterministic sample IDs. The split is performed at patient level.

Validate that the sanitized landmark coordinates reproduce the source angles:

```bash
PYTHONPATH=src python scripts/audit_hvangle_geometry.py \
  --manifest-dir data/processed/hvangleest \
  --output data/processed/hvangleest/geometry_audit.json
```

HVAngleEst is an open-access research dataset described by Wang et al. (2025):
<https://doi.org/10.1038/s41597-025-05261-9>. Its associated publication is
CC BY-NC-ND 4.0. Confirm the dataset repository's terms before redistributing
images or modified annotations. This project therefore keeps all generated data
ignored by Git by default.

## AASCE

AASCE data remains outside the repository until its challenge/data-use terms
are recorded locally. Public availability is not automatically permission to
redistribute a copy.
