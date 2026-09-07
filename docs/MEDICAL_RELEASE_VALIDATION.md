# Medical-only release validation

Checked locally on 2026-09-06 (America/Chicago), Python 3.11.

- Full suite: **97 passed, 2 skipped** (5.24 seconds).
- Both skips require a locked evaluation artifact intentionally excluded from
  the public repository: evaluation replay and model inference.
- Offline CLI, "Measure HVA and IMA": completed synthetic job, HVA 15.0 degrees,
  IMA 8.0 degrees, persisted execution contract and decision record.
- Medical-only scope checks reject SQL job payloads and nonmedical pipeline jobs.
- Dashboard checks prevent historical artifact accuracy from appearing as live
  performance cards.

This is software verification, not a new medical model accuracy evaluation.
No live model, clinical image, prospective cohort, PostgreSQL server, MinIO,
Orthanc, or OHIF integration was validated in this release.
Demo citations and similar cases are synthetic placeholders, not clinical
evidence. Production-user impact and clinical utility are unverified.

SQL-specific runtime branches, benchmarks, and demonstrations were removed from
the medical tree. Original shared history remains recoverable through Git.
The original local SQL worktree and medical archive were not edited.
