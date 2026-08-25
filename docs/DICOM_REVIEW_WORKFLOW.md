# DICOM and human-review workflow

This is a research-only local workflow, not a PACS integration certified for
clinical use.

1. Orthanc stores the source DICOM and exposes QIDO/WADO/STOW through DICOMWeb.
2. GeoMed retrieves a selected instance through a fixed, authenticated Orthanc
   URL; callers cannot supply arbitrary upstream URLs.
3. The API removes common direct identifiers and all private tags before the
   derived object is persisted in MinIO. The source object in Orthanc is not
   modified.
4. The inference service performs single-frame DICOM decoding, robust percentile
   windowing, MONOCHROME1 inversion where needed, image-quality checks, and live
   HVA/IMA inference.
5. Results always enter `needs_review`. An admin reviewer can approve, reject,
   or supply corrected HVA/IMA values. A second final decision is rejected.
6. `GET /v1/jobs/{job_id}/report` produces a report only for
   `review_approved` jobs and includes model hash/version, quality metrics,
   reviewer identity, corrections, provenance, and the research disclaimer.

Known limitations: only single-frame uncompressed DICOM pixel data is verified;
the de-identification routine is a narrow engineering safeguard, not a formal
DICOM PS3.15 confidentiality-profile implementation; OHIF displays studies but
measurement overlays are not yet synchronized back through DICOM SR.
