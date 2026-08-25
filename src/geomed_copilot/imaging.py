from __future__ import annotations

import io
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ImageQuality:
    width: int
    height: int
    contrast_std: float
    clipped_fraction: float
    gradient_energy: float
    passed: bool
    reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


DIRECT_IDENTIFIER_KEYWORDS = (
    "PatientName", "PatientID", "PatientBirthDate", "PatientSex",
    "AccessionNumber", "InstitutionName", "ReferringPhysicianName",
    "StudyID",
)


def deidentify_dicom(content: bytes) -> tuple[bytes, list[str]]:
    """Remove common direct identifiers and all private tags before persistence."""
    import pydicom

    dataset = pydicom.dcmread(io.BytesIO(content), force=False)
    removed = []
    for keyword in DIRECT_IDENTIFIER_KEYWORDS:
        if keyword in dataset and str(dataset.get(keyword, "")).strip():
            removed.append(keyword)
            del dataset[keyword]
    dataset.remove_private_tags()
    dataset.PatientIdentityRemoved = "YES"
    dataset.DeidentificationMethod = "GeoMed direct-identifier removal v1"
    stream = io.BytesIO()
    dataset.save_as(stream, enforce_file_format=True)
    return stream.getvalue(), removed


def _quality(array) -> ImageQuality:
    import numpy as np

    pixels = np.asarray(array, dtype=np.float32)
    if pixels.ndim == 3:
        pixels = pixels.mean(axis=2)
    if pixels.max() > 1:
        pixels /= 255.0
    height, width = pixels.shape
    contrast = float(pixels.std())
    clipped = float(((pixels <= 0.01) | (pixels >= 0.99)).mean())
    gy, gx = np.gradient(pixels)
    gradient = float(np.mean(gx * gx + gy * gy))
    reasons = []
    if min(width, height) < 128:
        reasons.append("resolution_below_128px")
    if contrast < 0.04:
        reasons.append("low_contrast")
    if clipped > 0.75:
        reasons.append("excessive_pixel_clipping")
    if gradient < 0.00002:
        reasons.append("low_detail_or_blur")
    return ImageQuality(width, height, contrast, clipped, gradient, not reasons, reasons)


def decode_medical_image(content: bytes, media_type: str):
    """Decode JPEG/PNG or single-frame DICOM into RGB PIL plus safe metadata."""
    import numpy as np
    from PIL import Image

    if media_type != "application/dicom":
        image = Image.open(io.BytesIO(content)).convert("RGB")
        return image, _quality(np.asarray(image)), {"format": image.format or media_type}

    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pydicom is required for DICOM inference") from exc
    dataset = pydicom.dcmread(io.BytesIO(content), force=False)
    if int(getattr(dataset, "NumberOfFrames", 1)) != 1:
        raise ValueError("only single-frame radiographs are supported")
    pixels = dataset.pixel_array.astype(np.float32)
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    pixels = pixels * slope + intercept
    lo, hi = np.percentile(pixels, [0.5, 99.5])
    if hi <= lo:
        raise ValueError("DICOM pixel data has no usable dynamic range")
    pixels = np.clip((pixels - lo) / (hi - lo), 0, 1)
    if str(getattr(dataset, "PhotometricInterpretation", "")) == "MONOCHROME1":
        pixels = 1.0 - pixels
    uint8 = (pixels * 255).astype(np.uint8)
    image = Image.fromarray(uint8, mode="L").convert("RGB")
    metadata = {
        "format": "DICOM",
        "sop_class_uid": str(getattr(dataset, "SOPClassUID", "")),
        "modality": str(getattr(dataset, "Modality", "")),
        "body_part_examined": str(getattr(dataset, "BodyPartExamined", "")),
        "photometric_interpretation": str(getattr(dataset, "PhotometricInterpretation", "")),
        "contains_direct_identifiers": any(
            bool(str(getattr(dataset, tag, "")).strip())
            for tag in ("PatientName", "PatientID", "PatientBirthDate", "AccessionNumber")
        ),
    }
    return image, _quality(uint8), metadata
