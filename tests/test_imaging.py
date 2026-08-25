import io

import numpy as np
from PIL import Image

from geomed_copilot.imaging import decode_medical_image


def _png(array):
    stream = io.BytesIO()
    Image.fromarray(array.astype("uint8")).save(stream, format="PNG")
    return stream.getvalue()


def test_image_quality_accepts_detailed_radiograph_like_pixels():
    y, x = np.mgrid[:256, :256]
    pixels = ((x + y + 35 * np.sin(x / 7)) % 256).astype("uint8")
    image, quality, metadata = decode_medical_image(_png(pixels), "image/png")
    assert image.mode == "RGB"
    assert quality.passed is True
    assert metadata["format"] == "image/png"


def test_image_quality_rejects_blank_or_tiny_input():
    _, quality, _ = decode_medical_image(_png(np.zeros((64, 64))), "image/png")
    assert quality.passed is False
    assert "resolution_below_128px" in quality.reasons
    assert "low_contrast" in quality.reasons
