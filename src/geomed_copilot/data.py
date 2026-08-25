from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path


def load_manifest(manifest: Path, image_dir: Path) -> Iterator[dict]:
    """Yield validated local records without loading image bytes into memory."""
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "patient_id" in record:
                raise ValueError(f"patient_id present in sanitized manifest at line {line_number}")
            image_path = image_dir / record["image_file"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            yield {**record, "image_path": image_path}

