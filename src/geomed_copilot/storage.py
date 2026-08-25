from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StoredArtifact:
    sha256: str
    path: str
    size_bytes: int
    media_type: str


class LocalArtifactStore:
    """Content-addressed local adapter with bounded, allow-listed uploads."""

    ALLOWED_MEDIA_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "application/dicom": ".dcm"}

    def __init__(self, root: Path, max_bytes: int = 25 * 1024 * 1024) -> None:
        self.root, self.max_bytes = root, max_bytes
        root.mkdir(parents=True, exist_ok=True)

    def put(self, content: bytes, media_type: str) -> tuple[StoredArtifact, bool]:
        if media_type not in self.ALLOWED_MEDIA_TYPES:
            raise ValueError(f"unsupported media type: {media_type}")
        if not content:
            raise ValueError("empty uploads are not accepted")
        if len(content) > self.max_bytes:
            raise ValueError(f"upload exceeds {self.max_bytes} bytes")
        digest = hashlib.sha256(content).hexdigest()
        suffix = self.ALLOWED_MEDIA_TYPES[media_type]
        target = self.root / digest[:2] / f"{digest}{suffix}"
        created = not target.exists()
        if created:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_bytes(content)
            temporary.replace(target)
        return StoredArtifact(digest, str(target), len(content), media_type), created
