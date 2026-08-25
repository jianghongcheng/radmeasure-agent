from pathlib import Path

from geomed_copilot.storage import LocalArtifactStore


def test_artifact_store_is_content_addressed_and_deduplicates(tmp_path: Path):
    store = LocalArtifactStore(tmp_path, max_bytes=10)
    first, created = store.put(b"image", "image/jpeg")
    second, created_again = store.put(b"image", "image/jpeg")
    assert created is True and created_again is False
    assert first.sha256 == second.sha256
    assert Path(first.path).read_bytes() == b"image"


def test_artifact_store_rejects_type_empty_and_size(tmp_path: Path):
    store = LocalArtifactStore(tmp_path, max_bytes=3)
    for content, media_type in [(b"x", "text/plain"), (b"", "image/png"), (b"long", "image/png")]:
        try:
            store.put(content, media_type)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("invalid upload accepted")
