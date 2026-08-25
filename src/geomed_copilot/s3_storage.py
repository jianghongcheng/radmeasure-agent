from __future__ import annotations

import hashlib

from .storage import LocalArtifactStore, StoredArtifact


class S3ArtifactStore:
    """S3-compatible content-addressed artifact adapter."""

    ALLOWED_MEDIA_TYPES = LocalArtifactStore.ALLOWED_MEDIA_TYPES

    def __init__(self, endpoint_url: str, bucket: str, access_key: str,
                 secret_key: str, max_bytes: int = 25 * 1024 * 1024) -> None:
        try:
            import boto3
            from botocore.config import Config
            from botocore.exceptions import ClientError
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install the production extra: pip install -e '.[prod]'") from exc
        self.bucket, self.max_bytes = bucket, max_bytes
        self._client_error = ClientError
        self.client = boto3.client(
            "s3", endpoint_url=endpoint_url,
            aws_access_key_id=access_key, aws_secret_access_key=secret_key,
            config=Config(signature_version="s3v4"), region_name="us-east-1",
        )
        try:
            self.client.head_bucket(Bucket=bucket)
        except ClientError as exc:
            if str(exc.response.get("Error", {}).get("Code")) not in {"404", "NoSuchBucket"}:
                raise
            self.client.create_bucket(Bucket=bucket)

    def put(self, content: bytes, media_type: str) -> tuple[StoredArtifact, bool]:
        if media_type not in self.ALLOWED_MEDIA_TYPES:
            raise ValueError(f"unsupported media type: {media_type}")
        if not content:
            raise ValueError("empty uploads are not accepted")
        if len(content) > self.max_bytes:
            raise ValueError(f"upload exceeds {self.max_bytes} bytes")
        digest = hashlib.sha256(content).hexdigest()
        key = f"sha256/{digest[:2]}/{digest}{self.ALLOWED_MEDIA_TYPES[media_type]}"
        created = False
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
        except self._client_error as exc:
            if str(exc.response.get("Error", {}).get("Code")) not in {"404", "NoSuchKey", "NotFound"}:
                raise
            self.client.put_object(
                Bucket=self.bucket, Key=key, Body=content,
                ContentType=media_type,
                Metadata={"sha256": digest},
            )
            created = True
        return StoredArtifact(digest, f"s3://{self.bucket}/{key}", len(content), media_type), created
