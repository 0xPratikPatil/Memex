from __future__ import annotations

import hashlib
import os
from pathlib import Path

from memex.engine.sources import Source, SourceFile, register_source


@register_source
class S3Source(Source):
    type = "s3"

    def __init__(
        self,
        name: str,
        bucket: str,
        prefix: str = "",
        extensions: list[str] | None = None,
        aws_access_key: str = "",
        aws_secret_key: str = "",
        region: str = "us-east-1",
        cache_dir: str = "",
    ) -> None:
        self.name = name
        self.bucket = bucket
        self.prefix = prefix
        self.extensions = extensions or []
        self.region = region
        self._aws_access_key = aws_access_key or os.getenv("AWS_ACCESS_KEY_ID", "")
        self._aws_secret_key = aws_secret_key or os.getenv("AWS_SECRET_ACCESS_KEY", "")
        self._cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "rag" / "s3"

    def _get_client(self):
        try:
            import boto3
        except ImportError as exc:
            raise ImportError("boto3 is required for S3 sources. Install it with: pip install boto3") from exc

        kwargs: dict = {"region_name": self.region}
        if self._aws_access_key:
            kwargs["aws_access_key_id"] = self._aws_access_key
        if self._aws_secret_key:
            kwargs["aws_secret_access_key"] = self._aws_secret_key
        return boto3.client("s3", **kwargs)

    def list_files(self) -> list[SourceFile]:
        client = self._get_client()
        result: list[SourceFile] = []
        paginator = client.get_paginator("list_objects_v2")

        pages = paginator.paginate(Bucket=self.bucket, Prefix=self.prefix)
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                suffix = Path(key).suffix.lower()
                if self.extensions and suffix not in self.extensions:
                    continue
                result.append(
                    SourceFile(
                        name=Path(key).name,
                        path=key,
                        size=obj["Size"],
                        modified_at=obj["LastModified"].timestamp(),
                    )
                )

        return result

    def get_content_hash(self, file: SourceFile) -> str:
        client = self._get_client()
        resp = client.get_object(Bucket=self.bucket, Key=file.path)
        h = hashlib.sha256()
        for chunk in resp["Body"].iter_chunks(chunk_size=8192):
            h.update(chunk)
        return h.hexdigest()

    def download(self, file: SourceFile, dest: Path) -> Path:
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / file.name

        if target.exists():
            st = target.stat()
            if st.st_size == file.size and abs(st.st_mtime - file.modified_at) < 1.0:
                return target

        client = self._get_client()
        client.download_file(self.bucket, file.path, str(target))
        return target
