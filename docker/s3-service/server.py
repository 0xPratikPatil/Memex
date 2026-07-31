"""S3 file access service for Memex RAG."""

from __future__ import annotations

import hashlib
import logging
import os
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

if TYPE_CHECKING:
    pass

try:
    import boto3
except ImportError:
    boto3 = None  # type: ignore[assignment]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Memex S3 Access Service", version="0.1.0")


class S3Credentials(BaseModel):
    aws_access_key: str
    aws_secret_key: str
    region: str = "us-east-1"


class ListRequest(BaseModel):
    bucket: str
    prefix: str = ""
    extensions: list[str] | None = None
    aws_access_key: str
    aws_secret_key: str
    region: str = "us-east-1"


class FileInfo(BaseModel):
    name: str
    path: str
    size: int
    modified_at: str


class ListResponse(BaseModel):
    files: list[FileInfo]


class DownloadRequest(BaseModel):
    bucket: str
    key: str
    dest_path: str
    aws_access_key: str
    aws_secret_key: str
    region: str = "us-east-1"


class DownloadResponse(BaseModel):
    local_path: str
    content_hash: str


class HashRequest(BaseModel):
    bucket: str
    key: str
    aws_access_key: str
    aws_secret_key: str
    region: str = "us-east-1"


class HashResponse(BaseModel):
    content_hash: str


def _get_client(creds: S3Credentials):
    if boto3 is None:
        raise HTTPException(status_code=503, detail="boto3 is not installed in the container")
    return boto3.client(
        "s3",
        region_name=creds.region,
        aws_access_key_id=creds.aws_access_key,
        aws_secret_access_key=creds.aws_secret_key,
    )


def _stream_hash(client, bucket: str, key: str) -> str:
    hasher = hashlib.sha256()
    resp = client.get_object(Bucket=bucket, Key=key)
    for chunk in resp["Body"].iter_chunks(chunk_size=1024 * 256):
        hasher.update(chunk)
    return hasher.hexdigest()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/list", response_model=ListResponse)
def list_objects(req: ListRequest):
    creds = S3Credentials(
        aws_access_key=req.aws_access_key,
        aws_secret_key=req.aws_secret_key,
        region=req.region,
    )
    client = _get_client(creds)

    try:
        paginator = client.get_paginator("list_objects_v2")
        files: list[FileInfo] = []

        for page in paginator.paginate(Bucket=req.bucket, Prefix=req.prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                if req.extensions:
                    ext = "." + key.rsplit(".", 1)[-1].lower() if "." in key else ""
                    if ext not in req.extensions:
                        continue
                name = key.rsplit("/", 1)[-1]
                files.append(
                    FileInfo(
                        name=name,
                        path=key,
                        size=obj["Size"],
                        modified_at=obj["LastModified"].isoformat(),
                    )
                )

        files.sort(key=lambda f: f.path)
        logger.info("Listed %d file(s) from s3://%s/%s", len(files), req.bucket, req.prefix)
        return ListResponse(files=files)

    except Exception as exc:
        logger.error("S3 list failed for s3://%s/%s: %s", req.bucket, req.prefix, exc)
        raise HTTPException(status_code=500, detail="S3 list failed") from exc


@app.post("/download", response_model=DownloadResponse)
def download_object(req: DownloadRequest):
    creds = S3Credentials(
        aws_access_key=req.aws_access_key,
        aws_secret_key=req.aws_secret_key,
        region=req.region,
    )
    client = _get_client(creds)

    dest = os.path.abspath(req.dest_path)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)

    try:
        client.download_file(req.bucket, req.key, dest)
        sha = _stream_hash(client, req.bucket, req.key)
        logger.info("Downloaded s3://%s/%s -> %s", req.bucket, req.key, dest)
        return DownloadResponse(local_path=dest, content_hash=sha)

    except Exception as exc:
        logger.error("S3 download failed for s3://%s/%s: %s", req.bucket, req.key, exc)
        raise HTTPException(status_code=500, detail="S3 download failed") from exc


@app.post("/hash", response_model=HashResponse)
def hash_object(req: HashRequest):
    creds = S3Credentials(
        aws_access_key=req.aws_access_key,
        aws_secret_key=req.aws_secret_key,
        region=req.region,
    )
    client = _get_client(creds)

    try:
        sha = _stream_hash(client, req.bucket, req.key)
        logger.info("Hashed s3://%s/%s: %s", req.bucket, req.key, sha[:12])
        return HashResponse(content_hash=sha)

    except Exception as exc:
        logger.error("S3 hash failed for s3://%s/%s: %s", req.bucket, req.key, exc)
        raise HTTPException(status_code=500, detail="S3 hash failed") from exc
