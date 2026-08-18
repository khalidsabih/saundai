from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SIDECAR_SUFFIX = ".saund-object.json"


def _truthy(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _user_token(user_id: str) -> str:
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:24]


def object_sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + SIDECAR_SUFFIX)


@dataclass(frozen=True)
class ObjectStoreConfig:
    backend: str
    bucket: str | None
    endpoint_url: str | None
    region: str
    access_key_id: str | None
    secret_access_key: str | None
    prefix: str
    presign_seconds: int
    delete_after_materialize: bool

    @classmethod
    def from_environment(cls) -> "ObjectStoreConfig":
        bucket = os.environ.get("SAUND_OBJECT_STORE_BUCKET", "").strip() or None
        requested = os.environ.get("SAUND_OBJECT_STORE", "").strip().lower()
        backend = requested or ("s3" if bucket else "off")
        return cls(
            backend=backend,
            bucket=bucket,
            endpoint_url=os.environ.get("SAUND_OBJECT_STORE_ENDPOINT", "").strip() or None,
            region=os.environ.get("SAUND_OBJECT_STORE_REGION", "auto").strip() or "auto",
            access_key_id=(os.environ.get("SAUND_OBJECT_STORE_ACCESS_KEY_ID", "").strip() or os.environ.get("AWS_ACCESS_KEY_ID", "").strip() or None),
            secret_access_key=(os.environ.get("SAUND_OBJECT_STORE_SECRET_ACCESS_KEY", "").strip() or os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip() or None),
            prefix=(os.environ.get("SAUND_OBJECT_STORE_PREFIX", "saund").strip(" /") or "saund"),
            presign_seconds=max(60, min(86400, int(os.environ.get("SAUND_OBJECT_STORE_PRESIGN_SECONDS", "900")))),
            delete_after_materialize=_truthy(os.environ.get("SAUND_OBJECT_STORE_DELETE_AFTER_MATERIALIZE")),
        )

    @property
    def enabled(self) -> bool:
        return self.backend == "s3" and bool(self.bucket)


class ObjectStore:
    """Minimal S3-compatible object boundary for hosted SAUND audio.

    Cloudflare R2 and AWS S3 both implement the operations used here. The
    browser receives only short-lived presigned URLs; permanent object-store
    credentials remain server-side.
    """

    def __init__(self, config: ObjectStoreConfig | None = None) -> None:
        self.config = config or ObjectStoreConfig.from_environment()
        self._client: Any = None
        if self.config.backend not in {"off", "s3"}:
            raise RuntimeError(f"Unsupported SAUND_OBJECT_STORE backend: {self.config.backend}")
        if self.config.backend == "s3" and not self.config.bucket:
            raise RuntimeError("SAUND_OBJECT_STORE=s3 requires SAUND_OBJECT_STORE_BUCKET.")

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def _s3(self):
        if not self.enabled:
            raise RuntimeError("Hosted object storage is not configured.")
        if self._client is not None:
            return self._client
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:  # pragma: no cover - hosted image only
            raise RuntimeError(
                "S3-compatible storage requires the hosted dependency extra: install SAUND.AI with .[hosted]."
            ) from exc
        kwargs: dict[str, Any] = {
            "service_name": "s3",
            "region_name": self.config.region,
            "config": Config(signature_version="s3v4", retries={"max_attempts": 4, "mode": "standard"}),
        }
        if self.config.endpoint_url:
            kwargs["endpoint_url"] = self.config.endpoint_url
        if self.config.access_key_id:
            kwargs["aws_access_key_id"] = self.config.access_key_id
        if self.config.secret_access_key:
            kwargs["aws_secret_access_key"] = self.config.secret_access_key
        self._client = boto3.client(**kwargs)
        return self._client

    def user_prefix(self, user_id: str) -> str:
        return f"{self.config.prefix}/users/{_user_token(user_id)}/"

    def owns_key(self, user_id: str, key: str) -> bool:
        return str(key).startswith(self.user_prefix(user_id))

    def upload_key(self, user_id: str, filename: str) -> str:
        return f"{self.user_prefix(user_id)}uploads/{uuid.uuid4().hex}/{Path(filename).name}"

    def result_key(self, user_id: str, job_id: str, filename: str) -> str:
        return f"{self.user_prefix(user_id)}results/{job_id}/{Path(filename).name}"

    def create_upload_intent(
        self,
        *,
        user_id: str,
        filename: str,
        size: int,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("Direct object uploads are disabled.")
        key = self.upload_key(user_id, filename)
        media_type = content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        params = {"Bucket": self.config.bucket, "Key": key, "ContentType": media_type}
        url = self._s3().generate_presigned_url(
            "put_object",
            Params=params,
            ExpiresIn=self.config.presign_seconds,
            HttpMethod="PUT",
        )
        return {
            "object_key": key,
            "upload_url": url,
            "method": "PUT",
            "headers": {"Content-Type": media_type},
            "expires_in": self.config.presign_seconds,
            "expected_size": int(size),
        }

    def head(self, key: str) -> dict[str, Any]:
        response = self._s3().head_object(Bucket=self.config.bucket, Key=key)
        return {
            "size": int(response.get("ContentLength", 0)),
            "content_type": str(response.get("ContentType") or "application/octet-stream"),
            "etag": str(response.get("ETag") or "").strip('"'),
        }

    def materialize(self, key: str, target: Path, *, expected_size: int | None = None) -> dict[str, Any]:
        metadata = self.head(key)
        if expected_size is not None and int(expected_size) != metadata["size"]:
            raise ValueError(
                f"Uploaded object size mismatch for {Path(target).name}: expected {expected_size}, got {metadata['size']}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
        try:
            digest = hashlib.sha256()
            with temporary.open("wb") as handle:
                class _HashingWriter:
                    def write(self, data: bytes) -> int:
                        digest.update(data)
                        return handle.write(data)

                    def __getattr__(self, name: str):
                        return getattr(handle, name)

                self._s3().download_fileobj(self.config.bucket, key, _HashingWriter())
            if temporary.stat().st_size != metadata["size"]:
                raise IOError("Object-store download did not match the advertised content length.")
            metadata["sha256"] = digest.hexdigest()
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        if self.config.delete_after_materialize:
            self.delete(key)
            object_sidecar_path(target).unlink(missing_ok=True)
        else:
            self.write_sidecar(target, key=key, metadata=metadata)
        return metadata

    def publish_file(self, path: Path, *, user_id: str, job_id: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        key = self.result_key(user_id, job_id, path.name)
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._s3().upload_file(
            str(path),
            self.config.bucket,
            key,
            ExtraArgs={"ContentType": content_type},
        )
        metadata = {"size": path.stat().st_size, "content_type": content_type}
        self.write_sidecar(path, key=key, metadata=metadata)
        return {
            "object_key": key,
            "media_url": self.presign_get(key, filename=path.name),
        }

    def presign_get(self, key: str, *, filename: str | None = None) -> str:
        params: dict[str, Any] = {"Bucket": self.config.bucket, "Key": key}
        if filename:
            safe = Path(filename).name.replace('"', "")
            params["ResponseContentDisposition"] = f'inline; filename="{safe}"'
        return self._s3().generate_presigned_url(
            "get_object",
            Params=params,
            ExpiresIn=self.config.presign_seconds,
            HttpMethod="GET",
        )

    def delete(self, key: str) -> None:
        if not self.enabled:
            return
        self._s3().delete_object(Bucket=self.config.bucket, Key=key)

    def delete_user_objects(self, user_id: str) -> int:
        if not self.enabled:
            return 0
        client = self._s3()
        prefix = self.user_prefix(user_id)
        deleted = 0
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.config.bucket, Prefix=prefix):
            objects = [{"Key": item["Key"]} for item in page.get("Contents", [])]
            for offset in range(0, len(objects), 1000):
                batch = objects[offset : offset + 1000]
                if batch:
                    client.delete_objects(Bucket=self.config.bucket, Delete={"Objects": batch, "Quiet": True})
                    deleted += len(batch)
        return deleted

    def ready(self) -> bool:
        if not self.enabled:
            return True
        try:
            self._s3().head_bucket(Bucket=self.config.bucket)
            return True
        except Exception:
            return False

    def status(self) -> dict[str, Any]:
        return {
            "backend": self.config.backend,
            "enabled": self.enabled,
            "bucket_configured": bool(self.config.bucket),
            "endpoint_configured": bool(self.config.endpoint_url),
            "region": self.config.region if self.enabled else None,
            "presign_seconds": self.config.presign_seconds if self.enabled else None,
            "direct_browser_upload": self.enabled,
        }

    @staticmethod
    def write_sidecar(path: Path, *, key: str, metadata: dict[str, Any]) -> None:
        sidecar = object_sidecar_path(path)
        payload = {
            "schema_version": "1.0",
            "object_key": key,
            "size": int(metadata.get("size", path.stat().st_size if path.exists() else 0)),
            "content_type": metadata.get("content_type"),
            "etag": metadata.get("etag"),
        }
        sidecar.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def read_sidecar(path: Path) -> dict[str, Any] | None:
        sidecar = object_sidecar_path(path)
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict) or not payload.get("object_key"):
            return None
        return payload

    def media_url_for_path(self, path: Path) -> str | None:
        if not self.enabled:
            return None
        payload = self.read_sidecar(path)
        if payload is None:
            return None
        return self.presign_get(str(payload["object_key"]), filename=path.name)


def remove_materialized_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
