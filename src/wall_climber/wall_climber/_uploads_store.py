"""On-disk uploads store.

Persists uploaded payloads (sketch images / SVG markup) under
``<uploads_dir>/<upload_id>.*`` so the rest of the backend can fetch
them by id later. The colored-image preprocessing worker that used to
live here was removed alongside the wider color stack: every upload
now goes through the sketch pipeline directly, which does its own
preprocessing inline.

Security note
-------------
``load_upload`` treats the on-disk metadata as untrusted JSON. It
re-derives ``source_type`` from the stored filename / content-type
rather than trusting a possibly tampered value.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from fastapi import HTTPException, UploadFile

from wall_climber.ingestion.upload_routing import (
    UploadedVectorFile,
    infer_uploaded_source_type,
)


UPLOAD_RETENTION_SECONDS = 24 * 60 * 60
UPLOAD_GC_MIN_INTERVAL_SECONDS = 5 * 60


class UploadStore:
    """Owns the uploads directory.

    The constructor still accepts ``theta_ref_provider`` and ``max_workers``
    for API compatibility with callers that have not been migrated yet, but
    they are unused: there is no longer a background worker.
    """

    def __init__(
        self,
        uploads_dir: Path,
        *,
        theta_ref_provider: Callable[[], float] | None = None,
        max_workers: int = 1,
    ) -> None:
        self._uploads_dir = Path(uploads_dir)
        self._uploads_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last_gc_ts: float = 0.0
        # Kept so external code that still references these attrs does not
        # break; both are unused by the slimmed-down store.
        self._theta_ref_provider = theta_ref_provider
        self._max_workers = max_workers

    @property
    def uploads_dir(self) -> Path:
        return self._uploads_dir

    def shutdown(self) -> None:
        # No background workers to tear down anymore.
        return

    # ------------------------------------------------------------------
    # Public CRUD
    # ------------------------------------------------------------------

    def store_upload(
        self,
        upload: UploadFile,
        content: bytes,
        *,
        upload_details: UploadedVectorFile,
    ) -> dict[str, Any]:
        self._run_gc_if_due()
        upload_id = uuid.uuid4().hex
        extension = upload_details.extension
        payload_path = self._uploads_dir / f'{upload_id}{extension}'
        metadata_path = self._uploads_dir / f'{upload_id}.json'
        payload_path.write_bytes(content)
        metadata: dict[str, Any] = {
            'upload_id': upload_id,
            'stored_filename': payload_path.name,
            'metadata_filename': metadata_path.name,
            'original_filename': upload.filename,
            'content_type': upload.content_type,
            'normalized_content_type': upload_details.normalized_content_type,
            'source_type': upload_details.source_type,
            'size_bytes': len(content),
            'stored_only': True,
            'created_at': datetime.now(timezone.utc).isoformat(),
        }
        if upload_details.image_size is not None:
            metadata['image_size'] = {
                'width_px': int(upload_details.image_size[0]),
                'height_px': int(upload_details.image_size[1]),
            }
        metadata_path.write_text(
            json.dumps(metadata, separators=(',', ':'), indent=2),
            encoding='utf-8',
        )
        return metadata

    def load_upload(self, upload_id: str) -> tuple[dict[str, Any], bytes]:
        metadata_path = self._uploads_dir / f'{upload_id}.json'
        if not metadata_path.is_file():
            raise HTTPException(status_code=404, detail='upload_id was not found')
        try:
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f'failed to read upload metadata: {exc}',
            )
        if not isinstance(metadata, dict):
            raise HTTPException(status_code=500, detail='upload metadata is invalid')
        metadata['source_type'] = infer_uploaded_source_type(
            stored_filename=metadata.get('stored_filename'),
            original_filename=metadata.get('original_filename'),
            content_type=metadata.get('normalized_content_type') or metadata.get('content_type'),
            source_type=metadata.get('source_type'),
        )
        stored_filename = metadata.get('stored_filename')
        if not isinstance(stored_filename, str) or not stored_filename:
            raise HTTPException(
                status_code=500, detail='upload metadata is missing stored filename',
            )
        payload_path = self._uploads_dir / stored_filename
        if not payload_path.is_file():
            raise HTTPException(status_code=404, detail='stored upload payload is missing')
        try:
            payload = payload_path.read_bytes()
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f'failed to read upload payload: {exc}',
            )
        return metadata, payload

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_gc_if_due(self) -> None:
        """Best-effort cleanup of stale uploads.

        Removes any ``<id>.json`` / ``<id>.<ext>`` pair whose mtime is older
        than :data:`UPLOAD_RETENTION_SECONDS`. Runs at most every
        :data:`UPLOAD_GC_MIN_INTERVAL_SECONDS`. Errors are swallowed so an
        upload never fails because of cleanup.
        """
        now = time.time()
        if now - self._last_gc_ts < UPLOAD_GC_MIN_INTERVAL_SECONDS:
            return
        self._last_gc_ts = now
        cutoff = now - UPLOAD_RETENTION_SECONDS
        try:
            entries = list(self._uploads_dir.iterdir())
        except OSError:
            return
        for entry in entries:
            try:
                if not entry.is_file():
                    continue
                if entry.stat().st_mtime >= cutoff:
                    continue
                entry.unlink(missing_ok=True)
            except OSError:
                continue


__all__ = [
    'UploadStore',
    'UPLOAD_RETENTION_SECONDS',
    'UPLOAD_GC_MIN_INTERVAL_SECONDS',
]
