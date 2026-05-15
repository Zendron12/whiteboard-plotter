"""On-disk uploads store with background image preprocessing.

This module factors the upload-related state and behaviour out of
``wall_climber.web_server.BackendRuntime`` so the HTTP backend can focus on
request routing. Behaviour is intentionally preserved byte-for-byte with the
original in-line implementation; public method signatures mirror what the rest
of ``web_server`` already expects.

Responsibilities
----------------
- Persist uploaded payload + metadata under ``<uploads_dir>/<upload_id>.*``.
- Track per-upload preprocessing state in memory (behind a lock).
- Trigger background image vectorization in a single-worker thread pool.
- Periodically garbage-collect stale uploads (best-effort, TTL-based).

Security note
-------------
``load_upload`` treats the on-disk metadata as untrusted JSON. It re-derives
``source_type`` from the stored filename/content-type rather than trusting a
possibly tampered value.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from fastapi import HTTPException, UploadFile

from wall_climber.ingestion.image import vectorize_image_to_canonical_plan
from wall_climber.ingestion.upload_routing import (
    UploadedVectorFile,
    infer_uploaded_source_type,
)


@dataclass(frozen=True)
class PreparedImageArtifact:
    """Cached result of ``vectorize_image_to_canonical_plan`` for an upload."""

    image_result: Any
    defaults: dict[str, Any]
    timings_ms: dict[str, float]


DEFAULT_IMAGE_PREP_OPTIONS: dict[str, Any] = {
    'min_perimeter_px': 8.0,
    'contour_simplify_ratio': 0.001,
    'max_strokes': 4096,
}

UPLOAD_RETENTION_SECONDS = 24 * 60 * 60
UPLOAD_GC_MIN_INTERVAL_SECONDS = 5 * 60


class UploadStore:
    """Owns the uploads directory, per-upload processing state, and worker pool."""

    def __init__(
        self,
        uploads_dir: Path,
        *,
        theta_ref_provider: Callable[[], float],
        max_workers: int = 1,
    ) -> None:
        self._uploads_dir = Path(uploads_dir)
        self._uploads_dir.mkdir(parents=True, exist_ok=True)
        self._theta_ref_provider = theta_ref_provider
        self._lock = threading.Lock()
        self._processing: dict[str, dict[str, Any]] = {}
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix='wall_climber_image_prepare',
        )
        self._last_gc_ts: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def uploads_dir(self) -> Path:
        return self._uploads_dir

    def shutdown(self) -> None:
        try:
            self._pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            # Older Python: no cancel_futures kwarg.
            self._pool.shutdown(wait=False)

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
        self._remember(metadata)
        if metadata['source_type'] == 'image':
            self.ensure_processing(upload_id, metadata=metadata, payload=content)
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

    def processing_snapshot(
        self,
        upload_id: str,
        *,
        metadata: dict[str, Any] | None = None,
        payload: bytes | None = None,
    ) -> dict[str, Any]:
        if metadata is None:
            metadata, payload = self.load_upload(upload_id)
        self._remember(metadata)
        source_type = infer_uploaded_source_type(
            stored_filename=metadata.get('stored_filename'),
            original_filename=metadata.get('original_filename'),
            content_type=metadata.get('normalized_content_type') or metadata.get('content_type'),
            source_type=metadata.get('source_type'),
        )
        if source_type == 'image':
            self.ensure_processing(upload_id, metadata=metadata, payload=payload)
        with self._lock:
            entry = dict(
                self._processing.get(upload_id)
                or self._default_processing_entry(metadata),
            )
        return {
            'upload_id': upload_id,
            'source_type': source_type,
            'state': str(entry.get('state') or 'uploaded'),
            'stage': str(entry.get('stage') or 'uploaded'),
            'progress': float(entry.get('progress') or 0.0),
            'message': str(entry.get('message') or ''),
            'image_size': (
                dict(entry['image_size'])
                if isinstance(entry.get('image_size'), dict)
                else None
            ),
            'route': (
                dict(entry['route'])
                if isinstance(entry.get('route'), dict)
                else None
            ),
            'timings_ms': dict(entry.get('timings_ms') or {}),
            'curve_fit_summary': dict(entry.get('curve_fit_summary') or {}),
        }

    def prepared_image_artifact(
        self,
        upload_id: str,
        *,
        metadata: dict[str, Any] | None = None,
        payload: bytes | None = None,
    ) -> PreparedImageArtifact:
        if metadata is None:
            metadata, payload = self.load_upload(upload_id)
        self.ensure_processing(upload_id, metadata=metadata, payload=payload)
        with self._lock:
            entry = self._processing.get(upload_id)
            artifact = entry.get('artifact') if entry else None
            state = entry.get('state') if entry else None
            message = str(entry.get('message') or '') if entry else ''
        if isinstance(artifact, PreparedImageArtifact):
            return artifact
        if state == 'error':
            raise HTTPException(
                status_code=422, detail=message or 'image preprocessing failed',
            )
        raise HTTPException(status_code=409, detail='image preprocessing is still running')

    def ensure_processing(
        self,
        upload_id: str,
        *,
        metadata: dict[str, Any] | None = None,
        payload: bytes | None = None,
    ) -> None:
        if metadata is None:
            metadata, payload = self.load_upload(upload_id)
        self._remember(metadata)
        source_type = infer_uploaded_source_type(
            stored_filename=metadata.get('stored_filename'),
            original_filename=metadata.get('original_filename'),
            content_type=metadata.get('normalized_content_type') or metadata.get('content_type'),
            source_type=metadata.get('source_type'),
        )
        if source_type != 'image':
            return

        with self._lock:
            entry = self._processing.setdefault(
                upload_id, self._default_processing_entry(metadata),
            )
            if entry.get('artifact') is not None and entry.get('state') == 'ready':
                return
            if entry.get('state') == 'error':
                return
            future = entry.get('future')
            if isinstance(future, Future) and not future.done():
                return
            entry['state'] = 'processing'
            entry['stage'] = 'vectorizing'
            entry['progress'] = 0.15
            entry['message'] = 'Preprocessing image in background.'
            future = self._pool.submit(
                self._prepare_image_upload_worker,
                upload_id,
                payload,
            )
            entry['future'] = future

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _default_processing_entry(self, metadata: dict[str, Any]) -> dict[str, Any]:
        source_type = infer_uploaded_source_type(
            stored_filename=metadata.get('stored_filename'),
            original_filename=metadata.get('original_filename'),
            content_type=metadata.get('normalized_content_type') or metadata.get('content_type'),
            source_type=metadata.get('source_type'),
        )
        image_size = (
            metadata.get('image_size')
            if isinstance(metadata.get('image_size'), dict)
            else None
        )
        entry = {
            'upload_id': metadata.get('upload_id'),
            'source_type': source_type,
            'state': 'ready' if source_type == 'svg' else 'uploaded',
            'stage': 'ready' if source_type == 'svg' else 'uploaded',
            'progress': 1.0 if source_type == 'svg' else 0.0,
            'message': (
                'SVG upload is ready.'
                if source_type == 'svg'
                else 'Upload stored and waiting for preprocessing.'
            ),
            'image_size': dict(image_size) if image_size else None,
            'route': None,
            'timings_ms': {},
            'curve_fit_summary': None,
            'artifact': None,
            'future': None,
        }
        return entry

    def _remember(self, metadata: dict[str, Any]) -> None:
        upload_id = metadata.get('upload_id')
        if not isinstance(upload_id, str) or not upload_id:
            return
        with self._lock:
            existing = self._processing.get(upload_id)
            if existing is None:
                self._processing[upload_id] = self._default_processing_entry(metadata)
                return
            if existing.get('image_size') is None and isinstance(
                metadata.get('image_size'), dict,
            ):
                existing['image_size'] = dict(metadata['image_size'])
            if existing.get('source_type') is None:
                existing['source_type'] = metadata.get('source_type')

    def _prepare_image_upload_worker(
        self, upload_id: str, payload: bytes | None,
    ) -> None:
        try:
            if payload is None:
                metadata, payload = self.load_upload(upload_id)
            else:
                metadata, _ = self.load_upload(upload_id)
            defaults = dict(DEFAULT_IMAGE_PREP_OPTIONS)
            with self._lock:
                entry = self._processing.setdefault(
                    upload_id, self._default_processing_entry(metadata),
                )
                entry['state'] = 'processing'
                entry['stage'] = 'vectorizing'
                entry['progress'] = 0.35
                entry['message'] = 'Tracing and fitting image geometry.'
            ingest_start = time.perf_counter()
            image_result = vectorize_image_to_canonical_plan(
                payload,
                theta_ref=self._theta_ref_provider(),
                **defaults,
            )
            timings_ms = {
                'ingest_ms': max(0.0, (time.perf_counter() - ingest_start) * 1000.0),
            }
            artifact = PreparedImageArtifact(
                image_result=image_result,
                defaults=defaults,
                timings_ms=timings_ms,
            )
            with self._lock:
                entry = self._processing.setdefault(
                    upload_id, self._default_processing_entry(metadata),
                )
                entry['state'] = 'ready'
                entry['stage'] = 'ready'
                entry['progress'] = 1.0
                entry['message'] = 'Vector preview is ready.'
                entry['route'] = image_result.route_decision.to_dict()
                entry['timings_ms'] = dict(timings_ms)
                entry['curve_fit_summary'] = dict(
                    image_result.to_metadata().get('pipeline', {}).get(
                        'curve_fit_summary',
                    ) or {},
                )
                entry['artifact'] = artifact
                entry['future'] = None
        except Exception as exc:
            with self._lock:
                entry = self._processing.setdefault(upload_id, {'upload_id': upload_id})
                entry['state'] = 'error'
                entry['stage'] = 'error'
                entry['progress'] = 1.0
                entry['message'] = str(exc)
                entry['artifact'] = None
                entry['future'] = None

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
        removed_ids: set[str] = set()
        for entry in entries:
            try:
                if not entry.is_file():
                    continue
                if entry.stat().st_mtime >= cutoff:
                    continue
                entry.unlink(missing_ok=True)
                removed_ids.add(entry.stem)
            except OSError:
                continue
        if removed_ids:
            with self._lock:
                for upload_id in removed_ids:
                    self._processing.pop(upload_id, None)


__all__ = [
    'PreparedImageArtifact',
    'UploadStore',
    'DEFAULT_IMAGE_PREP_OPTIONS',
    'UPLOAD_RETENTION_SECONDS',
    'UPLOAD_GC_MIN_INTERVAL_SECONDS',
]
