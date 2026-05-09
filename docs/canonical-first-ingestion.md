# Canonical-First Ingestion

## Status

Accepted for Gate 1 of the curve-aware migration.

## Decision

The internal source of truth for new text, SVG, and image build paths is `CanonicalPathPlan`.

`DrawPlan` and `DrawPolyline` were temporary compatibility/export formats during the initial
migration, but they are no longer part of the live runtime transport.

## Canonical Model

The canonical command set is:

- `PenUp`
- `PenDown`
- `TravelMove`
- `LineSegment`
- `ArcSegment`
- `QuadraticBezier`
- `CubicBezier`

## Ingestion Boundaries

- Text ingestion may still use grouped glyph outlines as a placement-oriented intermediate.
- SVG ingestion may still flatten unsupported source shapes during parsing today, but it must emit a canonical plan immediately after ingestion output is available.
- Image ingestion remains on the current contour tracer for now, but it must emit a canonical plan immediately after tracing.

## Placement and Cleanup

- Placement remains in Python during this gate.
- Cleanup remains in Python during this gate.
- Preview sampling and legacy export sampling happen late, through `canonical_adapters` and the C++ geometry sampler when available.

## Preview/Draw Cache Contract

Preview endpoints for text, SVG, uploaded SVG/image files, and sketch centerline
inputs now store the generated `CanonicalPathPlan` in a process-local preview
cache. Each preview response returns:

- `preview_id`
- `canonical_hash`
- the existing preview payload
- a compatibility `commit_request` that includes `preview_id`

`canonical_hash` is a SHA-256 hash of a stable serialized canonical plan with
rounded numeric values. It is used to prove that preview and draw are using the
same canonical geometry.

The generic draw endpoint is:

```text
POST /api/draw
```

It accepts `preview_id`, loads the cached `CanonicalPathPlan`, validates and
exports it to `PrimitivePathPlan`, and publishes it. It does not re-read or
re-vectorize the original source. Compatibility commit endpoints still exist
for the current UI, but when a `preview_id` is supplied they draw from the cache.

## Compatibility Rules

- Public HTTP endpoints do not change.
- Runtime transport now uses `PrimitivePathPlan`.
- Final preview-first draw flows should use `preview_id`; source-rebuild commit
  paths are compatibility behavior while the UI migration is in progress.
- `DrawPlan` may still appear only in older diagnostic/export helpers while those seams are
  being retired.
- `pen_strokes_to_canonical_plan()` and `canonical_plan_from_pen_strokes()` are legacy compatibility helpers and must not be used by the text/SVG/image production builders.

## Non-Goals for Gate 1

- No executor upgrade yet.
- No dual image router yet.
- No AI preprocessing.
- No primitive-native ROS transport.
