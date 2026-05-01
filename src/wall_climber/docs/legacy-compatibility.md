# Legacy Compatibility Boundary

## Status

The internal source of truth for drawing content is `CanonicalPathPlan`.

`DrawPlan` and `DrawPolyline` are no longer used as runtime ROS transport.
Stroke-shaped JSON payloads remain supported only where the browser preview still needs a
sampled render payload.

`PrimitivePathPlan` is now the only execution transport used by the web backend and the
executor.

## Rules

- Text, SVG, and image production builders must produce or immediately promote into
  `CanonicalPathPlan`.
- Late sampling happens through `canonical_adapters` and the geometry kernel.
- `canonical_plan_to_segment_payload()` is the neutral sampled-segment export used for
  diagnostics/debug output.
- `canonical_plan_to_legacy_strokes()` is retained only for preview/export payloads.
- `/api/draw/plan` remains intentionally disabled and the raw DrawPlan contract is no longer
  accepted.
- Legacy helpers such as `pen_strokes_to_canonical_plan()` and
  `canonical_plan_from_pen_strokes()` are compatibility-only and must not be reintroduced
  into text/SVG/image production paths.

## Diagnostics

`canonical_plan_diagnostics()` exposes a `legacy_contract` section so the compatibility
boundary is observable at runtime:

- canonical plan remains the internal truth
- runtime transport is primitive-path-plan only
- preview/runtime DrawPlan diagnostic export counts are visible
- stroke export counts are visible
- raw draw-plan ingestion remains disabled
