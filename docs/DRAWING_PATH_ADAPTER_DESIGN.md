# DrawingPathPlan Adapter Design

## Purpose

`DrawingPathPlan` is the future-facing drawing output for mode-based image, SVG, text, numbered-library, and voice-driven pipelines. It is intentionally independent from Webots and ROS. Its strokes are ordered board-space points in meters, plus source metadata and pipeline metrics.

The adapter in `wall_climber.image_pipeline.adapters` converts this neutral drawing model into the existing `CanonicalPathPlan` model without enabling it in production runtime.

## Existing Canonical Model

`CanonicalPathPlan` is defined in `wall_climber/canonical_path.py` and currently expects:

- `frame`: a non-empty string, with runtime paths using `board`
- `theta_ref`: finite float
- `commands`: non-empty tuple of canonical commands

Canonical commands include:

- `PenUp`
- `PenDown`
- `TravelMove`
- `LineSegment`
- `ArcSegment`
- `QuadraticBezier`
- `CubicBezier`

The existing runtime exports canonical commands to `PrimitivePathPlan` through `canonical_adapters.py`, then the C++ executor converts primitives into cable setpoints. This adapter does not publish ROS messages and does not call the executor.

## Conversion Rules

The first adapter version is intentionally narrow:

- `DrawingPathPlan.frame` must be `board`.
- `DrawingPathPlan.strokes` are treated as already placed board-space coordinates in meters.
- Stroke order is preserved exactly.
- Each drawable stroke maps to `PenDown`, one or more `LineSegment` commands, and `PenUp`.
- Non-contiguous drawable strokes get a `TravelMove` between the previous stroke end and next stroke start.
- Duplicate adjacent points inside a stroke are skipped.
- A stroke that becomes degenerate after duplicate removal raises `ValueError`.

This creates the same basic command shape already used by the existing draw-stroke and text canonical builders, but keeps the new image-pipeline model separate from production code paths.

## Pen Behavior

`Stroke(pen_down=True)` means the stroke is drawable.

`Stroke(pen_down=False)` is rejected for now with a clear `ValueError`. Explicit travel strokes may be useful later, but the first adapter version keeps travel generation deterministic by deriving travel moves from gaps between drawable strokes.

## Metadata And Metrics

The current `CanonicalPathPlan` only stores `frame`, `theta_ref`, and commands. It has no metadata or metrics container.

Adapter handling:

- `metadata["theta_ref"]` is used when present and finite.
- Missing, invalid, or non-finite `theta_ref` falls back to `0.0`.
- `source_id`, `mode`, pipeline metrics, stroke labels, and other metadata are intentionally ignored by the canonical output.

Future diagnostics may preserve richer metadata beside the canonical plan, but that is not part of this task.

## Not Supported Yet

This adapter does not:

- run image processing
- parse SVG
- vectorize text
- parse voice commands
- load numbered image library assets
- optimize paths
- publish ROS messages
- call Webots
- call backend endpoints
- convert to `PrimitivePathPlan`
- implement cable physics
- implement four-cable kinematics

## Runtime Boundary

The adapter is not wired into runtime endpoints because existing text/SVG/image builders already feed the production `CanonicalPathPlan` flow. Runtime connection should happen later only after mode-specific image pipeline tests exist and preview/runtime parity is verified.

Keeping this adapter offline prevents changes to:

- Webots worlds
- Webots plugins
- xacro files
- launch files
- ROS message definitions
- C++ executor behavior
- FastAPI production endpoints

## Two-Cable And Future Four-Cable Neutrality

The adapter outputs board-space canonical drawing commands. It does not know how the robot turns those commands into motion.

That separation preserves both:

- the current two-cable Webots plugin/executor flow
- a future plugin-based four-cable kinematic execution mode

Future two-cable and four-cable execution adapters should consume the same drawing/path flow after canonical validation, without forcing image, SVG, text, voice, or library pipelines to know about cable layout.
