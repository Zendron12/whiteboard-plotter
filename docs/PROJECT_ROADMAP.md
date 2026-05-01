# Project Roadmap

## Project Identity

This project is a Webots-based cable-driven whiteboard drawing robot for large boards. It is not a rail-based CNC machine or Cartesian XY plotter.

The existing Webots plugin-based simulation is intentional and must be preserved. The project should not replace the current plugin approach with full cable physics or cable dynamics unless that becomes an explicit, measured future requirement.

## Mechanical Direction

The original simulation used a Y-shape / two-cable supervisor. The integrated mechanical direction is now a four-cable kinematic supervisor with:

- top-left anchor
- top-right anchor
- bottom-left anchor
- bottom-right anchor

This remains plugin-based and kinematic. Do not replace it with a CNC rail model, a separate duplicate Webots world, a parallel runtime, or full cable physics.

## Target Software Architecture

The long-term graduation-project architecture should use a unified drawing representation between all input sources and the Webots execution adapter.

Planned layers:

- `DrawingPathPlan`: neutral drawing path model for future image, SVG, voice text, and numbered library sources.
- Hybrid image-to-path pipeline: selectable source-aware modes for sketches, SVG/vector input, photos, and optional hatching.
- Voice command layer: parses user commands into text plans, library image references, clear/pause/resume commands, or drawing requests.
- Numbered image library: maps spoken or UI-selected image numbers to stored assets and default pipeline modes.
- Path optimizer: removes tiny strokes, simplifies paths, orders strokes, reverses stroke direction when useful, and estimates drawing time.
- Metrics/evaluation dashboard: exposes stroke counts, point counts, travel length, drawing length, pen lifts, estimated time, and optional similarity metrics.
- Webots execution adapter: converts approved plans into the existing `CanonicalPathPlan` and `PrimitivePathPlan` execution path after tests prove parity.

The first offline adapter design from `DrawingPathPlan` to `CanonicalPathPlan` now exists as a tested foundation. It is not enabled in runtime endpoints; future PNG, SVG, photo, voice, and numbered-library pipelines should emit `DrawingPathPlan` first, then connect to runtime only after pipeline tests and parity checks exist.

## Hybrid Image Pipeline Modes

### Sketch Centerline Mode

Input:

- Black/white sketches or clean line art.

Future flow:

- grayscale
- adaptive threshold or Otsu threshold
- cleanup
- skeletonization
- graph/path tracing
- stroke simplification
- output to `DrawingPathPlan`

### SVG/Vector Mode

Input:

- SVG files or vectorizer output.

Future flow:

- parse SVG paths with a robust library
- sample curves into board-space points
- preserve curve metadata where useful
- output to `DrawingPathPlan`

### Photo Outline Mode

Input:

- Colored images, illustrations, or photo-like input.

Future flow:

- denoise
- contrast enhancement
- outline extraction
- cleanup
- path generation
- output to `DrawingPathPlan`

### Optional Hatching Mode

Input:

- Grayscale or photo input where tone should be represented by strokes.

Future flow:

- map darkness to hatch density
- generate hatching strokes
- combine with optional outlines
- output to `DrawingPathPlan`

## Voice Command Layer

Future voice command examples:

- `write hello world`
- `draw pic number 1`
- `draw picture number 2`
- `clear board`
- `pause`
- `resume`

Voice parsing should live in Python and should produce command objects or drawing plans. It should not bypass validation, path optimization, or the existing Webots execution safety checks.

## Numbered Image Library

The numbered image library should map stable numeric IDs to local assets and default pipeline modes. The example manifest in `assets/draw_library/manifest.example.json` defines the intended shape without adding binary images.

Future behavior:

- `draw pic number 1` selects manifest entry `1`.
- The entry chooses the default pipeline mode unless the user overrides it.
- The selected asset is converted into the same plan model as uploads.

## Python And C++ Responsibilities

Python should own:

- image processing
- vectorization
- SVG parsing
- path planning
- voice command parsing
- web UI/backend
- metrics and evaluation

C++ should own:

- performance-critical geometry
- execution loops
- Webots/ROS integration where already used by the project

Do not move image processing to C++ unless there is a measured performance reason.

## Near-Term Milestones

1. Keep the Webots plugin-based cable simulation stable as it moves from the original two-cable setup to the integrated four-cable kinematic supervisor model.
2. Add `DrawingPathPlan` as a neutral type-only foundation.
3. Keep the tested `DrawingPathPlan` to `CanonicalPathPlan` adapter offline until runtime integration is explicitly planned.
4. Add a manifest-backed numbered image library.
5. Add a voice parser that maps commands to existing backend actions or future plan builders.
6. Add a metrics dashboard section backed by existing diagnostics and future pipeline metrics.
7. Extend four-cable diagnostics and evaluation only after the kinematic supervisor behavior is covered by tests.

## Four-Cable Kinematic Supervisor Note

The existing Webots supervisor plugin is being upgraded in place to a four-cable kinematic model. This remains plugin-based and pose-driven; it is not a separate world, duplicate runtime, visual-only cable decoration, or full rope-physics simulation.
