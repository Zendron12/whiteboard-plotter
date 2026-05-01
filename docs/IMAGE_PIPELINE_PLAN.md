# Image Pipeline Plan

This document defines the future best-practice image pipeline for the Webots cable-driven whiteboard robot. It is a design plan only. The current task does not replace the existing production image pipeline.

An offline adapter design exists for converting tested `DrawingPathPlan` output into the current `CanonicalPathPlan` model. Sketch Centerline Mode now exists as an internal pipeline module with a preview-only backend endpoint. It is not enabled for robot drawing. Future PNG, SVG, photo, voice, and numbered-library paths should produce `DrawingPathPlan` first, with runtime connection added later after mode-specific tests exist.

## A. Sketch Centerline Mode

Purpose:

- Convert black/white sketches and clean line art into pen strokes that follow the center of the drawn lines.

Implemented internal foundation:

- Convert to grayscale.
- Use Otsu threshold.
- Remove noise with morphological cleanup.
- Skeletonize the binary mask.
- Convert the skeleton into a graph.
- Trace graph paths into strokes.
- Simplify strokes while preserving endpoints and corners.
- Emit a neutral `DrawingPathPlan`.
- Preview through `POST /api/sketch-centerline/preview` without publishing to the robot.

## B. SVG/Vector Mode

Purpose:

- Convert SVG input or vectorizer output into robot paths without unnecessary raster processing.

Future stages:

- Accept SVG input or vectorizer output.
- Parse SVG paths using a robust library in the future.
- Sample lines, arcs, and Bezier curves into points.
- Normalize and scale source coordinates.
- Convert to board coordinates.
- Emit a neutral `DrawingPathPlan`.

## C. Photo Outline Mode

Purpose:

- Convert colored images and photo-like inputs into drawable outline paths.

Future stages:

- Decode colored image input.
- Denoise while preserving edges.
- Enhance contrast.
- Extract edges or outlines.
- Clean and connect usable contours.
- Generate drawable paths.
- Emit a neutral `DrawingPathPlan`.

## D. Optional Hatching Mode

Purpose:

- Represent grayscale tone with hatch strokes when outlines alone are not enough.

Future stages:

- Convert input to grayscale.
- Estimate tone or darkness per region.
- Map tone to hatch density, spacing, and direction.
- Generate hatching strokes.
- Optionally combine with outline strokes.
- Emit a neutral `DrawingPathPlan`.

## E. Path Optimization

Future optimizer responsibilities:

- Remove tiny strokes.
- Simplify points.
- Order strokes to reduce pen-up travel.
- Reverse stroke direction when beneficial.
- Merge compatible nearby strokes only when it preserves drawing intent.
- Estimate drawing time from draw length, pen-up travel, pen lifts, and configured execution rates.

## F. Evaluation Metrics

Future metrics should include:

- number of strokes
- number of points before simplification
- number of points after simplification
- total drawing length
- pen-up travel length
- pen lift count
- estimated drawing time
- optional similarity metrics

Metrics should be available to the backend and dashboard without coupling image processing to Webots internals.
