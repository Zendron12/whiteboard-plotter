# Sketch Centerline Pipeline

Sketch Centerline Mode converts black/white or high-contrast sketch images into
board-space `DrawingPathPlan` strokes. It is an internal image-pipeline module
with a preview-only backend endpoint; it is not wired into robot drawing yet.

## Purpose

This mode is for line art where the robot should draw along the center of each
ink stroke. It avoids contour-only tracing because contour extraction follows
both sides of a thick line and can turn one intended pen stroke into a filled
outline. Skeletonization reduces the foreground mask to a one-pixel centerline
before path tracing.

## Pipeline Stages

1. Decode PNG/JPG input from bytes or a path.
2. Convert to grayscale and normalize contrast.
3. Threshold with Otsu thresholding.
4. Detect foreground polarity from the image border and support both
   black-on-white and white-on-black sketches.
5. Remove tiny connected components as noise.
6. Skeletonize the cleaned foreground mask with `skimage.morphology.skeletonize`,
   falling back to `cv2.ximgproc.thinning` when available.
7. Trace the skeleton graph into ordered strokes.
8. Simplify strokes lightly while preserving endpoints.
9. Scale and center the result into board coordinates while preserving aspect
   ratio.
10. Emit a `DrawingPathPlan` with metrics and processing metadata.

If no skeletonization backend is available, the pipeline raises `RuntimeError`.
It does not fall back to the unthinned binary mask.

## Preview Endpoint

Endpoint:

```text
POST /api/sketch-centerline/preview
```

Request format:

```text
multipart/form-data
```

Fields:

- `file`: required PNG/JPG upload.
- `margin_m`: optional board margin in meters.
- `max_image_dim`: optional maximum processing image dimension.
- `min_component_area_px`: optional tiny-noise component threshold.
- `min_stroke_length_px`: optional minimum traced stroke length.
- `simplify_epsilon_px`: optional light simplification tolerance.

Example:

```bash
curl -s -X POST http://127.0.0.1:8080/api/sketch-centerline/preview \
  -F "file=@sketch.png" \
  -F "margin_m=0.05" | python3 -m json.tool
```

Response fields:

- `ok`
- `mode`
- `stroke_count`
- `point_count`
- `canonical_command_count`
- `metrics`
- `metadata`
- `bounds`
- `warnings`
- `preview_svg`
- `preview.strokes`
- `preview.max_points`
- `preview.returned_point_count`
- `preview.original_point_count`
- `preview.truncated`

The SVG preview uses the board dimensions as its `viewBox` and preserves the
board coordinate frame: origin at top-left, x right, y down.

## Existing UI Preview

The existing File upload section exposes this mode as a preview-only option.
Choose a PNG/JPG in the normal file input, optionally adjust `Sketch Margin (m)`
and `Sketch Noise Area (px)`, then click `Preview as Sketch Centerline`.

The UI calls `/api/sketch-centerline/preview`, displays the returned
`preview_svg`, and draws the returned board-space preview strokes on the board
canvas. It also shows stroke and point counts, canonical command count, bounds,
preview truncation status, skeleton backend, threshold information, and any
warnings.

This path does not set an upload id, does not create a commit request, and does
not publish to the robot. `Commit File` is disabled while the Sketch Centerline
preview is active. Drawing/commit integration remains a later task; the normal
`Upload + Preview`, `Re-Preview`, and `Commit File` flow continues to use the
existing upload/vector pipeline.

## Runtime Status

The endpoint validates that the produced `DrawingPathPlan` can convert through
`drawing_path_plan_to_canonical(plan)`, but it does not publish a
`PrimitivePathPlan`, does not call Webots, and does not command the robot.
Runtime drawing integration is intentionally deferred until preview behavior is
stable and endpoint-level safety checks are added.

## Limitations

- This first version is tuned for clean sketches, not photos or shaded images.
- Complex junctions are traced with a simple graph traversal and may split
  dense drawings into more strokes than ideal.
- It does not optimize stroke ordering beyond preserving traced order.
- It does not implement SVG parsing, photo outlines, hatching, voice commands,
  or numbered-library runtime behavior.
