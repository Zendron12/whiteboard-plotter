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
3. Threshold with Otsu thresholding plus optional Line Sensitivity.
4. Detect foreground polarity from the image border and support both
   black-on-white and white-on-black sketches.
5. Remove tiny connected components as noise.
6. Skeletonize the cleaned foreground mask with `skimage.morphology.skeletonize`,
   falling back to `cv2.ximgproc.thinning` when available.
7. Trace the skeleton graph into ordered strokes.
8. Apply the selected optimization preset.
9. Simplify strokes lightly while preserving endpoints.
10. Optionally merge nearby compatible stroke endpoints when the selected
   preset enables it.
11. Scale and center the result into board coordinates while preserving aspect
   ratio.
12. Emit a point-stroke `DrawingPathPlan` with metrics and processing metadata.
13. For preview only, optionally fit smooth canonical curve primitives from the
   point strokes.

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
- `line_sensitivity`: optional `0.0..0.95` faint-line sensitivity.
- `merge_gap_px`: optional endpoint merge gap in processed pixels.
- `merge_max_angle_deg`: optional maximum endpoint direction angle for merging.
- `optimization_preset`: optional `raw`, `detail`, `balanced`, `fast`, or
  `custom`.
- `preview_geometry_mode`: optional `smooth_curves` or `polyline`.
- `curve_tolerance_px`: optional curve fitting tolerance in processed pixels.
- `curve_tolerance_m`: optional curve fitting tolerance in board meters; when
  supplied it takes precedence over `curve_tolerance_px`.
- `scale_percent`: optional scale applied after fit-to-board.
- `center_x_m`: optional board-space drawing center x coordinate.
- `center_y_m`: optional board-space drawing center y coordinate.

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
In `smooth_curves` mode, `preview_svg` is generated from real SVG path commands
including `Q` and `C` for existing canonical `QuadraticBezier` and
`CubicBezier` primitives where fitting succeeds. In `polyline` mode, the SVG
uses polylines and is intended as a debug view of the raw point-stroke output.
The SVG is the full preview output. `preview.strokes` is a capped point preview
for the board canvas and may be truncated for performance.

## Existing UI Preview

The existing File upload section exposes this mode as a preview-only option.
Choose a PNG/JPG in the normal file input, optionally adjust `Sketch Margin (m)`,
`Optimization Preset`, `Preview Geometry`, `Sketch Max Image Dim`, Curve
Tolerance, `Sketch Noise Area (px)`, Line Sensitivity, Min Stroke Length,
Stroke Merge Gap, Simplify Epsilon, Sketch Scale, and Sketch Center X/Y, then
click `Preview as Sketch Centerline`.

The UI calls `/api/sketch-centerline/preview`, displays the returned
`preview_svg`, and draws the returned board-space preview strokes on the board
canvas. It also shows stroke and point counts, canonical command count, bounds,
preview truncation status, raw/final stroke counts, merge counts, selected
optimization preset, preview geometry mode, curve/line primitive counts, timing
stages, effective merge/simplification settings, skeleton backend, threshold
information, placement metadata, and any warnings.

The File tab separates the two preview surfaces:

- `SVG Preview`: full Smooth Curves or Polyline Debug SVG output. This can be
  opened in a new tab or downloaded for close inspection.
- `Board Canvas`: sampled Polyline Canvas Preview from `preview.strokes`. If
  the response is capped, the UI warns that the board canvas is truncated and
  shows the full result in the SVG preview instead.

## Tuning Notes

Faint gray lines can disappear when Otsu thresholding separates dark foreground
from a bright background but leaves light pencil-like marks on the background
side of the threshold. Line Sensitivity keeps the Otsu polarity decision but
moves the effective threshold to include more faint foreground: higher values
include lighter gray strokes for dark-on-light images and dimmer light strokes
for light-on-dark images.

Fragmented strokes happen because skeleton tracing intentionally splits the
graph at endpoints and junctions. Aggressive endpoint merging can make the robot
lift the pen less often, but it can also damage detailed line-art by bridging
nearby unrelated hair, clothing, face, or body outlines. Shape extraction and
robot travel optimization are separate concerns: the default `detail` preset
prioritizes visual fidelity even when that leaves a high stroke count.

Stroke Merge Gap reconnects nearby compatible endpoints only when the selected
optimization preset enables merging. It can reduce segmented output, but high
values can connect unrelated nearby details. The merge pass also rejects nearly
perpendicular endpoint pairs and avoids ambiguous dense regions where one
endpoint has multiple plausible merge candidates.

Earlier preview fragmentation could be amplified visually by the frontend
treating preview-only sketch output like an invalid commit preview and drawing
it dashed. Sketch Centerline strokes are still preview-only, but the board
canvas now draws them as solid preview strokes while keeping `Commit File`
disabled.

The polygonal straight-edge look came from representing every traced centerline
as adjacent point-to-point `LineSegment` commands. The current smooth preview
path keeps the raw `DrawingPathPlan` unchanged, then fits board-space point
strokes into canonical `QuadraticBezier` and `CubicBezier` commands for the SVG
preview. This is not just styling: use `Preview Geometry = Polyline Debug` to
compare the raw point output against `Smooth Curves` and verify whether the SVG
contains real curve primitives.

Smooth Curves can still contain many line primitives. Short, sharp, noisy, or
junction-heavy strokes are left as `LineSegment`s to avoid oversmoothing facial,
hair, or clothing details. A high `line_primitive_count` relative to curve
counts means the fitter preserved shape fidelity rather than forcing unsafe
curves.

Scale and Center controls are placement-only. The default fits the sketch inside
the board margin and auto-centers it. `Sketch Scale (%)` scales that fitted
drawing around its center. Empty Center X/Y keeps auto-center; supplied values
place the drawing center at that board coordinate. Placements outside the board
are rejected instead of clipped.

For detailed anime or line-art sketches, start with Line Sensitivity `0.25` to
`0.45`, lower Noise Area to `1..4` to keep small components, keep Min Stroke
Length around `1..3`, keep Stroke Merge Gap at `0..1` in the `detail` preset,
use Simplify Epsilon `0..0.25`, choose `Smooth Curves`, set Curve Tolerance
around `0.75..1.5 px`, and use Sketch Max Image Dim `800..1200` depending on
speed/detail needs. Avoid high sensitivity or merge gap values because they can
add noise or join unrelated hair, clothing, or body outlines.

## Optimization Presets

- `raw`: minimal post-processing after skeleton tracing. This is useful for
  debugging and maximum fidelity, but it can produce many short strokes.
- `detail`: the default for anime and line-art. It disables endpoint merging,
  keeps short strokes, and uses very light simplification. This favors visual
  fidelity over drawing speed.
- `balanced`: enables conservative endpoint merging and moderate simplification
  for general sketches.
- `fast`: uses stronger cleanup, merging, and simplification. It reduces stroke
  count but can lose small details or smooth sharp features.
- `custom`: uses the submitted Min Stroke Length, Stroke Merge Gap, Merge Angle,
  and Simplify Epsilon values exactly.

Line Sensitivity is independent from these presets. It controls which pixels
survive thresholding; the optimization preset controls how traced strokes are
cleaned, merged, and simplified after skeletonization.

## Performance Diagnostics

The preview response includes `metadata.timing` for decode, resize, normalize,
threshold, cleanup, skeletonization, tracing, simplification, merge, curve fit,
scale, and total preview time. It also reports the slowest stage. This is CPU
processing; large detailed sketches can be slow because skeletonization and
graph tracing scale with foreground pixels, while endpoint merge and curve fit
scale with stroke/point count.

`Sketch Max Image Dim` is the first speed control: lower it to `800` or `600`
for faster preview, or raise toward `1200` for more detail. Merge is disabled
for the `raw` and default `detail` presets, and merge-enabled presets are capped
by time, pass count, and accepted merge count so one preview does not run for
many minutes. Curve fitting is also time-capped; if it exceeds its budget, the
remaining spans stay as line segments and a warning is returned.

This path does not set an upload id and does not reuse the normal file commit
request. `Commit File` is disabled while the Sketch Centerline preview is
active. The normal `Upload + Preview`, `Re-Preview`, and `Commit File` flow
continues to use the existing upload/vector pipeline.

## Runtime Status

`POST /api/sketch-centerline/preview` is still preview-first. On success it now
returns a `preview_id` and stores the backend-owned result in a temporary
process-local cache. The cache is intentionally short-lived: entries expire
after about 10 minutes, only a small number of previews are retained, and the
cache is cleared when the backend process restarts.

`POST /api/sketch-centerline/draw` accepts only that `preview_id`. It does not
accept browser-provided SVG, preview points, board-canvas data, or path
commands. The backend looks up the cached canonical plan, validates it against
the existing execution transport, converts it to `PrimitivePathPlan`, and
publishes through the existing ROS/Webots executor path.

The draw source matches the preview geometry mode:

- `Smooth Curves`: publishes the cached smooth canonical plan, including
  `QuadraticBezier`/`CubicBezier` primitives where curve fitting succeeded.
- `Polyline Debug`: publishes the cached line-only canonical plan from the
  DrawingPathPlan adapter.

If the cached plan is too large for the guarded transport limits, the draw
endpoint returns a clear `413` response with command/primitive counts instead of
crashing. If the preview expired or the runtime is not ready, the endpoint
returns a clear error and does not publish.

G-code is not used for runtime drawing. The project runtime remains
`CanonicalPathPlan` / `PrimitivePathPlan` / ROS executor. G-code may be a future
optional export format, but it is not the main execution path.

## Limitations

- This first version is tuned for clean sketches, not photos or shaded images.
- Complex junctions are traced with a simple graph traversal and may split
  dense drawings into more strokes than ideal.
- It does not optimize stroke ordering beyond preserving traced order.
- It does not implement SVG parsing, photo outlines, hatching, voice commands,
  or numbered-library runtime behavior.
