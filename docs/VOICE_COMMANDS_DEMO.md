# Voice Commands Demo

This is a small browser-based demo layer for the existing web UI. It uses the
browser Web Speech API when available and keeps all robot execution on the
existing backend-owned ROS path.

## Browser Support

Use Chromium, Chrome, or Edge for microphone support. Vivaldi may not support
the Web Speech API reliably. If speech recognition is unavailable or fails, use
the `Voice command text fallback` field; it runs the same command parser as the
microphone path.

## Supported Commands

- `text mode`, `write mode`, `typing mode`
  - Switches to the Text tab and requests existing text runtime mode.
- Dictated text while in Text mode
  - Appends the phrase to the existing text textarea.
  - Starts a 5-second auto-write countdown.
  - If the user clicks, focuses, types, clears, or manually writes text during
    the countdown, auto-write is canceled.
  - If not canceled, it calls the existing `Write Text` flow.
- `draw mode`, `file mode`, `image mode`
  - Switches to the File tab and requests existing draw runtime mode.
- `draw picture number N`
- `draw image number N`
- `draw pic number N`
- `draw photo number N`
  - Calls `POST /api/draw-library/draw` with only `{ "id": N }`.
  - The backend loads the numbered image and draws it through Sketch Centerline
    planning and the existing `CanonicalPathPlan -> PrimitivePathPlan -> ROS`
    executor path.
- `stop`, `stop drawing`, `cancel`
  - The UI recognizes the command, but no real runtime cancel endpoint exists
    yet. The UI reports this limitation instead of pretending to cancel.

## Numbered Image Library

Place demo images under:

```text
assets/draw_library/
```

Supported simple filenames:

```text
1.png
1.jpg
1.jpeg
2.png
2.jpg
2.jpeg
```

An optional installed `manifest.json` may map IDs to files:

```json
{
  "entries": [
    { "id": 1, "file": "1.png", "name": "Demo sketch" }
  ]
}
```

Only PNG/JPG/JPEG files are supported for this urgent demo endpoint. The assets
are installed into the ROS package share during `colcon build`, so rebuild after
adding or changing demo images.

## Demo Script

1. Build and launch normally:

   ```bash
   source /opt/ros/humble/setup.bash
   colcon build --packages-select wall_climber --cmake-args -DBUILD_TESTING=ON
   source install/setup.bash
   ros2 launch wall_climber my_robot.launch.py
   ```

2. Open `http://127.0.0.1:8080`.
3. Click `Start Voice`.
4. Say `text mode`.
5. Say a short phrase.
6. Wait 5 seconds and confirm the existing `Write Text` behavior triggers.
7. Say `draw mode`.
8. Say `draw picture number 1`.
9. Confirm the backend starts drawing `assets/draw_library/1.png` or equivalent.
10. Say `stop` and confirm the UI reports that runtime cancel is not implemented.

## Safety Model

The browser never sends strokes, SVG paths, G-code, or board-canvas data for
voice drawing. Numbered image drawing is backend-owned: the backend loads the
installed image, runs the Sketch Centerline pipeline with safe-fit placement and
Smooth Curves, optimizes stroke order, validates transport, and publishes the
existing ROS primitive plan.

