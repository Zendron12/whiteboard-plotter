from __future__ import annotations

from pathlib import Path

from wall_climber.canonical_adapters import canonical_plan_to_draw_strokes
from wall_climber.canonical_builders import (
    draw_strokes_to_canonical_plan,
    text_glyph_outlines_to_canonical_plan,
)
from wall_climber.canonical_path import LineSegment, PenDown, PenUp, TravelMove
from wall_climber.vector_pipeline import (
    TextGlyphOutline,
    VectorBounds,
    VectorPlacement,
    _effective_text_advance_em,
    cleanup_canonical_plan,
    default_image_placement,
    place_canonical_plan_on_board,
    vectorize_text_grouped,
)


def test_draw_strokes_to_canonical_plan_builds_direct_commands() -> None:
    plan = draw_strokes_to_canonical_plan(
        (
            ((0.0, 0.0), (1.0, 0.0)),
            ((2.0, 0.0), (3.0, 0.0)),
        ),
        theta_ref=0.0,
    )

    assert [type(command) for command in plan.commands] == [
        PenDown,
        LineSegment,
        PenUp,
        TravelMove,
        PenDown,
        LineSegment,
        PenUp,
    ]


def test_text_glyph_outlines_to_canonical_plan_uses_glyph_order_directly() -> None:
    glyphs = (
        TextGlyphOutline(
            line_index=0,
            word_index=0,
            text='A',
            strokes=(((0.0, 0.0), (1.0, 0.0)),),
            bbox=VectorBounds(0.0, 1.0, 0.0, 0.0),
            advance=1.0,
            source='test',
        ),
        TextGlyphOutline(
            line_index=0,
            word_index=1,
            text='B',
            strokes=(((2.0, 0.0), (3.0, 0.0)),),
            bbox=VectorBounds(2.0, 3.0, 0.0, 0.0),
            advance=1.0,
            source='test',
        ),
    )

    plan = text_glyph_outlines_to_canonical_plan(glyphs, theta_ref=0.0)
    assert [type(command) for command in plan.commands] == [
        PenDown,
        LineSegment,
        PenUp,
        TravelMove,
        PenDown,
        LineSegment,
        PenUp,
    ]


def test_place_canonical_plan_on_board_places_plan_without_legacy_rebuild() -> None:
    plan = draw_strokes_to_canonical_plan(
        (((0.0, 0.0), (2.0, 0.0)),),
        theta_ref=0.0,
    )

    placed_plan, placement_result = place_canonical_plan_on_board(
        plan,
        writable_bounds={'x_min': 0.0, 'x_max': 10.0, 'y_min': 0.0, 'y_max': 10.0},
        placement=VectorPlacement(x=5.0, y=5.0, scale=1.0),
        fit_padding=1.0,
    )

    placed_strokes = canonical_plan_to_draw_strokes(placed_plan)
    assert placed_strokes == (((0.0, 5.0), (10.0, 5.0)),)
    assert placement_result.bounds.x_min == 0.0
    assert placement_result.bounds.x_max == 10.0
    assert placement_result.outside_points == 0


def test_cleanup_canonical_plan_returns_cleaned_canonical_output() -> None:
    plan = draw_strokes_to_canonical_plan(
        (((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)),),
        theta_ref=0.0,
    )

    cleaned = cleanup_canonical_plan(
        plan,
        simplify_tolerance_m=0.01,
    )

    assert canonical_plan_to_draw_strokes(cleaned) == (((0.0, 0.0), (2.0, 0.0)),)


def test_default_image_placement_uses_safe_workspace_center_and_scale() -> None:
    placement = default_image_placement(
        {'x_min': 0.0, 'x_max': 10.0, 'y_min': 0.0, 'y_max': 8.0},
        safe_bounds={'x_min': 1.0, 'x_max': 9.0, 'y_min': 1.0, 'y_max': 7.0},
    )

    assert placement.x == 5.0
    assert placement.y == 4.0
    assert 0.73 < placement.scale < 0.76


def test_vectorize_text_grouped_accepts_dejavu_sans_font_source() -> None:
    glyphs = vectorize_text_grouped(
        'Hello',
        font_source='dejavu_sans',
    )

    assert len(glyphs) > 0
    assert any(glyph.strokes for glyph in glyphs)
    assert all(glyph.source in {'bundled_installed', 'bundled_source'} for glyph in glyphs)


def test_dejavu_sans_uses_tighter_letter_spacing_than_default_text_spacing() -> None:
    default_advance = _effective_text_advance_em(
        'A',
        base_advance=1.0,
        letter_spacing_em=0.25,
        word_spacing_em=0.4,
        uppercase_advance_scale=1.0,
        font_source='relief_singleline',
    )
    dejavu_advance = _effective_text_advance_em(
        'A',
        base_advance=1.0,
        letter_spacing_em=0.25,
        word_spacing_em=0.4,
        uppercase_advance_scale=1.0,
        font_source='dejavu_sans',
    )

    assert dejavu_advance < default_advance


def test_web_server_main_path_no_longer_uses_legacy_pen_stroke_helper() -> None:
    source = (
        Path(__file__).resolve().parents[1] / 'wall_climber' / 'web_server.py'
    ).read_text(encoding='utf-8')
    assert 'canonical_plan_from_pen_strokes(' not in source


def test_web_server_image_path_uses_dual_pipeline_builder() -> None:
    source = (
        Path(__file__).resolve().parents[1] / 'wall_climber' / 'web_server.py'
    ).read_text(encoding='utf-8')
    assert 'vectorize_image_to_canonical_plan(' in source
    assert 'optimize_canonical_plan(' in source
    assert 'from wall_climber.ingestion.image import vectorize_image_to_canonical_plan' in source


def test_web_server_uses_split_ingestion_and_canonical_ops_modules() -> None:
    source = (
        Path(__file__).resolve().parents[1] / 'wall_climber' / 'web_server.py'
    ).read_text(encoding='utf-8')
    assert 'from wall_climber.ingestion.text import (' in source
    assert 'from wall_climber.ingestion.svg import vectorize_svg' in source
    assert 'from wall_climber.canonical_ops import (' in source


def test_web_server_primitive_transport_publish_is_wired() -> None:
    source = (
        Path(__file__).resolve().parents[1] / 'wall_climber' / 'web_server.py'
    ).read_text(encoding='utf-8')
    assert 'publish_execution_plan(' in source
    assert 'PRIMITIVE_PATH_PLAN_TOPIC' in source
    assert '_draw_plan_pub' not in source
    assert 'DRAW_PLAN_TOPIC' not in source
    assert 'DrawPlan' not in source


def test_web_server_text_commit_omits_oversized_legacy_normalized_plan() -> None:
    source = (
        Path(__file__).resolve().parents[1] / 'wall_climber' / 'web_server.py'
    ).read_text(encoding='utf-8')
    assert 'def _normalized_text_plan_or_summary(' in source
    assert 'except HTTPException as exc:' in source
    assert 'if exc.status_code != 413:' in source
    assert "'normalized_plan_omitted': True" in source
    assert "'transport': 'primitive_path_plan'" in source
    assert 'normalized_plan = _normalized_text_plan_or_summary(' in source
    assert "'normalized_plan': normalized_plan" in source


def test_raw_draw_plan_endpoint_remains_explicitly_disabled() -> None:
    source = (
        Path(__file__).resolve().parents[1] / 'wall_climber' / 'web_server.py'
    ).read_text(encoding='utf-8')
    assert "detail='raw /api/draw/plan has been removed; use /api/vector/svg/commit or /api/vector/image/commit'" in source


def test_web_server_exposes_debug_endpoints() -> None:
    source = (
        Path(__file__).resolve().parents[1] / 'wall_climber' / 'web_server.py'
    ).read_text(encoding='utf-8')
    assert "@app.get('/api/debug/last-plan')" in source
    assert "@app.get('/api/debug/last-execution')" in source
    assert "@app.get('/api/debug/last-curve-fit')" in source
    assert 'EXECUTION_DIAGNOSTICS_TOPIC' in source


def test_web_server_exposes_unified_file_upload_routes() -> None:
    source = (
        Path(__file__).resolve().parents[1] / 'wall_climber' / 'web_server.py'
    ).read_text(encoding='utf-8')
    assert "@app.post('/api/draw/file')" in source
    assert "@app.get('/api/vector/file/status')" in source
    assert "@app.post('/api/vector/file/preview')" in source
    assert "@app.post('/api/vector/file/commit')" in source
    assert 'classify_uploaded_vector_file' in source
    assert "'stored_only': True" in source or '"stored_only": True' in source


def test_web_server_text_font_source_validation_includes_dejavu_sans() -> None:
    source = (
        Path(__file__).resolve().parents[1] / 'wall_climber' / 'web_server.py'
    ).read_text(encoding='utf-8')
    assert '"dejavu_sans"' in source
    assert 'font_source must be one of ["relief_singleline", "hershey_sans_1", "dejavu_sans"]' in source
    assert "'glyph_height_m': glyph_scale_m" in source


def test_index_exposes_curve_overlay_controls() -> None:
    source = (
        Path(__file__).resolve().parents[1] / 'web' / 'index.html'
    ).read_text(encoding='utf-8')
    assert 'overlay-raw-toggle' in source
    assert 'overlay-curves-toggle' in source
    assert 'overlay-fallback-toggle' in source
    assert 'overlay-color-toggle' in source
    assert 'debug-curve-fit-panel' in source
    assert 'tool-file-panel' in source
    assert '/api/draw/file' in source
    assert '/api/vector/file/preview' in source
    assert '/api/vector/file/commit' in source
    assert 'image/svg+xml,.svg' in source
    assert 'trace_mode:' in source
    assert 'chunk_count:' in source
    assert '<option value="dejavu_sans">DejaVu Sans (outline)</option>' in source


def test_index_disables_text_board_preview_and_supports_preview_dragging() -> None:
    source = (
        Path(__file__).resolve().parents[1] / 'web' / 'index.html'
    ).read_text(encoding='utf-8')
    assert "applyVectorPreview('text', payload, { origin: 'text', boardVisible: false })" in source
    assert 'function beginPreviewInteraction(event)' in source
    assert 'function movePreviewInteraction(event)' in source
    assert 'function drawPreviewPlacementControls(layout)' in source
    assert "clearVectorPreview(false);" in source
    assert "new Set(['relief_singleline', 'hershey_sans_1', 'dejavu_sans'])" in source
    assert 'function applyRasterPreview(' in source
    assert '/api/vector/file/status?upload_id=' in source
    assert 'const sentenceGap = Math.max(0.045, 0.38 * lineAdvance);' in source
    assert 'const wrappedSentenceGap = Math.max(0.020, 0.18 * lineAdvance);' in source


def test_executor_source_declares_chunked_execution_controls() -> None:
    source = (
        Path(__file__).resolve().parents[2] / 'wall_climber_draw_body' / 'src' / 'cable_draw_executor.cpp'
    ).read_text(encoding='utf-8')
    assert 'declare_parameter("chunk_max_paths", 48);' in source
    assert 'declare_parameter("chunk_max_samples", 2400);' in source
    assert 'declare_parameter("text_draw_resample_step_m", 0.0038);' in source
    assert 'declare_parameter("text_travel_resample_step_m", 0.012);' in source
    assert 'active_mode_ == "text"' in source
    assert 'build_next_schedule_chunk' in source
    assert '"chunk_count":' in source or '\\"chunk_count\\":' in source


def test_legacy_compatibility_rfc_exists() -> None:
    source = (
        Path(__file__).resolve().parents[1] / 'docs' / 'legacy-compatibility.md'
    ).read_text(encoding='utf-8')
    assert 'CanonicalPathPlan' in source
    assert 'raw draw-plan ingestion remains disabled' in source
    assert 'canonical_plan_to_segment_payload()' in source


def test_primitive_transport_rfc_exists() -> None:
    source = (
        Path(__file__).resolve().parents[1] / 'docs' / 'primitive-transport.md'
    ).read_text(encoding='utf-8')
    assert 'PrimitivePathPlan' in source
    assert '/wall_climber/primitive_path_plan' in source
