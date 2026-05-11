from __future__ import annotations

from pathlib import Path

from wall_climber.image_pipeline.svg_vector import svg_text_to_canonical_plan
from wall_climber.image_pipeline.vectorizers import autotrace_centerline
from wall_climber.image_pipeline.vectorizers import potrace_backend
from wall_climber.image_pipeline.vectorizers import vtracer_backend
from wall_climber.optimizers import vpype_optimizer


def _index_html() -> str:
    return (Path(__file__).resolve().parents[1] / 'web' / 'index.html').read_text(encoding='utf-8')


def _detail_select_html() -> str:
    html = _index_html()
    start = html.index('<select id="sketch-optimization-preset">')
    end = html.index('</select>', start)
    return html[start:end]


def test_primary_detail_level_only_exposes_line_art_and_custom() -> None:
    select = _detail_select_html()

    assert 'Line Art / Detailed' in select
    assert 'value="detail" selected' in select
    assert 'value="custom"' in select
    assert 'value="balanced"' not in select
    assert 'value="fast"' not in select
    assert 'Clean / Fast' not in select


def test_advanced_numeric_fields_hidden_until_custom_or_debug() -> None:
    html = _index_html()

    assert '<details id="advanced-sketch-settings" class="advanced-settings" hidden>' in html
    assert 'const showAdvanced = DEBUG_MODE || dom.sketchOptimizationPreset.value === \'custom\';' in html
    for field_id in (
        'sketch-min-component',
        'sketch-min-stroke',
        'sketch-skeleton-prune',
        'sketch-merge-gap',
        'sketch-simplify-epsilon',
        'sketch-curve-tolerance',
        'sketch-tiny-candidate-mm',
    ):
        assert field_id in html


def test_optional_engine_compare_controls_are_debug_only() -> None:
    html = _index_html()

    assert '<div class="field debug-only">' in html
    assert 'id="vectorization-engine"' in html
    assert 'id="compare-methods-btn" class="quiet-btn debug-only"' in html
    assert 'id="compare-results" class="compare-results debug-only" hidden' in html
    assert 'body.debug-mode .debug-only[hidden]' in html
    assert "['internal_centerline']" in html


def test_line_art_default_uses_adaptive_extraction_and_moderate_sensitivity() -> None:
    html = _index_html()

    assert '<option value="adaptive" selected>Adaptive Line Art (Recommended)</option>' in html
    select_start = html.index('<select id="sketch-extraction-method">')
    primary_select = html[select_start : html.index('</select>', select_start)]
    assert 'value="hysteresis_ink"' not in primary_select
    assert "option.textContent = 'Hysteresis Ink (Experimental)';" in html
    assert 'id="sketch-line-sensitivity" type="number" min="0" max="0.95" step="0.05" value="0.35"' in html
    assert 'id="sketch-skeleton-prune" type="number" min="0" max="100" step="1" value="4"' in html
    assert 'settings.skeleton_prune_px' in html


def test_color_line_art_method_replaces_opencv_main_flow() -> None:
    html = _index_html()

    assert 'Color Line-Art Method' in html
    assert '<option value="auto_outline" selected>Auto Outline (Recommended)</option>' in html
    assert '<option value="photo_diagram_edges">Photo / Diagram Edges</option>' in html
    assert '<option value="simple_cartoon">Simple Cartoon / Diagram Outline</option>' in html
    assert 'Color-to-Sketch Method' not in html
    assert 'OpenCV Pencil / Edge Sketch' not in html
    assert 'Colored images use local outline conversion before Adaptive Centerline.' in html
    assert "option.textContent = 'OpenCV Edge Diagnostic';" in html
    assert 'Photo / Diagram Edges → Adaptive Centerline' in html


def test_converted_line_art_preview_exists_before_executable_preview() -> None:
    html = _index_html()
    converted_index = html.index('id="converted-lineart-box"')
    executable_index = html.index('id="sketch-preview-box"')

    assert converted_index < executable_index
    assert 'converted_lineart_preview' in html
    assert 'Local Outline → Adaptive Centerline' in html


def test_optional_vectorizers_return_graceful_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(autotrace_centerline.shutil, 'which', lambda _name: None)
    monkeypatch.setattr(potrace_backend.shutil, 'which', lambda _name: None)
    monkeypatch.setattr(vtracer_backend.shutil, 'which', lambda _name: None)

    for result in (
        autotrace_centerline.vectorize_autotrace_centerline(b'not-an-image'),
        potrace_backend.vectorize_potrace_bw(b'not-an-image'),
        vtracer_backend.vectorize_vtracer_svg(b'not-an-image'),
    ):
        assert result.available is False
        assert result.canonical_plan is None
        assert result.warnings


def test_svg_import_path_parses_external_vectorizer_output() -> None:
    plan = svg_text_to_canonical_plan(
        '<svg viewBox="0 0 100 50"><path d="M 10 10 L 90 40"/></svg>',
        metadata={'vectorization_engine': 'mock'},
    )

    assert plan.commands


def test_vpype_missing_does_not_break_basic_optimizer_flow(monkeypatch) -> None:
    monkeypatch.setattr(vpype_optimizer.shutil, 'which', lambda _name: None)
    plan = svg_text_to_canonical_plan('<svg viewBox="0 0 10 10"><path d="M1 1 L9 9"/></svg>')

    optimized, metadata = vpype_optimizer.optimize_with_vpype(plan)

    assert optimized is None
    assert metadata['available'] is False
    assert metadata['warnings']
