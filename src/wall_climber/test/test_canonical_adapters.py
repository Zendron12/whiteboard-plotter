from __future__ import annotations

from wall_climber.canonical_adapters import (
    SamplingPolicy,
    canonical_plan_debug_payload,
    canonical_plan_diagnostics,
    canonical_plan_to_draw_strokes,
    canonical_plan_to_segment_payload,
    canonical_plan_to_legacy_strokes,
    canonical_plan_to_primitive_path_plan,
    pen_strokes_to_canonical_plan,
    sampled_paths_from_canonical_plan,
)
from wall_climber.canonical_path import (
    ArcSegment,
    CanonicalPathPlan,
    CubicBezier,
    LineSegment,
    PenDown,
    PenUp,
    QuadraticBezier,
    TravelMove,
)


def test_pen_strokes_to_canonical_plan_inserts_travel_and_pen_state() -> None:
    plan = pen_strokes_to_canonical_plan(
        (
            ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
            ((2.0, 1.0), (2.0, 2.0)),
        ),
        theta_ref=0.0,
        pen_offset_x_m=0.0,
        pen_offset_y_m=0.0,
    )

    assert [type(command) for command in plan.commands] == [
        PenDown,
        LineSegment,
        LineSegment,
        PenUp,
        TravelMove,
        PenDown,
        LineSegment,
        PenUp,
    ]


def test_canonical_plan_exports_preview_and_legacy_shapes() -> None:
    plan = pen_strokes_to_canonical_plan(
        (
            ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
            ((2.0, 1.0), (2.0, 2.0)),
        ),
        theta_ref=0.0,
        pen_offset_x_m=0.0,
        pen_offset_y_m=0.0,
    )

    draw_strokes = canonical_plan_to_draw_strokes(plan)
    assert draw_strokes == (
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
        ((2.0, 1.0), (2.0, 2.0)),
    )

    segment_payload = canonical_plan_to_segment_payload(plan)
    assert segment_payload['frame'] == 'board'
    assert segment_payload['theta_ref'] == 0.0
    assert segment_payload['segments'] == [
        {
            'draw': True,
            'type': 'polyline',
            'points': [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        },
        {
            'draw': False,
            'type': 'line',
            'points': [[1.0, 1.0], [2.0, 1.0]],
        },
        {
            'draw': True,
            'type': 'line',
            'points': [[2.0, 1.0], [2.0, 2.0]],
        },
    ]

    legacy_strokes = canonical_plan_to_legacy_strokes(plan)
    assert legacy_strokes == {
        'frame': 'board',
        'strokes': [
            {
                'draw': True,
                'type': 'polyline',
                'points': [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            },
            {
                'draw': True,
                'type': 'line',
                'points': [[2.0, 1.0], [2.0, 2.0]],
            },
        ],
    }


def test_curve_sampling_preserves_endpoints() -> None:
    plan = CanonicalPathPlan(
        frame='board',
        theta_ref=0.0,
        commands=(
            PenDown(),
            QuadraticBezier(
                start=(0.0, 0.0),
                control=(0.5, 1.0),
                end=(1.0, 0.0),
            ),
            CubicBezier(
                start=(1.0, 0.0),
                control1=(1.2, 0.6),
                control2=(1.8, -0.6),
                end=(2.0, 0.0),
            ),
            PenUp(),
        ),
    )

    draw_strokes = canonical_plan_to_draw_strokes(plan, curve_tolerance_m=0.1)
    assert len(draw_strokes) == 1
    assert draw_strokes[0][0] == (0.0, 0.0)
    assert draw_strokes[0][-1] == (2.0, 0.0)
    assert len(draw_strokes[0]) > 4


def test_sampled_paths_preserve_draw_and_travel_partition() -> None:
    plan = CanonicalPathPlan(
        frame='board',
        theta_ref=0.0,
        commands=(
            PenDown(),
            QuadraticBezier(
                start=(0.0, 0.0),
                control=(0.5, 1.0),
                end=(1.0, 0.0),
            ),
            PenUp(),
            TravelMove(start=(1.0, 0.0), end=(1.5, 0.0)),
            PenDown(),
            LineSegment(start=(1.5, 0.0), end=(2.0, 0.0)),
            PenUp(),
        ),
    )

    sampled_paths = sampled_paths_from_canonical_plan(plan, curve_tolerance_m=0.1)
    assert [sampled.draw for sampled in sampled_paths] == [True, False, True]
    assert sampled_paths[0].points[0] == (0.0, 0.0)
    assert sampled_paths[-1].points[-1] == (2.0, 0.0)


def test_sampling_policy_separates_preview_and_runtime_point_budgets() -> None:
    plan = CanonicalPathPlan(
        frame='board',
        theta_ref=0.0,
        commands=(
            PenDown(),
            LineSegment(start=(0.0, 0.0), end=(0.03, 0.0)),
            PenUp(),
        ),
    )

    preview_paths = sampled_paths_from_canonical_plan(
        plan,
        sampling_policy=SamplingPolicy(
            curve_tolerance_m=0.01,
            draw_step_m=0.03,
            travel_step_m=0.03,
            label='preview',
        ),
    )
    runtime_paths = sampled_paths_from_canonical_plan(
        plan,
        sampling_policy=SamplingPolicy(
            curve_tolerance_m=0.01,
            draw_step_m=0.01,
            travel_step_m=0.01,
            label='runtime',
        ),
    )

    assert len(preview_paths) == 1
    assert len(runtime_paths) == 1
    assert len(preview_paths[0].points) == 2
    assert len(runtime_paths[0].points) == 4


def test_canonical_plan_diagnostics_reports_sampling_and_parity() -> None:
    plan = CanonicalPathPlan(
        frame='board',
        theta_ref=0.0,
        commands=(
            PenDown(),
            LineSegment(start=(0.0, 0.0), end=(0.03, 0.0)),
            PenUp(),
            TravelMove(start=(0.03, 0.0), end=(0.05, 0.0)),
        ),
    )

    diagnostics = canonical_plan_diagnostics(
        plan,
        preview_sampling_policy=SamplingPolicy(
            curve_tolerance_m=0.01,
            draw_step_m=0.03,
            travel_step_m=0.03,
            label='preview',
        ),
        runtime_sampling_policy=SamplingPolicy(
            curve_tolerance_m=0.01,
            draw_step_m=0.01,
            travel_step_m=0.01,
            label='runtime',
        ),
    )

    assert diagnostics['canonical_plan']['command_count'] == 4
    assert diagnostics['canonical_plan']['primitive_counts']['LineSegment'] == 1
    assert diagnostics['canonical_plan']['primitive_counts']['TravelMove'] == 1
    assert diagnostics['preview_sampling']['policy']['label'] == 'preview'
    assert diagnostics['runtime_sampling']['policy']['label'] == 'runtime'
    assert diagnostics['runtime_sampling']['total_point_count'] > diagnostics['preview_sampling']['total_point_count']
    assert diagnostics['point_budget']['delta_points'] > 0
    assert diagnostics['parity']['status'] == 'ok'
    assert diagnostics['legacy_contract']['internal_truth'] == 'canonical_path_plan'
    assert diagnostics['legacy_contract']['draw_plan_role'] == 'diagnostic_export_only'
    assert diagnostics['legacy_contract']['stroke_payload_role'] == 'preview_payload_only'
    assert diagnostics['legacy_contract']['runtime_transport'] == 'primitive_path_plan_only'
    assert diagnostics['legacy_contract']['raw_draw_plan_endpoint_enabled'] is False
    assert diagnostics['legacy_contract']['runtime_export']['segment_count'] == 2
    assert diagnostics['legacy_contract']['runtime_export']['stroke_count'] == 1


def test_canonical_plan_exports_primitive_transport_descriptor() -> None:
    plan = CanonicalPathPlan(
        frame='board',
        theta_ref=0.25,
        commands=(
            PenDown(),
            LineSegment(start=(0.0, 0.0), end=(1.0, 0.0)),
            ArcSegment(center=(1.0, 1.0), radius=1.0, start_angle_rad=-1.57079632679, sweep_angle_rad=1.57079632679),
            CubicBezier(
                start=(1.0, 0.0),
                control1=(1.2, 0.5),
                control2=(1.8, -0.5),
                end=(2.0, 0.0),
            ),
            PenUp(),
        ),
    )

    descriptor = canonical_plan_to_primitive_path_plan(plan)

    assert descriptor['frame'] == 'board'
    assert descriptor['theta_ref'] == 0.25
    assert [primitive['type'] for primitive in descriptor['primitives']] == [
        'PEN_DOWN',
        'LINE_SEGMENT',
        'ARC_SEGMENT',
        'CUBIC_BEZIER',
        'PEN_UP',
    ]
    assert descriptor['primitives'][1]['pen_down'] is True
    assert descriptor['primitives'][2]['center'] == {'x': 1.0, 'y': 1.0}
    assert descriptor['primitives'][2]['clockwise'] is False
    assert descriptor['primitives'][3]['control2'] == {'x': 1.8, 'y': -0.5}


def test_canonical_plan_debug_payload_exposes_commands_and_bounds() -> None:
    plan = CanonicalPathPlan(
        frame='board',
        theta_ref=0.0,
        commands=(
            PenDown(),
            ArcSegment(center=(0.0, 0.0), radius=1.0, start_angle_rad=0.0, sweep_angle_rad=1.57079632679),
            PenUp(),
        ),
    )

    payload = canonical_plan_debug_payload(
        plan,
        sampling_policy=SamplingPolicy(curve_tolerance_m=0.05, label='debug'),
    )

    assert payload['command_count'] == 3
    assert payload['primitive_counts']['ArcSegment'] == 1
    assert payload['commands'][1]['type'] == 'arc'
    assert payload['commands'][1]['length_m'] > 1.5
    assert payload['sampled_bounds']['width'] > 0.9


def test_canonical_plan_debug_payload_merges_command_fit_metadata() -> None:
    plan = CanonicalPathPlan(
        frame='board',
        theta_ref=0.0,
        commands=(
            PenDown(),
            LineSegment(start=(0.0, 0.0), end=(1.0, 0.0)),
            PenUp(),
        ),
    )

    payload = canonical_plan_debug_payload(
        plan,
        command_metadata=(
            None,
            {
                'fit_source': 'span_arc_fit',
                'fit_error_px': 0.42,
                'span_id': 7,
            },
            None,
        ),
    )

    assert payload['commands'][1]['fit_source'] == 'span_arc_fit'
    assert payload['commands'][1]['fit_error_px'] == 0.42
    assert payload['commands'][1]['span_id'] == 7
