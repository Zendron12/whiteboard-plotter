from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from wall_climber.canonical_adapters import canonical_plan_to_draw_strokes
from wall_climber.canonical_path import CanonicalPathPlan
from wall_climber.image_pipeline.svg_vector import svg_text_to_canonical_plan


def is_available() -> bool:
    return shutil.which('vpype') is not None


def _canonical_plan_to_svg(plan: CanonicalPathPlan) -> str:
    strokes = canonical_plan_to_draw_strokes(plan)
    paths = []
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        first = stroke[0]
        commands = [f'M {float(first[0]):.6f} {float(first[1]):.6f}']
        for point in stroke[1:]:
            commands.append(f'L {float(point[0]):.6f} {float(point[1]):.6f}')
        paths.append(f'<path d="{" ".join(commands)}" fill="none" stroke="black" stroke-width="0.001"/>')
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 4 3">'
        + ''.join(paths)
        + '</svg>'
    )


def optimize_with_vpype(
    plan: CanonicalPathPlan,
    *,
    timeout_sec: float = 20.0,
) -> tuple[CanonicalPathPlan | None, dict[str, Any]]:
    binary = shutil.which('vpype')
    if binary is None:
        return None, {
            'available': False,
            'warnings': ('vpype is not installed or is not on PATH.',),
        }
    with tempfile.TemporaryDirectory(prefix='wall-climber-vpype-') as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / 'input.svg'
        output_path = temp_path / 'output.svg'
        input_path.write_text(_canonical_plan_to_svg(plan), encoding='utf-8')
        command = [
            binary,
            'read',
            str(input_path),
            'linemerge',
            'linesimplify',
            'linesort',
            'write',
            str(output_path),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=float(timeout_sec),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return None, {'available': True, 'warnings': (f'vpype failed: {exc}',)}
        if completed.returncode != 0 or not output_path.exists():
            message = completed.stderr.strip() or completed.stdout.strip() or f'exit {completed.returncode}'
            return None, {'available': True, 'warnings': (f'vpype failed: {message}',)}
        optimized_svg = output_path.read_text(encoding='utf-8')
    optimized_plan = svg_text_to_canonical_plan(
        optimized_svg,
        metadata={'optimizer': 'vpype'},
    )
    return optimized_plan, {
        'available': True,
        'warnings': (),
        'optimizer': 'vpype',
    }
