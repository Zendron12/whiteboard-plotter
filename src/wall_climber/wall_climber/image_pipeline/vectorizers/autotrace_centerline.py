from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from wall_climber.image_pipeline.svg_vector import svg_text_to_canonical_plan
from wall_climber.image_pipeline.vectorizers.base import VectorizationEngineResult


ENGINE_NAME = 'autotrace_centerline'


def is_available() -> bool:
    return shutil.which('autotrace') is not None


def vectorize_autotrace_centerline(
    image_bytes: bytes,
    *,
    timeout_sec: float = 20.0,
) -> VectorizationEngineResult:
    binary = shutil.which('autotrace')
    if binary is None:
        return VectorizationEngineResult(
            engine_name=ENGINE_NAME,
            available=False,
            warnings=('AutoTrace is not installed or is not on PATH.',),
        )
    if not image_bytes:
        return VectorizationEngineResult(
            engine_name=ENGINE_NAME,
            available=True,
            warnings=('Input image payload is empty.',),
        )

    with tempfile.TemporaryDirectory(prefix='wall-climber-autotrace-') as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / 'input.png'
        output_path = temp_path / 'output.svg'
        input_path.write_bytes(image_bytes)
        command = [
            binary,
            '--centerline',
            '--output-format',
            'svg',
            '--output-file',
            str(output_path),
            str(input_path),
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
            return VectorizationEngineResult(
                engine_name=ENGINE_NAME,
                available=True,
                warnings=(f'AutoTrace failed: {exc}',),
            )
        if completed.returncode != 0 or not output_path.exists():
            message = completed.stderr.strip() or completed.stdout.strip() or f'exit {completed.returncode}'
            return VectorizationEngineResult(
                engine_name=ENGINE_NAME,
                available=True,
                warnings=(f'AutoTrace centerline failed: {message}',),
            )
        svg_output = output_path.read_text(encoding='utf-8')
    canonical_plan = svg_text_to_canonical_plan(
        svg_output,
        metadata={'vectorization_engine': ENGINE_NAME},
    )
    return VectorizationEngineResult(
        engine_name=ENGINE_NAME,
        available=True,
        canonical_plan=canonical_plan,
        svg_output=svg_output,
        preview_svg=svg_output,
        metrics={'external_command': 'autotrace --centerline'},
    )
