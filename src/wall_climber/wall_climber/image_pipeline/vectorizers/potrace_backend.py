from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2  # type: ignore
import numpy

from wall_climber.image_pipeline.svg_vector import svg_text_to_canonical_plan
from wall_climber.image_pipeline.vectorizers.base import VectorizationEngineResult


ENGINE_NAME = 'potrace_bw'


def is_available() -> bool:
    return shutil.which('potrace') is not None


def vectorize_potrace_bw(
    image_bytes: bytes,
    *,
    timeout_sec: float = 20.0,
) -> VectorizationEngineResult:
    binary = shutil.which('potrace')
    if binary is None:
        return VectorizationEngineResult(
            engine_name=ENGINE_NAME,
            available=False,
            warnings=('Potrace is not installed or is not on PATH.',),
        )
    image_array = numpy.frombuffer(image_bytes, dtype=numpy.uint8)
    gray = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return VectorizationEngineResult(
            engine_name=ENGINE_NAME,
            available=True,
            warnings=('Potrace input could not be decoded as an image.',),
        )
    _threshold, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    with tempfile.TemporaryDirectory(prefix='wall-climber-potrace-') as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / 'input.pbm'
        output_path = temp_path / 'output.svg'
        ok = cv2.imwrite(str(input_path), binary_image)
        if not ok:
            return VectorizationEngineResult(
                engine_name=ENGINE_NAME,
                available=True,
                warnings=('Failed to prepare PBM input for Potrace.',),
            )
        try:
            completed = subprocess.run(
                [binary, '--svg', '-o', str(output_path), str(input_path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=float(timeout_sec),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return VectorizationEngineResult(
                engine_name=ENGINE_NAME,
                available=True,
                warnings=(f'Potrace failed: {exc}',),
            )
        if completed.returncode != 0 or not output_path.exists():
            message = completed.stderr.strip() or completed.stdout.strip() or f'exit {completed.returncode}'
            return VectorizationEngineResult(
                engine_name=ENGINE_NAME,
                available=True,
                warnings=(f'Potrace failed: {message}',),
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
        metrics={'external_command': 'potrace --svg'},
    )
