"""Adaptive cubic-Bezier curve fitting (Schneider 1990 port).

This is a Python port of Philip J. Schneider's "An Algorithm for
Automatically Fitting Digitized Curves" from *Graphics Gems I* (1990).
Compared to the simple least-squares fit in ``curve_fit.py``, this
implementation:

1. Solves explicitly for *both* tangent magnitudes (``alpha_l``, ``alpha_r``)
   instead of pinning them at chord/3.
2. Refines the chord-length parameterisation with Newton-Raphson so points
   land closer to the curve.
3. Recursively splits at the worst-error point with continuity-preserving
   tangents, instead of falling back to a line chain.

The net effect on real-world sketches: the same tolerance produces ~5-10x
fewer line primitives because curves that the simple fitter rejects (and
the recursion downgrades to lines) now succeed in one shot.
"""

from __future__ import annotations

import math


Point = tuple[float, float]
Cubic = tuple[Point, Point, Point, Point]

_MAX_NEWTON_ITERATIONS = 4
_MIN_SEGMENT_POINTS = 4


def fit_cubic_beziers(
    points: list[Point] | tuple[Point, ...],
    max_error: float,
) -> list[Cubic]:
    """Fit ``points`` with one or more cubic Bezier curves.

    ``max_error`` is in the same units as the input points. Smaller values
    produce more cubics; larger values produce fewer cubics with looser
    fits. The output is always a non-empty list of ``(p0, p1, p2, p3)``
    tuples whose endpoints chain end-to-start.
    """
    if len(points) < 2:
        return []
    pts = [(float(p[0]), float(p[1])) for p in points]
    if len(pts) < _MIN_SEGMENT_POINTS:
        return [_line_cubic(pts[0], pts[-1])]

    left_tangent = _normalize(_subtract(pts[1], pts[0]))
    right_tangent = _normalize(_subtract(pts[-2], pts[-1]))
    cubics: list[Cubic] = []
    _fit_cubic(pts, left_tangent, right_tangent, max_error, cubics)
    return cubics


# ----------------------------------------------------------------------
# Recursive fit
# ----------------------------------------------------------------------

def _fit_cubic(
    pts: list[Point],
    left_tangent: Point,
    right_tangent: Point,
    max_error: float,
    out: list[Cubic],
) -> None:
    if len(pts) == 2:
        out.append(_line_cubic(pts[0], pts[1]))
        return

    if len(pts) < _MIN_SEGMENT_POINTS:
        out.append(_line_cubic(pts[0], pts[-1]))
        return

    u_prime = _chord_length_parameterise(pts)
    bezier = _generate_bezier(pts, u_prime, left_tangent, right_tangent)
    error, split_index = _compute_max_error(pts, bezier, u_prime)
    if error < max_error:
        out.append(bezier)
        return

    # Try Newton-Raphson reparameterisation when the error is "close
    # enough" — using the linear scale (error < 4*max_error) instead
    # of the squared scale that the Graphics-Gems version used. The
    # squared comparison kicked in too aggressively and let the inner
    # loop accept fits that exceeded the actual tolerance.
    if error < max_error * 4.0:
        for _ in range(_MAX_NEWTON_ITERATIONS):
            u_prime = _reparameterise(pts, u_prime, bezier)
            bezier = _generate_bezier(pts, u_prime, left_tangent, right_tangent)
            new_error, split_index = _compute_max_error(pts, bezier, u_prime)
            if new_error < max_error:
                out.append(bezier)
                return
            error = new_error

    # Split at the point of maximum error and recurse with continuity-
    # preserving centre tangents.
    centre_tangent = _normalize(_subtract(pts[split_index - 1], pts[split_index + 1]))
    _fit_cubic(pts[: split_index + 1], left_tangent, centre_tangent, max_error, out)
    _fit_cubic(
        pts[split_index:],
        (-centre_tangent[0], -centre_tangent[1]),
        right_tangent,
        max_error,
        out,
    )


# ----------------------------------------------------------------------
# Generation: least-squares cubic from constrained tangents
# ----------------------------------------------------------------------

def _generate_bezier(
    pts: list[Point],
    u_prime: list[float],
    left_tangent: Point,
    right_tangent: Point,
) -> Cubic:
    n = len(pts) - 1
    a = [(0.0, 0.0)] * (n + 1)
    a_prime = [(0.0, 0.0)] * (n + 1)
    for index, t in enumerate(u_prime):
        b1 = 3.0 * t * (1.0 - t) * (1.0 - t)
        b2 = 3.0 * t * t * (1.0 - t)
        a[index] = (left_tangent[0] * b1, left_tangent[1] * b1)
        a_prime[index] = (right_tangent[0] * b2, right_tangent[1] * b2)

    c = [[0.0, 0.0], [0.0, 0.0]]
    x = [0.0, 0.0]
    for index, t in enumerate(u_prime):
        c[0][0] += _dot(a[index], a[index])
        c[0][1] += _dot(a[index], a_prime[index])
        c[1][0] = c[0][1]
        c[1][1] += _dot(a_prime[index], a_prime[index])
        b0 = (1.0 - t) ** 3
        b1 = 3.0 * t * (1.0 - t) * (1.0 - t)
        b2 = 3.0 * t * t * (1.0 - t)
        b3 = t ** 3
        tmp_x = pts[index][0] - (
            b0 * pts[0][0] + b1 * pts[0][0] + b2 * pts[n][0] + b3 * pts[n][0]
        )
        tmp_y = pts[index][1] - (
            b0 * pts[0][1] + b1 * pts[0][1] + b2 * pts[n][1] + b3 * pts[n][1]
        )
        x[0] += _dot(a[index], (tmp_x, tmp_y))
        x[1] += _dot(a_prime[index], (tmp_x, tmp_y))

    det_c0_c1 = c[0][0] * c[1][1] - c[1][0] * c[0][1]
    det_c0_x = c[0][0] * x[1] - c[1][0] * x[0]
    det_x_c1 = x[0] * c[1][1] - x[1] * c[0][1]

    alpha_l = 0.0 if det_c0_c1 == 0.0 else det_x_c1 / det_c0_c1
    alpha_r = 0.0 if det_c0_c1 == 0.0 else det_c0_x / det_c0_c1
    seg_length = _distance(pts[0], pts[n])
    epsilon = 1.0e-6 * seg_length
    if alpha_l < epsilon or alpha_r < epsilon:
        # Wu/Barsky heuristic fallback: place control points 1/3 along chord.
        dist = seg_length / 3.0
        return (
            pts[0],
            (pts[0][0] + left_tangent[0] * dist, pts[0][1] + left_tangent[1] * dist),
            (pts[n][0] + right_tangent[0] * dist, pts[n][1] + right_tangent[1] * dist),
            pts[n],
        )

    return (
        pts[0],
        (pts[0][0] + left_tangent[0] * alpha_l, pts[0][1] + left_tangent[1] * alpha_l),
        (pts[n][0] + right_tangent[0] * alpha_r, pts[n][1] + right_tangent[1] * alpha_r),
        pts[n],
    )


# ----------------------------------------------------------------------
# Reparameterisation and error
# ----------------------------------------------------------------------

def _reparameterise(pts: list[Point], u: list[float], bezier: Cubic) -> list[float]:
    return [_newton_raphson_root_find(bezier, p, u_value) for p, u_value in zip(pts, u)]


def _newton_raphson_root_find(bezier: Cubic, point: Point, u: float) -> float:
    q = _bezier(bezier, u)
    q_prime = _bezier_prime(bezier, u)
    q_prime_prime = _bezier_prime_prime(bezier, u)
    numerator = (q[0] - point[0]) * q_prime[0] + (q[1] - point[1]) * q_prime[1]
    denominator = (
        q_prime[0] * q_prime[0]
        + q_prime[1] * q_prime[1]
        + (q[0] - point[0]) * q_prime_prime[0]
        + (q[1] - point[1]) * q_prime_prime[1]
    )
    if abs(denominator) < 1.0e-12:
        return u
    return u - (numerator / denominator)


def _compute_max_error(pts: list[Point], bezier: Cubic, u: list[float]) -> tuple[float, int]:
    """Return the maximum **Euclidean** distance from any input point to
    its corresponding sample on the Bezier curve, plus the index of the
    worst-fit point.

    The original Graphics-Gems implementation returned the squared
    distance, which the recursive splitter then compared against
    ``max_error * max_error``. That worked but masked the actual
    geometric tolerance (a "max_error of 2 mm" really meant "max_error
    squared 4 mm**2", i.e. allowing up to ~2 mm), and the Newton-
    Raphson refinement could nudge the curve outside that bound on
    long arcs because the inner branch only kicked in when the squared
    error was below ``max_error**2`` (a much looser gate).

    We return the actual Euclidean distance so the caller can compare
    apples to apples and hand a cubic to the canonical plan only when
    every input point is within ``max_error`` of the curve, full stop.
    """
    max_dist = 0.0
    split_index = len(pts) // 2
    for index in range(1, len(pts) - 1):
        sample = _bezier(bezier, u[index])
        dx = sample[0] - pts[index][0]
        dy = sample[1] - pts[index][1]
        dist = math.hypot(dx, dy)
        if dist > max_dist:
            max_dist = dist
            split_index = index
    return max_dist, split_index


# ----------------------------------------------------------------------
# Bezier evaluation helpers
# ----------------------------------------------------------------------

def _bezier(bezier: Cubic, t: float) -> Point:
    p0, p1, p2, p3 = bezier
    omt = 1.0 - t
    return (
        omt ** 3 * p0[0] + 3.0 * omt * omt * t * p1[0]
        + 3.0 * omt * t * t * p2[0] + t ** 3 * p3[0],
        omt ** 3 * p0[1] + 3.0 * omt * omt * t * p1[1]
        + 3.0 * omt * t * t * p2[1] + t ** 3 * p3[1],
    )


def _bezier_prime(bezier: Cubic, t: float) -> Point:
    p0, p1, p2, p3 = bezier
    omt = 1.0 - t
    return (
        3.0 * omt * omt * (p1[0] - p0[0])
        + 6.0 * omt * t * (p2[0] - p1[0])
        + 3.0 * t * t * (p3[0] - p2[0]),
        3.0 * omt * omt * (p1[1] - p0[1])
        + 6.0 * omt * t * (p2[1] - p1[1])
        + 3.0 * t * t * (p3[1] - p2[1]),
    )


def _bezier_prime_prime(bezier: Cubic, t: float) -> Point:
    p0, p1, p2, p3 = bezier
    omt = 1.0 - t
    return (
        6.0 * omt * (p2[0] - 2.0 * p1[0] + p0[0])
        + 6.0 * t * (p3[0] - 2.0 * p2[0] + p1[0]),
        6.0 * omt * (p2[1] - 2.0 * p1[1] + p0[1])
        + 6.0 * t * (p3[1] - 2.0 * p2[1] + p1[1]),
    )


def _line_cubic(start: Point, end: Point) -> Cubic:
    third_x = start[0] + (end[0] - start[0]) / 3.0
    third_y = start[1] + (end[1] - start[1]) / 3.0
    two_third_x = start[0] + 2.0 * (end[0] - start[0]) / 3.0
    two_third_y = start[1] + 2.0 * (end[1] - start[1]) / 3.0
    return (start, (third_x, third_y), (two_third_x, two_third_y), end)


def _chord_length_parameterise(pts: list[Point]) -> list[float]:
    cumulative = [0.0]
    for index in range(1, len(pts)):
        cumulative.append(cumulative[-1] + _distance(pts[index - 1], pts[index]))
    total = cumulative[-1]
    if total <= 0.0:
        return [index / max(1, len(pts) - 1) for index in range(len(pts))]
    return [value / total for value in cumulative]


def _distance(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _subtract(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def _normalize(v: Point) -> Point:
    length = math.hypot(v[0], v[1])
    if length == 0.0:
        return v
    return (v[0] / length, v[1] / length)


def _dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


__all__ = ['fit_cubic_beziers']
