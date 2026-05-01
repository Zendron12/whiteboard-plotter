#include "wall_climber_draw_body/geometry_sampling.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <variant>

#include "wall_climber_draw_body/geometry_eval.hpp"

namespace wall_climber::geometry {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr int kMaxSamplingDepth = 18;

void append_unique_point(std::vector<Point2> * points, const Point2 & point) {
  if (!points->empty() && approximately_equal(points->back(), point)) {
    return;
  }
  points->push_back(point);
}

double point_line_distance(const Point2 & point, const Point2 & start, const Point2 & end) {
  const double length = distance_xy(start, end);
  if (length <= kGeometryEpsilon) {
    return distance_xy(point, start);
  }
  const double dx = end.x - start.x;
  const double dy = end.y - start.y;
  const double t =
    std::max(
    0.0,
    std::min(1.0, ((point.x - start.x) * dx + (point.y - start.y) * dy) / (dx * dx + dy * dy)));
  return distance_xy(point, Point2{start.x + dx * t, start.y + dy * t});
}

double quadratic_flatness(const QuadraticBezier & segment) {
  return point_line_distance(segment.control, segment.start, segment.end);
}

double cubic_flatness(const CubicBezier & segment) {
  return std::max(
    point_line_distance(segment.control1, segment.start, segment.end),
    point_line_distance(segment.control2, segment.start, segment.end));
}

double heading_delta_rad(const Point2 & first, const Point2 & second) {
  const double first_heading = heading_rad(first);
  const double second_heading = heading_rad(second);
  return std::abs(std::atan2(
    std::sin(second_heading - first_heading),
    std::cos(second_heading - first_heading)));
}

double sanitized_heading_delta(const SamplePolicy & policy) {
  if (!std::isfinite(policy.max_heading_delta_rad) || policy.max_heading_delta_rad <= 0.0) {
    return kPi / 9.0;
  }
  return std::max(1.0e-4, policy.max_heading_delta_rad);
}

double sanitized_tolerance(const SamplePolicy & policy);

std::pair<QuadraticBezier, QuadraticBezier> split_quadratic(const QuadraticBezier & segment) {
  const Point2 p01 = lerp(segment.start, segment.control, 0.5);
  const Point2 p12 = lerp(segment.control, segment.end, 0.5);
  const Point2 p012 = lerp(p01, p12, 0.5);
  return {
    QuadraticBezier{segment.start, p01, p012},
    QuadraticBezier{p012, p12, segment.end},
  };
}

std::pair<CubicBezier, CubicBezier> split_cubic(const CubicBezier & segment) {
  const Point2 p01 = lerp(segment.start, segment.control1, 0.5);
  const Point2 p12 = lerp(segment.control1, segment.control2, 0.5);
  const Point2 p23 = lerp(segment.control2, segment.end, 0.5);
  const Point2 p012 = lerp(p01, p12, 0.5);
  const Point2 p123 = lerp(p12, p23, 0.5);
  const Point2 p0123 = lerp(p012, p123, 0.5);
  return {
    CubicBezier{segment.start, p01, p012, p0123},
    CubicBezier{p0123, p123, p23, segment.end},
  };
}

void sample_quadratic_recursive(
  const QuadraticBezier & segment,
  const SamplePolicy & policy,
  const int depth,
  std::vector<Point2> * points)
{
  const double tolerance_m = sanitized_tolerance(policy);
  const bool heading_ok =
    heading_delta_rad(tangent_vector(segment, 0.0), tangent_vector(segment, 1.0)) <=
    sanitized_heading_delta(policy);
  if (depth >= kMaxSamplingDepth || (quadratic_flatness(segment) <= tolerance_m && heading_ok)) {
    append_unique_point(points, segment.start);
    append_unique_point(points, segment.end);
    return;
  }

  const auto halves = split_quadratic(segment);
  sample_quadratic_recursive(halves.first, policy, depth + 1, points);
  sample_quadratic_recursive(halves.second, policy, depth + 1, points);
}

void sample_cubic_recursive(
  const CubicBezier & segment,
  const SamplePolicy & policy,
  const int depth,
  std::vector<Point2> * points)
{
  const double tolerance_m = sanitized_tolerance(policy);
  const bool heading_ok =
    heading_delta_rad(tangent_vector(segment, 0.0), tangent_vector(segment, 1.0)) <=
    sanitized_heading_delta(policy);
  if (depth >= kMaxSamplingDepth || (cubic_flatness(segment) <= tolerance_m && heading_ok)) {
    append_unique_point(points, segment.start);
    append_unique_point(points, segment.end);
    return;
  }

  const auto halves = split_cubic(segment);
  sample_cubic_recursive(halves.first, policy, depth + 1, points);
  sample_cubic_recursive(halves.second, policy, depth + 1, points);
}

double sanitized_tolerance(const SamplePolicy & policy) {
  return std::max(policy.curve_tolerance_m, policy.min_step_m);
}

double sanitized_linear_step(const double step_m, const SamplePolicy & policy) {
  if (!std::isfinite(step_m) || step_m <= 0.0) {
    return 0.0;
  }
  return std::max(step_m, policy.min_step_m);
}

std::vector<Point2> sample_linear_points(
  const Point2 & start,
  const Point2 & end,
  const double step_m)
{
  std::vector<Point2> points;
  const double segment_length = distance_xy(start, end);
  const int subdivisions = (step_m > 0.0 && segment_length > kGeometryEpsilon)
    ? std::max(1, static_cast<int>(std::ceil(segment_length / step_m)))
    : 1;
  points.reserve(static_cast<std::size_t>(subdivisions) + 1U);
  for (int index = 0; index <= subdivisions; ++index) {
    const double t = static_cast<double>(index) / static_cast<double>(subdivisions);
    append_unique_point(&points, lerp(start, end, t));
  }
  return points;
}

double polyline_length(const std::vector<Point2> & points) {
  double total = 0.0;
  if (points.size() < 2U) {
    return total;
  }
  for (std::size_t index = 1; index < points.size(); ++index) {
    total += distance_xy(points[index - 1], points[index]);
  }
  return total;
}

int arc_subdivisions(const ArcSegment & segment, const SamplePolicy & policy) {
  const double tolerance_m = sanitized_tolerance(policy);
  const double abs_sweep = std::abs(segment.sweep_angle_rad);
  const double max_heading_delta = sanitized_heading_delta(policy);
  if (segment.radius <= tolerance_m) {
    return std::max(1, static_cast<int>(std::ceil(abs_sweep / std::min(kPi / 8.0, max_heading_delta))));
  }

  const double clamped = std::clamp(1.0 - (tolerance_m / segment.radius), -1.0, 1.0);
  double max_flatness_step = 2.0 * std::acos(clamped);
  if (!std::isfinite(max_flatness_step) || max_flatness_step <= 1.0e-6) {
    max_flatness_step = kPi / 8.0;
  }
  const double max_step = std::min(max_flatness_step, max_heading_delta);
  return std::max(1, static_cast<int>(std::ceil(abs_sweep / max_step)));
}

}  // namespace

std::vector<Point2> sample_geometry(const TravelMove & segment, const SamplePolicy & policy) {
  return sample_linear_points(
    segment.start,
    segment.end,
    sanitized_linear_step(policy.travel_step_m, policy));
}

std::vector<Point2> sample_geometry(const LineSegment & segment, const SamplePolicy & policy) {
  return sample_linear_points(
    segment.start,
    segment.end,
    sanitized_linear_step(policy.draw_step_m, policy));
}

std::vector<Point2> sample_geometry(const ArcSegment & segment, const SamplePolicy & policy) {
  const int subdivisions = arc_subdivisions(segment, policy);
  std::vector<Point2> points;
  points.reserve(static_cast<std::size_t>(subdivisions) + 1U);
  for (int index = 0; index <= subdivisions; ++index) {
    const double t = static_cast<double>(index) / static_cast<double>(subdivisions);
    append_unique_point(&points, evaluate_point(segment, t));
  }
  return points;
}

std::vector<Point2> sample_geometry(const QuadraticBezier & segment, const SamplePolicy & policy) {
  std::vector<Point2> points;
  points.reserve(16U);
  sample_quadratic_recursive(segment, policy, 0, &points);
  return points;
}

std::vector<Point2> sample_geometry(const CubicBezier & segment, const SamplePolicy & policy) {
  std::vector<Point2> points;
  points.reserve(32U);
  sample_cubic_recursive(segment, policy, 0, &points);
  return points;
}

std::vector<Point2> sample_polyline(
  const std::vector<Point2> & points,
  const bool draw,
  const SamplePolicy & policy)
{
  if (points.size() < 2U) {
    return {};
  }

  std::vector<Point2> sampled;
  sampled.reserve(points.size());
  const double step_m = sanitized_linear_step(
    draw ? policy.draw_step_m : policy.travel_step_m,
    policy);
  for (std::size_t index = 1; index < points.size(); ++index) {
    const auto segment_points = sample_linear_points(points[index - 1], points[index], step_m);
    for (const auto & point : segment_points) {
      append_unique_point(&sampled, point);
    }
  }
  return sampled;
}

std::vector<Point2> optimize_polyline(
  const std::vector<Point2> & points,
  const double min_segment_length_m)
{
  if (points.size() < 2U) {
    return {};
  }

  std::vector<Point2> deduped;
  deduped.reserve(points.size());
  for (const auto & point : points) {
    append_unique_point(&deduped, point);
  }
  if (deduped.size() < 2U) {
    return {};
  }

  const double threshold = std::max(min_segment_length_m, kGeometryEpsilon);
  if (polyline_length(deduped) <= threshold) {
    return {};
  }

  std::vector<Point2> optimized;
  optimized.reserve(deduped.size());
  optimized.push_back(deduped.front());

  double accumulated_length = 0.0;
  for (std::size_t index = 1; index < deduped.size(); ++index) {
    accumulated_length += distance_xy(deduped[index - 1], deduped[index]);
    const bool is_last = (index + 1U == deduped.size());
    if (accumulated_length + kGeometryEpsilon < threshold && !is_last) {
      continue;
    }
    append_unique_point(&optimized, deduped[index]);
    accumulated_length = 0.0;
  }

  if (optimized.size() < 2U || polyline_length(optimized) <= threshold) {
    return {};
  }
  return optimized;
}

std::vector<SampledPath> sampled_paths_from_plan(const PathPlan & plan, const SamplePolicy & policy) {
  std::vector<SampledPath> sampled_paths;
  std::vector<Point2> active_points;
  bool pen_down = false;
  bool active_draw = false;
  bool has_active = false;

  const auto flush_active = [&]() {
    if (!has_active || active_points.size() < 2U) {
      active_points.clear();
      has_active = false;
      return;
    }
    sampled_paths.push_back(SampledPath{active_draw, active_points});
    active_points.clear();
    has_active = false;
  };

  const auto append_geometry = [&](const std::vector<Point2> & points, const bool draw) {
    if (points.size() < 2U) {
      return;
    }
    if (!has_active) {
      active_draw = draw;
      has_active = true;
    } else if (active_draw != draw) {
      flush_active();
      active_draw = draw;
      has_active = true;
    }
    for (const auto & point : points) {
      append_unique_point(&active_points, point);
    }
  };

  for (const auto & command : plan.commands) {
    if (std::holds_alternative<PenUp>(command)) {
      flush_active();
      pen_down = false;
      continue;
    }
    if (std::holds_alternative<PenDown>(command)) {
      flush_active();
      pen_down = true;
      continue;
    }
    if (const auto * travel = std::get_if<TravelMove>(&command)) {
      append_geometry(sample_geometry(*travel, policy), false);
      continue;
    }
    if (const auto * line = std::get_if<LineSegment>(&command)) {
      append_geometry(sample_geometry(*line, policy), pen_down);
      continue;
    }
    if (const auto * arc = std::get_if<ArcSegment>(&command)) {
      append_geometry(sample_geometry(*arc, policy), pen_down);
      continue;
    }
    if (const auto * quadratic = std::get_if<QuadraticBezier>(&command)) {
      append_geometry(sample_geometry(*quadratic, policy), pen_down);
      continue;
    }
    if (const auto * cubic = std::get_if<CubicBezier>(&command)) {
      append_geometry(sample_geometry(*cubic, policy), pen_down);
    }
  }

  flush_active();
  return sampled_paths;
}

}  // namespace wall_climber::geometry
