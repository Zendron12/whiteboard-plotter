#include "wall_climber_draw_body/geometry_eval.hpp"

#include <algorithm>
#include <cmath>

namespace wall_climber::geometry {

namespace {

constexpr int kMaxIntegrationDepth = 18;

double clamp_unit_interval(const double t) {
  return std::max(0.0, std::min(1.0, t));
}

Point2 subtract(const Point2 & a, const Point2 & b) {
  return Point2{a.x - b.x, a.y - b.y};
}

double norm(const Point2 & value) {
  return std::hypot(value.x, value.y);
}

Point2 add(const Point2 & a, const Point2 & b) {
  return Point2{a.x + b.x, a.y + b.y};
}

Point2 scale(const Point2 & value, const double factor) {
  return Point2{value.x * factor, value.y * factor};
}

Point2 quadratic_second_derivative(const QuadraticBezier & segment) {
  return Point2{
    2.0 * (segment.end.x - (2.0 * segment.control.x) + segment.start.x),
    2.0 * (segment.end.y - (2.0 * segment.control.y) + segment.start.y),
  };
}

Point2 cubic_second_derivative(const CubicBezier & segment, const double t) {
  const double ratio = clamp_unit_interval(t);
  const double omt = 1.0 - ratio;
  return Point2{
    (6.0 * omt * (segment.control2.x - (2.0 * segment.control1.x) + segment.start.x)) +
      (6.0 * ratio * (segment.end.x - (2.0 * segment.control2.x) + segment.control1.x)),
    (6.0 * omt * (segment.control2.y - (2.0 * segment.control1.y) + segment.start.y)) +
      (6.0 * ratio * (segment.end.y - (2.0 * segment.control2.y) + segment.control1.y)),
  };
}

double cross_z(const Point2 & a, const Point2 & b) {
  return (a.x * b.y) - (a.y * b.x);
}

double safe_speed(const Point2 & tangent) {
  return std::max(norm(tangent), 1.0e-12);
}

template<typename SegmentT>
double simpson_arc_length(
  const SegmentT & segment,
  const double a,
  const double b)
{
  const double mid = 0.5 * (a + b);
  const double fa = safe_speed(tangent_vector(segment, a));
  const double fm = safe_speed(tangent_vector(segment, mid));
  const double fb = safe_speed(tangent_vector(segment, b));
  return ((b - a) / 6.0) * (fa + (4.0 * fm) + fb);
}

template<typename SegmentT>
double adaptive_arc_length_recursive(
  const SegmentT & segment,
  const double a,
  const double b,
  const double whole,
  const double tolerance_m,
  const int depth)
{
  const double mid = 0.5 * (a + b);
  const double left = simpson_arc_length(segment, a, mid);
  const double right = simpson_arc_length(segment, mid, b);
  const double delta = std::abs((left + right) - whole);
  if (depth >= kMaxIntegrationDepth || delta <= (15.0 * tolerance_m)) {
    return left + right + (((left + right) - whole) / 15.0);
  }
  return adaptive_arc_length_recursive(segment, a, mid, left, 0.5 * tolerance_m, depth + 1) +
         adaptive_arc_length_recursive(segment, mid, b, right, 0.5 * tolerance_m, depth + 1);
}

template<typename SegmentT>
double adaptive_arc_length(const SegmentT & segment, const double tolerance_m) {
  const double whole = simpson_arc_length(segment, 0.0, 1.0);
  return adaptive_arc_length_recursive(
    segment,
    0.0,
    1.0,
    whole,
    std::max(tolerance_m, 1.0e-7),
    0);
}

double signed_curvature_from_derivatives(
  const Point2 & first_derivative,
  const Point2 & second_derivative)
{
  const double speed = norm(first_derivative);
  if (speed <= 1.0e-12) {
    return 0.0;
  }
  return cross_z(first_derivative, second_derivative) / (speed * speed * speed);
}

}  // namespace

double distance_xy(const Point2 & a, const Point2 & b) {
  return std::hypot(a.x - b.x, a.y - b.y);
}

bool approximately_equal(const Point2 & a, const Point2 & b, const double eps) {
  return distance_xy(a, b) <= eps;
}

Point2 lerp(const Point2 & start, const Point2 & end, const double t) {
  const double ratio = clamp_unit_interval(t);
  return Point2{
    start.x + (end.x - start.x) * ratio,
    start.y + (end.y - start.y) * ratio,
  };
}

double heading_rad(const Point2 & vector) {
  return std::atan2(vector.y, vector.x);
}

Point2 evaluate_point(const TravelMove & segment, const double t) {
  return lerp(segment.start, segment.end, t);
}

Point2 evaluate_point(const LineSegment & segment, const double t) {
  return lerp(segment.start, segment.end, t);
}

Point2 evaluate_point(const ArcSegment & segment, const double t) {
  const double ratio = clamp_unit_interval(t);
  const double angle = segment.start_angle_rad + segment.sweep_angle_rad * ratio;
  return Point2{
    segment.center.x + segment.radius * std::cos(angle),
    segment.center.y + segment.radius * std::sin(angle),
  };
}

Point2 evaluate_point(const QuadraticBezier & segment, const double t) {
  const double ratio = clamp_unit_interval(t);
  const double omt = 1.0 - ratio;
  return Point2{
    (omt * omt * segment.start.x) +
      (2.0 * omt * ratio * segment.control.x) +
      (ratio * ratio * segment.end.x),
    (omt * omt * segment.start.y) +
      (2.0 * omt * ratio * segment.control.y) +
      (ratio * ratio * segment.end.y),
  };
}

Point2 evaluate_point(const CubicBezier & segment, const double t) {
  const double ratio = clamp_unit_interval(t);
  const double omt = 1.0 - ratio;
  return Point2{
    (omt * omt * omt * segment.start.x) +
      (3.0 * omt * omt * ratio * segment.control1.x) +
      (3.0 * omt * ratio * ratio * segment.control2.x) +
      (ratio * ratio * ratio * segment.end.x),
    (omt * omt * omt * segment.start.y) +
      (3.0 * omt * omt * ratio * segment.control1.y) +
      (3.0 * omt * ratio * ratio * segment.control2.y) +
      (ratio * ratio * ratio * segment.end.y),
  };
}

Point2 tangent_vector(const TravelMove & segment, const double /*t*/) {
  return subtract(segment.end, segment.start);
}

Point2 tangent_vector(const LineSegment & segment, const double /*t*/) {
  return subtract(segment.end, segment.start);
}

Point2 tangent_vector(const ArcSegment & segment, const double t) {
  const double ratio = clamp_unit_interval(t);
  const double angle = segment.start_angle_rad + segment.sweep_angle_rad * ratio;
  const double sweep_sign = (segment.sweep_angle_rad >= 0.0) ? 1.0 : -1.0;
  return Point2{
    -std::sin(angle) * sweep_sign * segment.radius,
    std::cos(angle) * sweep_sign * segment.radius,
  };
}

Point2 tangent_vector(const QuadraticBezier & segment, const double t) {
  const double ratio = clamp_unit_interval(t);
  return Point2{
    2.0 * (1.0 - ratio) * (segment.control.x - segment.start.x) +
      2.0 * ratio * (segment.end.x - segment.control.x),
    2.0 * (1.0 - ratio) * (segment.control.y - segment.start.y) +
      2.0 * ratio * (segment.end.y - segment.control.y),
  };
}

Point2 tangent_vector(const CubicBezier & segment, const double t) {
  const double ratio = clamp_unit_interval(t);
  const double omt = 1.0 - ratio;
  return Point2{
    3.0 * omt * omt * (segment.control1.x - segment.start.x) +
      6.0 * omt * ratio * (segment.control2.x - segment.control1.x) +
      3.0 * ratio * ratio * (segment.end.x - segment.control2.x),
    3.0 * omt * omt * (segment.control1.y - segment.start.y) +
      6.0 * omt * ratio * (segment.control2.y - segment.control1.y) +
      3.0 * ratio * ratio * (segment.end.y - segment.control2.y),
  };
}

double path_length(const TravelMove & segment) {
  return distance_xy(segment.start, segment.end);
}

double path_length(const LineSegment & segment) {
  return distance_xy(segment.start, segment.end);
}

double path_length(const ArcSegment & segment) {
  return std::abs(segment.radius * segment.sweep_angle_rad);
}

double path_length(const QuadraticBezier & segment, const double tolerance_m) {
  return adaptive_arc_length(segment, tolerance_m);
}

double path_length(const CubicBezier & segment, const double tolerance_m) {
  return adaptive_arc_length(segment, tolerance_m);
}

double curvature(const TravelMove & /*segment*/, const double /*t*/) {
  return 0.0;
}

double curvature(const LineSegment & /*segment*/, const double /*t*/) {
  return 0.0;
}

double curvature(const ArcSegment & segment, const double /*t*/) {
  if (std::abs(segment.sweep_angle_rad) <= 1.0e-12) {
    return 0.0;
  }
  return (segment.sweep_angle_rad >= 0.0 ? 1.0 : -1.0) / segment.radius;
}

double curvature(const QuadraticBezier & segment, const double t) {
  return signed_curvature_from_derivatives(
    tangent_vector(segment, t),
    quadratic_second_derivative(segment));
}

double curvature(const CubicBezier & segment, const double t) {
  return signed_curvature_from_derivatives(
    tangent_vector(segment, t),
    cubic_second_derivative(segment, t));
}

}  // namespace wall_climber::geometry
