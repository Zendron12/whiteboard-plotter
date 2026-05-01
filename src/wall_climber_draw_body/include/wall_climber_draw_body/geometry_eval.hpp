#pragma once

#include "wall_climber_draw_body/geometry_types.hpp"

namespace wall_climber::geometry {

constexpr double kGeometryEpsilon = 1.0e-9;

double distance_xy(const Point2 & a, const Point2 & b);
bool approximately_equal(const Point2 & a, const Point2 & b, double eps = kGeometryEpsilon);
Point2 lerp(const Point2 & start, const Point2 & end, double t);
double heading_rad(const Point2 & vector);

Point2 evaluate_point(const TravelMove & segment, double t);
Point2 evaluate_point(const LineSegment & segment, double t);
Point2 evaluate_point(const ArcSegment & segment, double t);
Point2 evaluate_point(const QuadraticBezier & segment, double t);
Point2 evaluate_point(const CubicBezier & segment, double t);

Point2 tangent_vector(const TravelMove & segment, double t);
Point2 tangent_vector(const LineSegment & segment, double t);
Point2 tangent_vector(const ArcSegment & segment, double t);
Point2 tangent_vector(const QuadraticBezier & segment, double t);
Point2 tangent_vector(const CubicBezier & segment, double t);

double path_length(const TravelMove & segment);
double path_length(const LineSegment & segment);
double path_length(const ArcSegment & segment);
double path_length(const QuadraticBezier & segment, double tolerance_m = 1.0e-4);
double path_length(const CubicBezier & segment, double tolerance_m = 1.0e-4);

double curvature(const TravelMove & segment, double t);
double curvature(const LineSegment & segment, double t);
double curvature(const ArcSegment & segment, double t);
double curvature(const QuadraticBezier & segment, double t);
double curvature(const CubicBezier & segment, double t);

}  // namespace wall_climber::geometry
