#pragma once

#include <vector>

#include "wall_climber_draw_body/geometry_types.hpp"

namespace wall_climber::geometry {

std::vector<Point2> sample_geometry(const TravelMove & segment, const SamplePolicy & policy);
std::vector<Point2> sample_geometry(const LineSegment & segment, const SamplePolicy & policy);
std::vector<Point2> sample_geometry(const ArcSegment & segment, const SamplePolicy & policy);
std::vector<Point2> sample_geometry(const QuadraticBezier & segment, const SamplePolicy & policy);
std::vector<Point2> sample_geometry(const CubicBezier & segment, const SamplePolicy & policy);
std::vector<Point2> sample_polyline(
  const std::vector<Point2> & points,
  bool draw,
  const SamplePolicy & policy);
std::vector<Point2> optimize_polyline(
  const std::vector<Point2> & points,
  double min_segment_length_m);

std::vector<SampledPath> sampled_paths_from_plan(
  const PathPlan & plan,
  const SamplePolicy & policy);

}  // namespace wall_climber::geometry
