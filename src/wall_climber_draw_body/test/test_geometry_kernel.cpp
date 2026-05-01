#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "wall_climber_draw_body/geometry_eval.hpp"
#include "wall_climber_draw_body/geometry_sampling.hpp"

namespace geometry = wall_climber::geometry;

namespace {
constexpr double kPi = 3.14159265358979323846;
}

TEST(GeometryEval, EvaluatesLineAndQuadraticDeterministically) {
  const geometry::LineSegment line{{0.0, 0.0}, {2.0, 0.0}};
  const geometry::Point2 midpoint = geometry::evaluate_point(line, 0.5);
  EXPECT_NEAR(midpoint.x, 1.0, 1.0e-9);
  EXPECT_NEAR(midpoint.y, 0.0, 1.0e-9);

  const geometry::QuadraticBezier curve{{0.0, 0.0}, {1.0, 2.0}, {2.0, 0.0}};
  const geometry::Point2 tangent = geometry::tangent_vector(curve, 0.5);
  EXPECT_NEAR(tangent.x, 2.0, 1.0e-9);
  EXPECT_NEAR(tangent.y, 0.0, 1.0e-9);

  const double length_a = geometry::path_length(curve, 1.0e-4);
  const double length_b = geometry::path_length(curve, 1.0e-4);
  EXPECT_NEAR(length_a, length_b, 1.0e-12);
  EXPECT_GT(length_a, geometry::distance_xy(curve.start, curve.end));
  EXPECT_LT(geometry::curvature(curve, 0.5), 0.0);
}

TEST(GeometryEval, ComputesArcEndpointsAndHeading) {
  const geometry::ArcSegment arc{{0.0, 0.0}, 1.0, 0.0, kPi / 2.0};
  const geometry::Point2 start = geometry::evaluate_point(arc, 0.0);
  const geometry::Point2 end = geometry::evaluate_point(arc, 1.0);
  const geometry::Point2 tangent = geometry::tangent_vector(arc, 0.0);

  EXPECT_NEAR(start.x, 1.0, 1.0e-9);
  EXPECT_NEAR(start.y, 0.0, 1.0e-9);
  EXPECT_NEAR(end.x, 0.0, 1.0e-9);
  EXPECT_NEAR(end.y, 1.0, 1.0e-9);
  EXPECT_NEAR(geometry::heading_rad(tangent), kPi / 2.0, 1.0e-9);
  EXPECT_NEAR(geometry::path_length(arc), kPi / 2.0, 1.0e-9);
  EXPECT_NEAR(geometry::curvature(arc, 0.5), 1.0, 1.0e-9);
}

TEST(GeometrySampling, SamplesCanonicalPlanIntoSeparatedDrawAndTravelPaths) {
  geometry::PathPlan plan;
  plan.frame = "board";
  plan.theta_ref = 0.0;
  plan.commands = {
    geometry::PenDown{},
    geometry::QuadraticBezier{{0.0, 0.0}, {0.5, 1.0}, {1.0, 0.0}},
    geometry::PenUp{},
    geometry::TravelMove{{1.0, 0.0}, {2.0, 0.0}},
    geometry::PenDown{},
    geometry::CubicBezier{{2.0, 0.0}, {2.2, 0.5}, {2.8, -0.5}, {3.0, 0.0}},
    geometry::PenUp{},
  };

  const auto sampled = geometry::sampled_paths_from_plan(
    plan,
    geometry::SamplePolicy{0.1, 1.0e-4, 0.0, 0.0, 0.35});

  ASSERT_EQ(sampled.size(), 3U);
  EXPECT_TRUE(sampled[0].draw);
  EXPECT_FALSE(sampled[1].draw);
  EXPECT_TRUE(sampled[2].draw);

  ASSERT_GE(sampled[0].points.size(), 3U);
  ASSERT_GE(sampled[2].points.size(), 4U);
  EXPECT_NEAR(sampled[0].points.front().x, 0.0, 1.0e-9);
  EXPECT_NEAR(sampled[0].points.front().y, 0.0, 1.0e-9);
  EXPECT_NEAR(sampled[2].points.back().x, 3.0, 1.0e-9);
  EXPECT_NEAR(sampled[2].points.back().y, 0.0, 1.0e-9);
}

TEST(GeometrySampling, TightHeadingDeltaProducesDenserCurveSampling) {
  const geometry::CubicBezier curve{{0.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {1.0, 0.0}};
  geometry::SamplePolicy loose;
  loose.curve_tolerance_m = 0.05;
  loose.max_heading_delta_rad = 0.6;
  geometry::SamplePolicy tight = loose;
  tight.max_heading_delta_rad = 0.1;

  const auto loose_points = geometry::sample_geometry(curve, loose);
  const auto tight_points = geometry::sample_geometry(curve, tight);

  EXPECT_GT(tight_points.size(), loose_points.size());
}

TEST(GeometrySampling, SamplesPolylineWithExecutionStepSizes) {
  const std::vector<geometry::Point2> polyline{{0.0, 0.0}, {0.03, 0.0}};
  geometry::SamplePolicy policy;
  policy.curve_tolerance_m = 0.1;
  policy.min_step_m = 1.0e-4;
  policy.draw_step_m = 0.01;
  policy.travel_step_m = 0.02;

  const auto draw_points = geometry::sample_polyline(polyline, true, policy);
  const auto travel_points = geometry::sample_polyline(polyline, false, policy);

  ASSERT_EQ(draw_points.size(), 4U);
  EXPECT_NEAR(draw_points[1].x, 0.01, 1.0e-9);
  EXPECT_NEAR(draw_points[2].x, 0.02, 1.0e-9);

  ASSERT_EQ(travel_points.size(), 3U);
  EXPECT_NEAR(travel_points[1].x, 0.015, 1.0e-9);
}

TEST(GeometrySampling, OptimizesPolylineByRemovingTinySegments) {
  const std::vector<geometry::Point2> polyline{
    {0.0, 0.0},
    {1.0e-6, 0.0},
    {0.02, 0.0},
    {0.020001, 0.0},
    {0.05, 0.0},
  };

  const auto optimized = geometry::optimize_polyline(polyline, 0.01);

  ASSERT_EQ(optimized.size(), 3U);
  EXPECT_NEAR(optimized.front().x, 0.0, 1.0e-9);
  EXPECT_NEAR(optimized[1].x, 0.02, 1.0e-9);
  EXPECT_NEAR(optimized.back().x, 0.05, 1.0e-9);
}
