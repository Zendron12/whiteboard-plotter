#pragma once

#include <string>
#include <variant>
#include <vector>

namespace wall_climber::geometry {

struct Point2 {
  double x{0.0};
  double y{0.0};
};

struct PenUp {};

struct PenDown {};

struct TravelMove {
  Point2 start;
  Point2 end;
};

struct LineSegment {
  Point2 start;
  Point2 end;
};

struct ArcSegment {
  Point2 center;
  double radius{0.0};
  double start_angle_rad{0.0};
  double sweep_angle_rad{0.0};
};

struct QuadraticBezier {
  Point2 start;
  Point2 control;
  Point2 end;
};

struct CubicBezier {
  Point2 start;
  Point2 control1;
  Point2 control2;
  Point2 end;
};

using PathCommand = std::variant<
  PenUp,
  PenDown,
  TravelMove,
  LineSegment,
  ArcSegment,
  QuadraticBezier,
  CubicBezier>;

struct PathPlan {
  std::string frame{"board"};
  double theta_ref{0.0};
  std::vector<PathCommand> commands;
};

struct SamplePolicy {
  double curve_tolerance_m{0.01};
  double min_step_m{1.0e-4};
  double draw_step_m{0.0};
  double travel_step_m{0.0};
  double max_heading_delta_rad{0.3490658503988659};  // 20 deg
};

struct SampledPath {
  bool draw{false};
  std::vector<Point2> points;
};

}  // namespace wall_climber::geometry
