#include "wall_climber_draw_body/transport_conversions.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "wall_climber_draw_body/geometry_eval.hpp"
#include "wall_climber_interfaces/msg/board_point.hpp"
#include "wall_climber_interfaces/msg/path_primitive.hpp"

namespace wall_climber::transport {

namespace {

namespace geometry = wall_climber::geometry;

geometry::Point2 to_point(const wall_climber_interfaces::msg::BoardPoint & point) {
  return geometry::Point2{point.x, point.y};
}

void validate_point(const geometry::Point2 & point, const char * label) {
  if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
    throw std::runtime_error(std::string(label) + " must be finite");
  }
}

void sync_pen_state(
  const bool desired_pen_down,
  bool * current_pen_down,
  std::vector<geometry::PathCommand> * commands)
{
  if (*current_pen_down == desired_pen_down) {
    return;
  }
  if (desired_pen_down) {
    commands->push_back(geometry::PenDown{});
  } else {
    commands->push_back(geometry::PenUp{});
  }
  *current_pen_down = desired_pen_down;
}

double normalized_sweep(const wall_climber_interfaces::msg::PathPrimitive & primitive) {
  double sweep = primitive.sweep_angle_rad;
  if (primitive.clockwise && sweep > 0.0) {
    sweep = -sweep;
  } else if (!primitive.clockwise && sweep < 0.0) {
    sweep = -sweep;
  }
  return sweep;
}

}  // namespace

geometry::PathPlan primitive_path_plan_to_path_plan(
  const wall_climber_interfaces::msg::PrimitivePathPlan & plan)
{
  geometry::PathPlan output;
  output.frame = plan.frame;
  output.theta_ref = plan.theta_ref;

  bool pen_down = false;
  for (const auto & primitive : plan.primitives) {
    switch (primitive.type) {
      case wall_climber_interfaces::msg::PathPrimitive::PEN_UP:
        output.commands.push_back(geometry::PenUp{});
        pen_down = false;
        break;
      case wall_climber_interfaces::msg::PathPrimitive::PEN_DOWN:
        output.commands.push_back(geometry::PenDown{});
        pen_down = true;
        break;
      case wall_climber_interfaces::msg::PathPrimitive::TRAVEL_MOVE: {
        sync_pen_state(false, &pen_down, &output.commands);
        const geometry::Point2 start = to_point(primitive.start);
        const geometry::Point2 end = to_point(primitive.end);
        validate_point(start, "TravelMove.start");
        validate_point(end, "TravelMove.end");
        output.commands.push_back(geometry::TravelMove{start, end});
        break;
      }
      case wall_climber_interfaces::msg::PathPrimitive::LINE_SEGMENT: {
        sync_pen_state(primitive.pen_down, &pen_down, &output.commands);
        const geometry::Point2 start = to_point(primitive.start);
        const geometry::Point2 end = to_point(primitive.end);
        validate_point(start, "LineSegment.start");
        validate_point(end, "LineSegment.end");
        output.commands.push_back(geometry::LineSegment{start, end});
        break;
      }
      case wall_climber_interfaces::msg::PathPrimitive::ARC_SEGMENT: {
        sync_pen_state(primitive.pen_down, &pen_down, &output.commands);
        const geometry::Point2 center = to_point(primitive.center);
        validate_point(center, "ArcSegment.center");
        if (!std::isfinite(primitive.radius) || primitive.radius <= 0.0) {
          throw std::runtime_error("ArcSegment.radius must be > 0");
        }
        const double start_angle_rad = primitive.start_angle_rad;
        const double sweep_angle_rad = normalized_sweep(primitive);
        if (!std::isfinite(start_angle_rad) || !std::isfinite(sweep_angle_rad)) {
          throw std::runtime_error("ArcSegment angles must be finite");
        }
        output.commands.push_back(
          geometry::ArcSegment{center, primitive.radius, start_angle_rad, sweep_angle_rad});
        break;
      }
      case wall_climber_interfaces::msg::PathPrimitive::QUADRATIC_BEZIER: {
        sync_pen_state(primitive.pen_down, &pen_down, &output.commands);
        const geometry::Point2 start = to_point(primitive.start);
        const geometry::Point2 control = to_point(primitive.control1);
        const geometry::Point2 end = to_point(primitive.end);
        validate_point(start, "QuadraticBezier.start");
        validate_point(control, "QuadraticBezier.control");
        validate_point(end, "QuadraticBezier.end");
        output.commands.push_back(geometry::QuadraticBezier{start, control, end});
        break;
      }
      case wall_climber_interfaces::msg::PathPrimitive::CUBIC_BEZIER: {
        sync_pen_state(primitive.pen_down, &pen_down, &output.commands);
        const geometry::Point2 start = to_point(primitive.start);
        const geometry::Point2 control1 = to_point(primitive.control1);
        const geometry::Point2 control2 = to_point(primitive.control2);
        const geometry::Point2 end = to_point(primitive.end);
        validate_point(start, "CubicBezier.start");
        validate_point(control1, "CubicBezier.control1");
        validate_point(control2, "CubicBezier.control2");
        validate_point(end, "CubicBezier.end");
        output.commands.push_back(geometry::CubicBezier{start, control1, control2, end});
        break;
      }
      default:
        throw std::runtime_error("primitive path plan contains an unsupported primitive type");
    }
  }

  if (output.commands.empty()) {
    throw std::runtime_error("primitive path plan must contain at least one primitive");
  }
  return output;
}

}  // namespace wall_climber::transport
