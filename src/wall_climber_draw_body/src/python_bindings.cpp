#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "wall_climber_draw_body/geometry_eval.hpp"
#include "wall_climber_draw_body/geometry_sampling.hpp"

namespace py = pybind11;

namespace wall_climber::geometry {
namespace {

Point2 parse_point(const py::handle & handle, const char * field_name) {
  const auto point = py::cast<std::vector<double>>(handle);
  if (point.size() != 2U) {
    throw std::runtime_error(std::string(field_name) + " must contain exactly two coordinates");
  }
  return Point2{point[0], point[1]};
}

PathCommand parse_command(const py::dict & command_dict) {
  const std::string type = py::cast<std::string>(command_dict["type"]);
  if (type == "pen_up") {
    return PenUp{};
  }
  if (type == "pen_down") {
    return PenDown{};
  }
  if (type == "travel") {
    return TravelMove{
      parse_point(command_dict["start"], "TravelMove.start"),
      parse_point(command_dict["end"], "TravelMove.end"),
    };
  }
  if (type == "line") {
    return LineSegment{
      parse_point(command_dict["start"], "LineSegment.start"),
      parse_point(command_dict["end"], "LineSegment.end"),
    };
  }
  if (type == "arc") {
    return ArcSegment{
      parse_point(command_dict["center"], "ArcSegment.center"),
      py::cast<double>(command_dict["radius"]),
      py::cast<double>(command_dict["start_angle_rad"]),
      py::cast<double>(command_dict["sweep_angle_rad"]),
    };
  }
  if (type == "quadratic") {
    return QuadraticBezier{
      parse_point(command_dict["start"], "QuadraticBezier.start"),
      parse_point(command_dict["control"], "QuadraticBezier.control"),
      parse_point(command_dict["end"], "QuadraticBezier.end"),
    };
  }
  if (type == "cubic") {
    return CubicBezier{
      parse_point(command_dict["start"], "CubicBezier.start"),
      parse_point(command_dict["control1"], "CubicBezier.control1"),
      parse_point(command_dict["control2"], "CubicBezier.control2"),
      parse_point(command_dict["end"], "CubicBezier.end"),
    };
  }
  throw std::runtime_error("Unsupported canonical command type: " + type);
}

PathPlan parse_plan(const py::dict & plan_dict) {
  PathPlan plan;
  plan.frame = py::cast<std::string>(plan_dict["frame"]);
  plan.theta_ref = py::cast<double>(plan_dict["theta_ref"]);
  for (const auto & item : py::cast<py::list>(plan_dict["commands"])) {
    plan.commands.push_back(parse_command(py::cast<py::dict>(item)));
  }
  if (plan.commands.empty()) {
    throw std::runtime_error("Canonical plan must contain at least one command");
  }
  return plan;
}

py::list sample_canonical_plan(
  const py::dict & plan_dict,
  const double curve_tolerance_m,
  const py::object & draw_step_m,
  const py::object & travel_step_m,
  const py::object & max_heading_delta_rad)
{
  const PathPlan plan = parse_plan(plan_dict);
  SamplePolicy policy;
  policy.curve_tolerance_m = curve_tolerance_m;
  policy.min_step_m = 1.0e-4;
  if (!draw_step_m.is_none()) {
    policy.draw_step_m = py::cast<double>(draw_step_m);
  }
  if (!travel_step_m.is_none()) {
    policy.travel_step_m = py::cast<double>(travel_step_m);
  }
  if (!max_heading_delta_rad.is_none()) {
    policy.max_heading_delta_rad = py::cast<double>(max_heading_delta_rad);
  }
  const auto sampled_paths = sampled_paths_from_plan(
    plan,
    policy);

  py::list output;
  for (const auto & sampled : sampled_paths) {
    py::list points;
    for (const auto & point : sampled.points) {
      points.append(py::make_tuple(point.x, point.y));
    }
    py::dict item;
    item["draw"] = sampled.draw;
    item["points"] = points;
    output.append(item);
  }
  return output;
}

py::dict sample_canonical_plan_with_metadata(
  const py::dict & plan_dict,
  const double curve_tolerance_m,
  const py::object & draw_step_m,
  const py::object & travel_step_m,
  const py::object & max_heading_delta_rad)
{
  const py::list sampled = sample_canonical_plan(
    plan_dict,
    curve_tolerance_m,
    draw_step_m,
    travel_step_m,
    max_heading_delta_rad);
  std::size_t draw_path_count = 0U;
  std::size_t travel_path_count = 0U;
  std::size_t draw_point_count = 0U;
  std::size_t travel_point_count = 0U;
  for (const auto & item : sampled) {
    const auto sampled_item = py::cast<py::dict>(item);
    const bool draw = py::cast<bool>(sampled_item["draw"]);
    const auto points = py::cast<py::list>(sampled_item["points"]);
    if (draw) {
      ++draw_path_count;
      draw_point_count += points.size();
    } else {
      ++travel_path_count;
      travel_point_count += points.size();
    }
  }

  py::dict metrics;
  metrics["draw_path_count"] = py::int_(draw_path_count);
  metrics["travel_path_count"] = py::int_(travel_path_count);
  metrics["draw_point_count"] = py::int_(draw_point_count);
  metrics["travel_point_count"] = py::int_(travel_point_count);
  metrics["total_point_count"] = py::int_(draw_point_count + travel_point_count);

  py::dict payload;
  payload["paths"] = sampled;
  payload["metrics"] = metrics;
  return payload;
}

double command_path_length(const py::dict & command_dict, const double tolerance_m) {
  const auto command = parse_command(command_dict);
  return std::visit(
    [&](const auto & segment) -> double {
      using SegmentT = std::decay_t<decltype(segment)>;
      if constexpr (std::is_same_v<SegmentT, PenUp> || std::is_same_v<SegmentT, PenDown>) {
        return 0.0;
      } else if constexpr (
        std::is_same_v<SegmentT, TravelMove> ||
        std::is_same_v<SegmentT, LineSegment> ||
        std::is_same_v<SegmentT, ArcSegment>)
      {
        return path_length(segment);
      } else {
        return path_length(segment, tolerance_m);
      }
    },
    command);
}

double command_curvature(const py::dict & command_dict, const double t) {
  const auto command = parse_command(command_dict);
  return std::visit(
    [&](const auto & segment) -> double {
      using SegmentT = std::decay_t<decltype(segment)>;
      if constexpr (std::is_same_v<SegmentT, PenUp> || std::is_same_v<SegmentT, PenDown>) {
        return 0.0;
      } else {
        return curvature(segment, t);
      }
    },
    command);
}

}  // namespace
}  // namespace wall_climber::geometry

PYBIND11_MODULE(wall_climber_geometry_cpp, module) {
  module.doc() = "Curve-aware geometry sampler for wall_climber canonical plans";
  module.def(
    "sample_canonical_plan",
    &wall_climber::geometry::sample_canonical_plan,
    py::arg("plan"),
    py::arg("curve_tolerance_m") = 0.01,
    py::arg("draw_step_m") = py::none(),
    py::arg("travel_step_m") = py::none(),
    py::arg("max_heading_delta_rad") = py::none());
  module.def(
    "sample_canonical_plan_with_metadata",
    &wall_climber::geometry::sample_canonical_plan_with_metadata,
    py::arg("plan"),
    py::arg("curve_tolerance_m") = 0.01,
    py::arg("draw_step_m") = py::none(),
    py::arg("travel_step_m") = py::none(),
    py::arg("max_heading_delta_rad") = py::none());
  module.def(
    "path_length",
    &wall_climber::geometry::command_path_length,
    py::arg("command"),
    py::arg("tolerance_m") = 1.0e-4);
  module.def(
    "curvature",
    &wall_climber::geometry::command_curvature,
    py::arg("command"),
    py::arg("t") = 0.5);
}
