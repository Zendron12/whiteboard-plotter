#pragma once

#include "wall_climber_draw_body/geometry_types.hpp"
#include "wall_climber_interfaces/msg/primitive_path_plan.hpp"

namespace wall_climber::transport {

geometry::PathPlan primitive_path_plan_to_path_plan(
  const wall_climber_interfaces::msg::PrimitivePathPlan & plan);

}  // namespace wall_climber::transport
