#include <cmath>

#include "gtest/gtest.h"

#include "wall_climber_draw_body/transport_conversions.hpp"
#include "wall_climber_interfaces/msg/path_primitive.hpp"
#include "wall_climber_interfaces/msg/primitive_path_plan.hpp"

namespace transport = wall_climber::transport;
namespace geometry = wall_climber::geometry;

TEST(TransportConversions, PrimitivePathPlanConversionPreservesCurvesAndPenEvents) {
  wall_climber_interfaces::msg::PrimitivePathPlan plan;
  plan.frame = "board";
  plan.theta_ref = 0.0;

  wall_climber_interfaces::msg::PathPrimitive pen_down;
  pen_down.type = wall_climber_interfaces::msg::PathPrimitive::PEN_DOWN;
  pen_down.pen_down = true;
  plan.primitives.push_back(pen_down);

  wall_climber_interfaces::msg::PathPrimitive quadratic;
  quadratic.type = wall_climber_interfaces::msg::PathPrimitive::QUADRATIC_BEZIER;
  quadratic.pen_down = true;
  quadratic.start.x = 0.0;
  quadratic.start.y = 0.0;
  quadratic.control1.x = 0.5;
  quadratic.control1.y = 1.0;
  quadratic.end.x = 1.0;
  quadratic.end.y = 0.0;
  plan.primitives.push_back(quadratic);

  wall_climber_interfaces::msg::PathPrimitive arc;
  arc.type = wall_climber_interfaces::msg::PathPrimitive::ARC_SEGMENT;
  arc.pen_down = true;
  arc.center.x = 1.0;
  arc.center.y = 1.0;
  arc.radius = 0.5;
  arc.start_angle_rad = 0.0;
  arc.sweep_angle_rad = -1.57079632679;
  arc.clockwise = true;
  plan.primitives.push_back(arc);

  wall_climber_interfaces::msg::PathPrimitive pen_up;
  pen_up.type = wall_climber_interfaces::msg::PathPrimitive::PEN_UP;
  plan.primitives.push_back(pen_up);

  const geometry::PathPlan converted = transport::primitive_path_plan_to_path_plan(plan);

  ASSERT_EQ(converted.commands.size(), 4U);
  EXPECT_TRUE(std::holds_alternative<geometry::PenDown>(converted.commands[0]));
  EXPECT_TRUE(std::holds_alternative<geometry::QuadraticBezier>(converted.commands[1]));
  EXPECT_TRUE(std::holds_alternative<geometry::ArcSegment>(converted.commands[2]));
  EXPECT_TRUE(std::holds_alternative<geometry::PenUp>(converted.commands[3]));

  const auto & arc_segment = std::get<geometry::ArcSegment>(converted.commands[2]);
  EXPECT_LT(arc_segment.sweep_angle_rad, 0.0);
  EXPECT_NEAR(arc_segment.radius, 0.5, 1.0e-9);
}
