#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "geometry_msgs/msg/pose2_d.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "wall_climber_draw_body/geometry_eval.hpp"
#include "wall_climber_draw_body/geometry_sampling.hpp"
#include "wall_climber_draw_body/transport_conversions.hpp"
#include "wall_climber_interfaces/msg/cable_setpoint.hpp"
#include "wall_climber_interfaces/msg/primitive_path_plan.hpp"

namespace {

namespace geometry = wall_climber::geometry;

rclcpp::QoS transient_local_qos() {
  return rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
}

using Point2D = geometry::Point2;

struct PlannedSample {
  wall_climber_interfaces::msg::CableSetpoint setpoint;
};

struct ScheduledPath {
  bool draw{false};
  int32_t segment_index{0};
  std::vector<Point2D> points;
};

struct ExecutionBuildDiagnostics {
  std::string chosen_transport;
  double transport_parse_ms{0.0};
  double schedule_build_ms{0.0};
  std::size_t primitive_count{0U};
  std::size_t sampled_path_count{0U};
  std::size_t sampled_point_count{0U};
  std::size_t schedule_count{0U};
  std::size_t chunk_count{0U};
};

double executor_distance_xy(const Point2D & a, const Point2D & b) {
  return geometry::distance_xy(a, b);
}

bool finite_point(const Point2D & point) {
  return std::isfinite(point.x) && std::isfinite(point.y);
}

bool executor_approximately_equal(
  const Point2D & a,
  const Point2D & b,
  const double eps = 1.0e-9)
{
  return geometry::approximately_equal(a, b, eps);
}

bool within_closed_interval(
  const double value, const double minimum, const double maximum, const double eps = 1.0e-6)
{
  return value >= (minimum - eps) && value <= (maximum + eps);
}

}  // namespace

class CableDrawExecutor final : public rclcpp::Node {
 public:
  CableDrawExecutor()
  : Node("cable_draw_executor") {
    declare_parameter("anchor_left_x", 0.0);
    declare_parameter("anchor_left_y", 0.0);
    declare_parameter("anchor_right_x", 6.3);
    declare_parameter("anchor_right_y", 0.0);
    declare_parameter("carriage_attachment_left_x", -0.104);
    declare_parameter("carriage_attachment_left_y", -0.075);
    declare_parameter("carriage_attachment_right_x", 0.104);
    declare_parameter("carriage_attachment_right_y", -0.075);
    declare_parameter("pen_offset_x", 0.203);
    declare_parameter("pen_offset_y", 0.020);
    declare_parameter("initial_pen_x", 3.353);
    declare_parameter("initial_pen_y", 0.970);
    declare_parameter("fixed_theta_rad", 0.0);
    declare_parameter("draw_resample_step_m", 0.003);
    declare_parameter("travel_resample_step_m", 0.010);
    declare_parameter("text_draw_resample_step_m", 0.0038);
    declare_parameter("text_travel_resample_step_m", 0.012);
    declare_parameter("publish_period_sec", 0.05);
    declare_parameter("chunk_max_paths", 48);
    declare_parameter("chunk_max_samples", 2400);
    declare_parameter("text_end_retreat_m", 0.018);
    declare_parameter("writable_x_min", 0.10);
    declare_parameter("writable_x_max", 6.20);
    declare_parameter("writable_y_min", 0.10);
    declare_parameter("writable_y_max", 2.90);
    declare_parameter("safe_x_min", 0.16);
    declare_parameter("safe_x_max", 6.14);
    declare_parameter("safe_y_min", 0.32);
    declare_parameter("safe_y_max", 2.82);
    declare_parameter("body_safe_writable_x_min", 0.348);
    declare_parameter("body_safe_writable_x_max", 6.20);
    declare_parameter("body_safe_writable_y_min", 0.12);
    declare_parameter("body_safe_writable_y_max", 2.90);
    declare_parameter("body_safe_safe_x_min", 0.348);
    declare_parameter("body_safe_safe_x_max", 6.14);
    declare_parameter("body_safe_safe_y_min", 0.32);
    declare_parameter("body_safe_safe_y_max", 2.82);
    declare_parameter("corner_keepout_radius", 0.36);
    declare_parameter("pen_down_settle_sec", 0.05);

    setpoint_pub_ = create_publisher<wall_climber_interfaces::msg::CableSetpoint>(
      "/wall_climber/cable_setpoint",
      transient_local_qos());
    status_pub_ = create_publisher<std_msgs::msg::String>(
      "/wall_climber/cable_executor_status",
      transient_local_qos());
    diagnostics_pub_ = create_publisher<std_msgs::msg::String>(
      "/wall_climber/executor_diagnostics",
      transient_local_qos());
    active_mode_sub_ = create_subscription<std_msgs::msg::String>(
      "/wall_climber/internal/active_mode",
      transient_local_qos(),
      std::bind(&CableDrawExecutor::active_mode_callback, this, std::placeholders::_1));
    primitive_plan_sub_ = create_subscription<wall_climber_interfaces::msg::PrimitivePathPlan>(
      "/wall_climber/primitive_path_plan",
      10,
      std::bind(&CableDrawExecutor::primitive_path_plan_callback, this, std::placeholders::_1));

    timer_ = create_wall_timer(
      std::chrono::duration<double>(read_publish_period_sec()),
      std::bind(&CableDrawExecutor::on_timer, this));

    current_pen_point_ = Point2D{
      get_parameter("initial_pen_x").as_double(),
      get_parameter("initial_pen_y").as_double(),
    };
    set_status("idle");
    RCLCPP_INFO(get_logger(), "Cable draw executor ready.");
  }

 private:
  struct GeometryParams {
    Point2D anchor_left;
    Point2D anchor_right;
    Point2D attach_left;
    Point2D attach_right;
    Point2D pen_offset;
    Point2D initial_pen;
    double fixed_theta_rad;
    double draw_resample_step_m;
    double travel_resample_step_m;
    double text_draw_resample_step_m;
    double text_travel_resample_step_m;
    std::size_t chunk_max_paths;
    std::size_t chunk_max_samples;
    double writable_x_min;
    double writable_x_max;
    double writable_y_min;
    double writable_y_max;
    double safe_x_min;
    double safe_x_max;
    double safe_y_min;
    double safe_y_max;
    double body_safe_writable_x_min;
    double body_safe_writable_x_max;
    double body_safe_writable_y_min;
    double body_safe_writable_y_max;
    double body_safe_safe_x_min;
    double body_safe_safe_x_max;
    double body_safe_safe_y_min;
    double body_safe_safe_y_max;
    double corner_keepout_radius;
    double pen_down_settle_sec;
    double text_end_retreat_m;
  };

  GeometryParams read_geometry_params() const {
    return GeometryParams{
      Point2D{get_parameter("anchor_left_x").as_double(), get_parameter("anchor_left_y").as_double()},
      Point2D{get_parameter("anchor_right_x").as_double(), get_parameter("anchor_right_y").as_double()},
      Point2D{
        get_parameter("carriage_attachment_left_x").as_double(),
        get_parameter("carriage_attachment_left_y").as_double(),
      },
      Point2D{
        get_parameter("carriage_attachment_right_x").as_double(),
        get_parameter("carriage_attachment_right_y").as_double(),
      },
      Point2D{get_parameter("pen_offset_x").as_double(), get_parameter("pen_offset_y").as_double()},
      Point2D{get_parameter("initial_pen_x").as_double(), get_parameter("initial_pen_y").as_double()},
      get_parameter("fixed_theta_rad").as_double(),
      std::max(1.0e-4, get_parameter("draw_resample_step_m").as_double()),
      std::max(1.0e-4, get_parameter("travel_resample_step_m").as_double()),
      std::max(1.0e-4, get_parameter("text_draw_resample_step_m").as_double()),
      std::max(1.0e-4, get_parameter("text_travel_resample_step_m").as_double()),
      static_cast<std::size_t>(std::max<int64_t>(1, get_parameter("chunk_max_paths").as_int())),
      static_cast<std::size_t>(std::max<int64_t>(1, get_parameter("chunk_max_samples").as_int())),
      get_parameter("writable_x_min").as_double(),
      get_parameter("writable_x_max").as_double(),
      get_parameter("writable_y_min").as_double(),
      get_parameter("writable_y_max").as_double(),
      get_parameter("safe_x_min").as_double(),
      get_parameter("safe_x_max").as_double(),
      get_parameter("safe_y_min").as_double(),
      get_parameter("safe_y_max").as_double(),
      get_parameter("body_safe_writable_x_min").as_double(),
      get_parameter("body_safe_writable_x_max").as_double(),
      get_parameter("body_safe_writable_y_min").as_double(),
      get_parameter("body_safe_writable_y_max").as_double(),
      get_parameter("body_safe_safe_x_min").as_double(),
      get_parameter("body_safe_safe_x_max").as_double(),
      get_parameter("body_safe_safe_y_min").as_double(),
      get_parameter("body_safe_safe_y_max").as_double(),
      std::max(0.0, get_parameter("corner_keepout_radius").as_double()),
      std::max(0.0, get_parameter("pen_down_settle_sec").as_double()),
      std::max(0.0, get_parameter("text_end_retreat_m").as_double()),
    };
  }

  double read_publish_period_sec() const {
    return std::max(1.0e-3, get_parameter("publish_period_sec").as_double());
  }

  bool point_within_writable(const Point2D & point, const GeometryParams & params) const {
    return within_closed_interval(point.x, params.writable_x_min, params.writable_x_max) &&
           within_closed_interval(point.y, params.writable_y_min, params.writable_y_max);
  }

  bool point_keeps_body_on_board(const Point2D & point, const GeometryParams & params) const {
    return within_closed_interval(
             point.x, params.body_safe_writable_x_min, params.body_safe_writable_x_max) &&
           within_closed_interval(
             point.y, params.body_safe_writable_y_min, params.body_safe_writable_y_max);
  }

  bool point_within_body_safe_workspace(const Point2D & point, const GeometryParams & params) const {
    return within_closed_interval(
             point.x, params.body_safe_safe_x_min, params.body_safe_safe_x_max) &&
           within_closed_interval(
             point.y, params.body_safe_safe_y_min, params.body_safe_safe_y_max);
  }

  bool point_within_safe_workspace(const Point2D & point, const GeometryParams & params) const {
    if (!within_closed_interval(point.x, params.safe_x_min, params.safe_x_max) ||
        !within_closed_interval(point.y, params.safe_y_min, params.safe_y_max)) {
      return false;
    }
    const double left_dist = executor_distance_xy(point, params.anchor_left);
    const double right_dist = executor_distance_xy(point, params.anchor_right);
    return left_dist >= params.corner_keepout_radius &&
           right_dist >= params.corner_keepout_radius;
  }

  bool point_valid_for_pen_motion(const Point2D & point, const GeometryParams & params) const {
    return finite_point(point) &&
           point_within_writable(point, params) &&
           point_keeps_body_on_board(point, params) &&
           point_within_body_safe_workspace(point, params) &&
           point_within_safe_workspace(point, params);
  }

  geometry::SamplePolicy execution_sampling_policy(const GeometryParams & params) const {
    const bool text_mode = active_mode_ == "text";
    const double draw_step = text_mode ? params.text_draw_resample_step_m : params.draw_resample_step_m;
    const double travel_step = text_mode ? params.text_travel_resample_step_m : params.travel_resample_step_m;

    geometry::SamplePolicy policy;
    policy.curve_tolerance_m = draw_step;
    policy.min_step_m = std::min(draw_step, travel_step);
    policy.draw_step_m = draw_step;
    policy.travel_step_m = travel_step;
    policy.max_heading_delta_rad = 0.16;
    return policy;
  }

  double hygiene_segment_length_m(const GeometryParams & params) const {
    return std::max(
      1.0e-6,
      0.25 * std::min(params.draw_resample_step_m, params.travel_resample_step_m));
  }

  double polyline_length(const std::vector<Point2D> & points) const {
    double total = 0.0;
    if (points.size() < 2U) {
      return total;
    }
    for (std::size_t index = 1; index < points.size(); ++index) {
      total += executor_distance_xy(points[index - 1], points[index]);
    }
    return total;
  }

  void append_scheduled_path(
    std::vector<ScheduledPath> * paths,
    std::vector<Point2D> points,
    const bool draw,
    const int32_t segment_index,
    const double min_segment_length_m) const
  {
    points = geometry::optimize_polyline(points, min_segment_length_m);
    if (points.size() < 2U) {
      return;
    }

    if (!paths->empty()) {
      auto & last = paths->back();
      if (last.draw == draw &&
          last.segment_index == segment_index &&
          executor_approximately_equal(last.points.back(), points.front(), min_segment_length_m)) {
        for (std::size_t index = 1; index < points.size(); ++index) {
          if (!executor_approximately_equal(last.points.back(), points[index], min_segment_length_m)) {
            last.points.push_back(points[index]);
          }
        }
        last.points = geometry::optimize_polyline(last.points, min_segment_length_m);
        return;
      }
    }

    paths->push_back(ScheduledPath{draw, segment_index, std::move(points)});
  }

  void append_optional_completion_park(
    const Point2D & cursor,
    const GeometryParams & params,
    const int32_t segment_index,
    const geometry::SamplePolicy & policy,
    std::vector<ScheduledPath> * paths) const
  {
    if (active_mode_ != "text" && active_mode_ != "draw") {
      return;
    }

    const Point2D park_point{
      params.body_safe_safe_x_min,
      params.body_safe_safe_y_max,
    };
    if (!point_valid_for_pen_motion(park_point, params) ||
        executor_approximately_equal(cursor, park_point)) {
      return;
    }

    append_scheduled_path(
      paths,
      geometry::sample_polyline(std::vector<Point2D>{cursor, park_point}, false, policy),
      false,
      segment_index,
      hygiene_segment_length_m(params));
  }

  Point2D pen_to_carriage_center(const Point2D & pen_point, const GeometryParams & params) const {
    return Point2D{
      pen_point.x - params.pen_offset.x,
      pen_point.y - params.pen_offset.y,
    };
  }

  wall_climber_interfaces::msg::CableSetpoint make_setpoint(
    const Point2D & pen_point,
    const bool pen_down,
    const int32_t segment_index,
    const double progress,
    const GeometryParams & params) const {
    const Point2D carriage_center = pen_to_carriage_center(pen_point, params);
    const Point2D attach_left{
      carriage_center.x + params.attach_left.x,
      carriage_center.y + params.attach_left.y,
    };
    const Point2D attach_right{
      carriage_center.x + params.attach_right.x,
      carriage_center.y + params.attach_right.y,
    };
    wall_climber_interfaces::msg::CableSetpoint setpoint;
    setpoint.carriage_pose.x = carriage_center.x;
    setpoint.carriage_pose.y = carriage_center.y;
    setpoint.carriage_pose.theta = params.fixed_theta_rad;
    setpoint.left_cable_length = executor_distance_xy(params.anchor_left, attach_left);
    setpoint.right_cable_length = executor_distance_xy(params.anchor_right, attach_right);
    setpoint.pen_down = pen_down;
    setpoint.active_segment_index = segment_index;
    setpoint.progress = progress;
    return setpoint;
  }

  void append_state_sample(
    const Point2D & point,
    const bool pen_down,
    const int32_t segment_index,
    std::vector<Point2D> * out_points,
    std::vector<bool> * out_pen_down,
    std::vector<int32_t> * out_segment_indices) const {
    if (!out_points->empty() &&
        executor_approximately_equal(out_points->back(), point) &&
        out_pen_down->back() == pen_down &&
        out_segment_indices->back() == segment_index) {
      return;
    }
    out_points->push_back(point);
    out_pen_down->push_back(pen_down);
    out_segment_indices->push_back(segment_index);
  }

  void reset_pending_execution_state() {
    schedule_.clear();
    pending_paths_.clear();
    pending_total_length_m_ = 1.0;
    pending_traversed_length_m_ = 0.0;
    has_pending_last_state_ = false;
  }

  bool append_chunk_sample(
    const Point2D & point,
    const bool pen_down,
    const int32_t segment_index,
    const GeometryParams & params,
    std::deque<PlannedSample> * schedule,
    std::string * failure)
  {
    if (has_pending_last_state_ &&
        executor_approximately_equal(pending_last_point_, point) &&
        pending_last_pen_down_ == pen_down &&
        pending_last_segment_index_ == segment_index) {
      return true;
    }
    if (!finite_point(point)) {
      *failure = "path plan contains non-finite sampled coordinates";
      return false;
    }
    if (!point_within_writable(point, params)) {
      *failure = "path plan extends outside writable board bounds";
      return false;
    }
    if (!point_within_body_safe_workspace(point, params)) {
      *failure = "execution plan would move the carriage body outside the board frame";
      return false;
    }
    if (!point_within_safe_workspace(point, params)) {
      *failure = "execution plan exits the configured safe cable workspace";
      return false;
    }
    if (has_pending_last_state_) {
      pending_traversed_length_m_ += executor_distance_xy(pending_last_point_, point);
    }
    PlannedSample sample;
    sample.setpoint = make_setpoint(
      point,
      pen_down,
      segment_index,
      std::min(1.0, pending_traversed_length_m_ / pending_total_length_m_),
      params);
    if (!std::isfinite(sample.setpoint.left_cable_length) ||
        !std::isfinite(sample.setpoint.right_cable_length)) {
      *failure = "failed to compute finite cable lengths";
      return false;
    }
    schedule->push_back(sample);
    pending_last_point_ = point;
    pending_last_pen_down_ = pen_down;
    pending_last_segment_index_ = segment_index;
    has_pending_last_state_ = true;
    return true;
  }

  bool build_next_schedule_chunk(const GeometryParams & params, std::string * failure) {
    schedule_.clear();
    if (pending_paths_.empty()) {
      return false;
    }

    const int settle_samples = std::max(
      1,
      static_cast<int>(std::ceil(params.pen_down_settle_sec / read_publish_period_sec())));
    std::size_t consumed_paths = 0U;
    std::size_t emitted_samples = 0U;

    while (!pending_paths_.empty()) {
      const auto path = pending_paths_.front();
      if (path.points.size() < 2U) {
        pending_paths_.pop_front();
        continue;
      }
      const std::size_t estimated_path_samples = path.points.size() + (path.draw ? settle_samples : 0U);
      if (consumed_paths > 0U &&
          (consumed_paths >= params.chunk_max_paths ||
           emitted_samples + estimated_path_samples > params.chunk_max_samples)) {
        break;
      }

      pending_paths_.pop_front();
      const std::size_t before_size = schedule_.size();
      if (path.draw) {
        for (int settle_index = 0; settle_index < settle_samples; ++settle_index) {
          if (!append_chunk_sample(
                path.points.front(),
                true,
                path.segment_index,
                params,
                &schedule_,
                failure)) {
            schedule_.clear();
            return false;
          }
        }
      }
      for (const auto & point : path.points) {
        if (!append_chunk_sample(
              point,
              path.draw,
              path.segment_index,
              params,
              &schedule_,
              failure)) {
          schedule_.clear();
          return false;
        }
      }
      emitted_samples += schedule_.size() - before_size;
      consumed_paths += 1U;
    }

    if (schedule_.empty()) {
      *failure = "execution plan produced no executable samples";
      return false;
    }

    if (pending_paths_.empty()) {
      PlannedSample final_lift;
      final_lift.setpoint = schedule_.back().setpoint;
      final_lift.setpoint.pen_down = false;
      final_lift.setpoint.progress = 1.0;
      schedule_.push_back(final_lift);
    }
    return true;
  }

  bool build_schedule_from_path_plan(
    const geometry::PathPlan & plan,
    std::string * failure,
    ExecutionBuildDiagnostics * diagnostics) {
    reset_pending_execution_state();
    const GeometryParams params = read_geometry_params();
    if (plan.frame != "board") {
      *failure = "path plan frame must be 'board'";
      return false;
    }
    if (plan.commands.empty()) {
      *failure = "path plan must contain at least one command";
      return false;
    }

    const geometry::SamplePolicy sampling_policy = execution_sampling_policy(params);
    const auto sampled_plan_paths = geometry::sampled_paths_from_plan(plan, sampling_policy);
    if (sampled_plan_paths.empty()) {
      *failure = "path plan produced no sampled paths";
      return false;
    }
    if (diagnostics != nullptr) {
      diagnostics->primitive_count = plan.commands.size();
      diagnostics->sampled_path_count = sampled_plan_paths.size();
    }
    std::vector<ScheduledPath> sampled_paths;
    Point2D cursor = current_pen_point_;
    if (!finite_point(cursor)) {
      cursor = params.initial_pen;
    }
    const int settle_samples = std::max(
      1,
      static_cast<int>(std::ceil(params.pen_down_settle_sec / read_publish_period_sec())));

    for (std::size_t path_index = 0; path_index < sampled_plan_paths.size(); ++path_index) {
      const auto & sampled_path = sampled_plan_paths[path_index];
      if (sampled_path.points.size() < 2U) {
        continue;
      }
      const Point2D segment_start = sampled_path.points.front();
      if (!executor_approximately_equal(cursor, segment_start)) {
        append_scheduled_path(
          &sampled_paths,
          geometry::sample_polyline(
            std::vector<Point2D>{cursor, segment_start},
            false,
            sampling_policy),
          false,
          static_cast<int32_t>(path_index),
          hygiene_segment_length_m(params));
      }
      append_scheduled_path(
        &sampled_paths,
        sampled_path.points,
        sampled_path.draw,
        static_cast<int32_t>(path_index),
        hygiene_segment_length_m(params));
      cursor = sampled_path.points.back();
    }

    append_optional_completion_park(
      cursor,
      params,
      static_cast<int32_t>(sampled_plan_paths.size()),
      sampling_policy,
      &sampled_paths);

    double total_length = 0.0;
    std::size_t estimated_schedule_count = 0U;
    std::size_t estimated_chunk_count = 0U;
    std::size_t chunk_path_count = 0U;
    std::size_t chunk_sample_count = 0U;
    for (const auto & path : sampled_paths) {
      if (path.points.size() < 2U) {
        continue;
      }
      total_length += polyline_length(path.points);
      estimated_schedule_count += path.points.size() + (path.draw ? settle_samples : 0U);
      const std::size_t path_budget = path.points.size() + (path.draw ? settle_samples : 0U);
      if (chunk_path_count > 0U &&
          (chunk_path_count >= params.chunk_max_paths ||
           chunk_sample_count + path_budget > params.chunk_max_samples)) {
        estimated_chunk_count += 1U;
        chunk_path_count = 0U;
        chunk_sample_count = 0U;
      }
      chunk_path_count += 1U;
      chunk_sample_count += path_budget;
    }

    if (sampled_paths.empty()) {
      *failure = "execution plan produced no executable samples";
      return false;
    }
    estimated_schedule_count += 1U;
    if (chunk_path_count > 0U) {
      estimated_chunk_count += 1U;
    }
    if (diagnostics != nullptr) {
      diagnostics->sampled_point_count = estimated_schedule_count;
      diagnostics->schedule_count = estimated_schedule_count;
      diagnostics->chunk_count = estimated_chunk_count;
    }

    pending_total_length_m_ = total_length <= 1.0e-9 ? 1.0 : total_length;
    pending_traversed_length_m_ = 0.0;
    has_pending_last_state_ = false;
    pending_paths_ = std::deque<ScheduledPath>(sampled_paths.begin(), sampled_paths.end());
    return build_next_schedule_chunk(params, failure);
  }

  void active_mode_callback(const std_msgs::msg::String::SharedPtr msg) {
    if (!msg) {
      return;
    }
    active_mode_ = msg->data;
  }

  void primitive_path_plan_callback(
    const wall_climber_interfaces::msg::PrimitivePathPlan::SharedPtr msg)
  {
    if (!msg) {
      return;
    }
    if (status_ == "running") {
      RCLCPP_WARN(
        get_logger(),
        "Ignoring incoming primitive path plan because the cable executor is busy.");
      return;
    }

    std::string failure;
    ExecutionBuildDiagnostics diagnostics;
    diagnostics.chosen_transport = "primitive_path_plan";
    try {
      const auto parse_start = std::chrono::steady_clock::now();
      const auto path_plan = wall_climber::transport::primitive_path_plan_to_path_plan(*msg);
      const auto parse_end = std::chrono::steady_clock::now();
      diagnostics.transport_parse_ms = std::chrono::duration<double, std::milli>(
        parse_end - parse_start).count();
      const auto build_start = std::chrono::steady_clock::now();
      if (!build_schedule_from_path_plan(path_plan, &failure, &diagnostics)) {
        reset_pending_execution_state();
        set_status("error");
        RCLCPP_ERROR(get_logger(), "Rejected primitive path plan: %s", failure.c_str());
        return;
      }
      const auto build_end = std::chrono::steady_clock::now();
      diagnostics.schedule_build_ms = std::chrono::duration<double, std::milli>(
        build_end - build_start).count();
    } catch (const std::exception & exc) {
      failure = exc.what();
      reset_pending_execution_state();
      set_status("error");
      RCLCPP_ERROR(get_logger(), "Rejected primitive path plan: %s", failure.c_str());
      return;
    }

    set_status("running");
    publish_execution_diagnostics(diagnostics);
    RCLCPP_INFO(
      get_logger(),
      "Accepted primitive path plan with %zu total samples across %zu primitives in %zu chunks.",
      diagnostics.schedule_count,
      msg->primitives.size(),
      diagnostics.chunk_count);
  }

  void on_timer() {
    if (schedule_.empty()) {
      if (!pending_paths_.empty()) {
        std::string failure;
        if (!build_next_schedule_chunk(read_geometry_params(), &failure)) {
          reset_pending_execution_state();
          set_status("error");
          RCLCPP_ERROR(get_logger(), "Failed to build execution chunk: %s", failure.c_str());
          return;
        }
      }
    }
    if (schedule_.empty()) {
      if (status_ == "running") {
        set_status("done");
      }
      return;
    }

    const auto next = schedule_.front().setpoint;
    current_pen_point_ = Point2D{
      next.carriage_pose.x + get_parameter("pen_offset_x").as_double(),
      next.carriage_pose.y + get_parameter("pen_offset_y").as_double(),
    };
    setpoint_pub_->publish(next);
    schedule_.pop_front();
    if (schedule_.empty() && pending_paths_.empty()) {
      set_status("done");
    }
  }

  void set_status(const std::string & status) {
    if (status_ == status) {
      return;
    }
    status_ = status;
    std_msgs::msg::String msg;
    msg.data = status;
    status_pub_->publish(msg);
  }

  void publish_execution_diagnostics(const ExecutionBuildDiagnostics & diagnostics) {
    std::ostringstream payload;
    payload
      << "{"
      << "\"chosen_transport\":\"" << diagnostics.chosen_transport << "\","
      << "\"transport_parse_ms\":" << diagnostics.transport_parse_ms << ","
      << "\"schedule_build_ms\":" << diagnostics.schedule_build_ms << ","
      << "\"primitive_count\":" << diagnostics.primitive_count << ","
      << "\"sampled_path_count\":" << diagnostics.sampled_path_count << ","
      << "\"sampled_point_count\":" << diagnostics.sampled_point_count << ","
      << "\"schedule_count\":" << diagnostics.schedule_count << ","
      << "\"chunk_count\":" << diagnostics.chunk_count
      << "}";
    std_msgs::msg::String msg;
    msg.data = payload.str();
    diagnostics_pub_->publish(msg);
  }

  rclcpp::Publisher<wall_climber_interfaces::msg::CableSetpoint>::SharedPtr setpoint_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr diagnostics_pub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr active_mode_sub_;
  rclcpp::Subscription<wall_climber_interfaces::msg::PrimitivePathPlan>::SharedPtr primitive_plan_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::deque<PlannedSample> schedule_;
  std::deque<ScheduledPath> pending_paths_;
  double pending_total_length_m_{1.0};
  double pending_traversed_length_m_{0.0};
  bool has_pending_last_state_{false};
  Point2D pending_last_point_;
  bool pending_last_pen_down_{false};
  int32_t pending_last_segment_index_{-1};
  Point2D current_pen_point_;
  std::string status_;
  std::string active_mode_{"off"};
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CableDrawExecutor>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
