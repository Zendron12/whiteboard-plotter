#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "geometry_msgs/msg/pose2_d.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/string.hpp"
#include "wall_climber_interfaces/msg/cable_setpoint.hpp"
#include "wall_climber_interfaces/msg/draw_plan.hpp"
#include "wall_climber_interfaces/msg/draw_polyline.hpp"

namespace {

rclcpp::QoS transient_local_qos() {
  return rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
}

struct Point2D {
  double x{0.0};
  double y{0.0};
};

struct PlannedSample {
  wall_climber_interfaces::msg::CableSetpoint setpoint;
};

double distance_xy(const Point2D & a, const Point2D & b) {
  return std::hypot(a.x - b.x, a.y - b.y);
}

bool finite_point(const Point2D & point) {
  return std::isfinite(point.x) && std::isfinite(point.y);
}

Point2D to_point(const geometry_msgs::msg::Point & point) {
  return Point2D{point.x, point.y};
}

bool approximately_equal(const Point2D & a, const Point2D & b, const double eps = 1.0e-9) {
  return std::abs(a.x - b.x) <= eps && std::abs(a.y - b.y) <= eps;
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
    declare_parameter("draw_resample_step_m", 0.012);
    declare_parameter("travel_resample_step_m", 0.025);
    declare_parameter("publish_period_sec", 0.05);
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
    active_mode_sub_ = create_subscription<std_msgs::msg::String>(
      "/wall_climber/internal/active_mode",
      transient_local_qos(),
      std::bind(&CableDrawExecutor::active_mode_callback, this, std::placeholders::_1));
    pen_attached_sub_ = create_subscription<std_msgs::msg::Bool>(
      "/wall_climber/pen_attached",
      transient_local_qos(),
      std::bind(&CableDrawExecutor::pen_attached_callback, this, std::placeholders::_1));
    draw_plan_sub_ = create_subscription<wall_climber_interfaces::msg::DrawPlan>(
      "/wall_climber/draw_plan",
      10,
      std::bind(&CableDrawExecutor::draw_plan_callback, this, std::placeholders::_1));

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
    };
  }

  double read_publish_period_sec() const {
    return std::max(1.0e-3, get_parameter("publish_period_sec").as_double());
  }

  bool point_within_writable(const Point2D & point, const GeometryParams & params) const {
    return point.x >= params.writable_x_min &&
           point.x <= params.writable_x_max &&
           point.y >= params.writable_y_min &&
           point.y <= params.writable_y_max;
  }

  bool point_keeps_body_on_board(const Point2D & point, const GeometryParams & params) const {
    return point.x >= params.body_safe_writable_x_min &&
           point.x <= params.body_safe_writable_x_max &&
           point.y >= params.body_safe_writable_y_min &&
           point.y <= params.body_safe_writable_y_max;
  }

  bool point_within_body_safe_workspace(const Point2D & point, const GeometryParams & params) const {
    return point.x >= params.body_safe_safe_x_min &&
           point.x <= params.body_safe_safe_x_max &&
           point.y >= params.body_safe_safe_y_min &&
           point.y <= params.body_safe_safe_y_max;
  }

  bool point_within_safe_workspace(const Point2D & point, const GeometryParams & params) const {
    if (point.x < params.safe_x_min || point.x > params.safe_x_max ||
        point.y < params.safe_y_min || point.y > params.safe_y_max) {
      return false;
    }
    const double left_dist = distance_xy(point, params.anchor_left);
    const double right_dist = distance_xy(point, params.anchor_right);
    return left_dist >= params.corner_keepout_radius &&
           right_dist >= params.corner_keepout_radius;
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
    setpoint.left_cable_length = distance_xy(params.anchor_left, attach_left);
    setpoint.right_cable_length = distance_xy(params.anchor_right, attach_right);
    setpoint.pen_down = pen_down;
    setpoint.active_segment_index = segment_index;
    setpoint.progress = progress;
    return setpoint;
  }

  void append_resampled_segment(
    const std::vector<Point2D> & points,
    const bool pen_down,
    const int32_t segment_index,
    const double step_m,
    std::vector<Point2D> * out_points,
    std::vector<bool> * out_pen_down,
    std::vector<int32_t> * out_segment_indices,
    double * total_length) const {
    if (points.size() < 2) {
      return;
    }
    if (out_points->empty() || !approximately_equal(out_points->back(), points.front())) {
      out_points->push_back(points.front());
      out_pen_down->push_back(pen_down);
      out_segment_indices->push_back(segment_index);
    }
    for (std::size_t index = 1; index < points.size(); ++index) {
      const Point2D start = points[index - 1];
      const Point2D end = points[index];
      const double segment_length = distance_xy(start, end);
      *total_length += segment_length;
      if (segment_length <= 1.0e-9) {
        continue;
      }
      const int subdivisions = std::max(1, static_cast<int>(std::ceil(segment_length / step_m)));
      for (int step_index = 1; step_index <= subdivisions; ++step_index) {
        const double t = static_cast<double>(step_index) / static_cast<double>(subdivisions);
        Point2D sample{
          start.x + (end.x - start.x) * t,
          start.y + (end.y - start.y) * t,
        };
        if (!out_points->empty() && approximately_equal(out_points->back(), sample)) {
          continue;
        }
        out_points->push_back(sample);
        out_pen_down->push_back(pen_down);
        out_segment_indices->push_back(segment_index);
      }
    }
  }

  void append_state_sample(
    const Point2D & point,
    const bool pen_down,
    const int32_t segment_index,
    std::vector<Point2D> * out_points,
    std::vector<bool> * out_pen_down,
    std::vector<int32_t> * out_segment_indices) const {
    if (!out_points->empty() &&
        approximately_equal(out_points->back(), point) &&
        out_pen_down->back() == pen_down &&
        out_segment_indices->back() == segment_index) {
      return;
    }
    out_points->push_back(point);
    out_pen_down->push_back(pen_down);
    out_segment_indices->push_back(segment_index);
  }

  bool build_schedule(
    const wall_climber_interfaces::msg::DrawPlan & plan,
    std::deque<PlannedSample> * schedule,
    std::string * failure) const {
    const GeometryParams params = read_geometry_params();
    if (plan.frame_id != "board") {
      *failure = "draw plan frame_id must be 'board'";
      return false;
    }
    if (plan.segments.empty()) {
      *failure = "draw plan must contain at least one segment";
      return false;
    }

    std::vector<Point2D> sampled_points;
    std::vector<bool> sampled_pen_down;
    std::vector<int32_t> sampled_segment_indices;
    double total_length = 0.0;
    Point2D cursor = current_pen_point_;
    if (!finite_point(cursor)) {
      cursor = params.initial_pen;
    }
    const int settle_samples = std::max(
      1,
      static_cast<int>(std::ceil(params.pen_down_settle_sec / read_publish_period_sec())));

    for (std::size_t segment_index = 0; segment_index < plan.segments.size(); ++segment_index) {
      const auto & segment = plan.segments[segment_index];
      if (segment.points.size() < 2) {
        *failure = "draw plan contains a degenerate segment";
        return false;
      }
      std::vector<Point2D> raw_points;
      raw_points.reserve(segment.points.size());
      for (const auto & point : segment.points) {
        Point2D candidate = to_point(point);
        if (!finite_point(candidate)) {
          *failure = "draw plan contains non-finite coordinates";
          return false;
        }
        if (!point_within_writable(candidate, params)) {
          *failure = "draw plan extends outside writable board bounds";
          return false;
        }
        if (!point_keeps_body_on_board(candidate, params)) {
          *failure = "draw plan would move the carriage body outside the board frame";
          return false;
        }
        raw_points.push_back(candidate);
      }
      const Point2D segment_start = raw_points.front();
      if (!approximately_equal(cursor, segment_start)) {
        std::vector<Point2D> travel_points{cursor, segment_start};
        append_resampled_segment(
          travel_points,
          false,
          static_cast<int32_t>(segment_index),
          params.travel_resample_step_m,
          &sampled_points,
          &sampled_pen_down,
          &sampled_segment_indices,
          &total_length);
      }
      if (segment.draw) {
        for (int settle_index = 0; settle_index < settle_samples; ++settle_index) {
          append_state_sample(
            segment_start,
            true,
            static_cast<int32_t>(segment_index),
            &sampled_points,
            &sampled_pen_down,
            &sampled_segment_indices);
        }
      }
      append_resampled_segment(
        raw_points,
        segment.draw,
        static_cast<int32_t>(segment_index),
        segment.draw ? params.draw_resample_step_m : params.travel_resample_step_m,
        &sampled_points,
        &sampled_pen_down,
        &sampled_segment_indices,
        &total_length);
      cursor = raw_points.back();
    }

    // After text execution, park near the lower-left safe corner with pen lifted.
    if (active_mode_ == "text") {
      const Point2D park_point{params.body_safe_safe_x_min, params.body_safe_safe_y_max};
      if (finite_point(park_point) &&
          point_keeps_body_on_board(park_point, params) &&
          point_within_safe_workspace(park_point, params) &&
          !approximately_equal(cursor, park_point)) {
        const Point2D park_waypoint{cursor.x, params.body_safe_safe_y_max};
        const int32_t park_segment_index = static_cast<int32_t>(plan.segments.size());
        if (!approximately_equal(cursor, park_waypoint) &&
            point_keeps_body_on_board(park_waypoint, params) &&
            point_within_safe_workspace(park_waypoint, params)) {
          std::vector<Point2D> travel_down{cursor, park_waypoint};
          append_resampled_segment(
            travel_down,
            false,
            park_segment_index,
            params.travel_resample_step_m,
            &sampled_points,
            &sampled_pen_down,
            &sampled_segment_indices,
            &total_length);
          cursor = park_waypoint;
        }
        if (!approximately_equal(cursor, park_point)) {
          std::vector<Point2D> travel_left{cursor, park_point};
          append_resampled_segment(
            travel_left,
            false,
            park_segment_index,
            params.travel_resample_step_m,
            &sampled_points,
            &sampled_pen_down,
            &sampled_segment_indices,
            &total_length);
        }
      }
    }

    if (sampled_points.empty()) {
      *failure = "draw plan produced no executable samples";
      return false;
    }

    if (total_length <= 1.0e-9) {
      total_length = 1.0;
    }

    double traversed_length = 0.0;
    for (std::size_t index = 0; index < sampled_points.size(); ++index) {
      if (!point_within_body_safe_workspace(sampled_points[index], params)) {
        *failure = "draw plan would move the carriage body outside the board frame";
        return false;
      }
      if (!point_within_safe_workspace(sampled_points[index], params)) {
        *failure = "draw plan exits the configured safe cable workspace";
        return false;
      }
      if (index > 0) {
        traversed_length += distance_xy(sampled_points[index - 1], sampled_points[index]);
      }
      PlannedSample sample;
      sample.setpoint = make_setpoint(
        sampled_points[index],
        sampled_pen_down[index],
        sampled_segment_indices[index],
        std::min(1.0, traversed_length / total_length),
        params);
      if (!std::isfinite(sample.setpoint.left_cable_length) ||
          !std::isfinite(sample.setpoint.right_cable_length)) {
        *failure = "failed to compute finite cable lengths";
        return false;
      }
      schedule->push_back(sample);
    }

    PlannedSample final_lift;
    final_lift.setpoint = schedule->back().setpoint;
    final_lift.setpoint.pen_down = false;
    final_lift.setpoint.progress = 1.0;
    schedule->push_back(final_lift);
    return true;
  }

  void active_mode_callback(const std_msgs::msg::String::SharedPtr msg) {
    if (!msg) {
      return;
    }
    active_mode_ = msg->data;
  }

  void pen_attached_callback(const std_msgs::msg::Bool::SharedPtr msg) {
    if (!msg) {
      return;
    }
    pen_attached_ = msg->data;
  }

  void draw_plan_callback(const wall_climber_interfaces::msg::DrawPlan::SharedPtr msg) {
    if (!msg) {
      return;
    }
    if (status_ == "running") {
      RCLCPP_WARN(get_logger(), "Ignoring incoming draw plan because the cable executor is busy.");
      return;
    }
    if (!pen_attached_) {
      set_status("error");
      RCLCPP_ERROR(get_logger(), "Rejected draw plan: robot has no pen attached.");
      return;
    }

    std::deque<PlannedSample> next_schedule;
    std::string failure;
    if (!build_schedule(*msg, &next_schedule, &failure)) {
      set_status("error");
      RCLCPP_ERROR(get_logger(), "Rejected draw plan: %s", failure.c_str());
      return;
    }

    schedule_ = std::move(next_schedule);
    set_status("running");
    RCLCPP_INFO(
      get_logger(),
      "Accepted draw plan with %zu samples across %zu segments.",
      schedule_.size(),
      msg->segments.size());
  }

  void on_timer() {
    if (!pen_attached_) {
      if (!schedule_.empty()) {
        schedule_.clear();
        set_status("error");
        RCLCPP_ERROR(get_logger(), "Stopped execution because the pen is no longer attached.");
      }
      return;
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
    if (schedule_.empty()) {
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

  rclcpp::Publisher<wall_climber_interfaces::msg::CableSetpoint>::SharedPtr setpoint_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr active_mode_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr pen_attached_sub_;
  rclcpp::Subscription<wall_climber_interfaces::msg::DrawPlan>::SharedPtr draw_plan_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::deque<PlannedSample> schedule_;
  Point2D current_pen_point_;
  std::string status_;
  std::string active_mode_{"off"};
  bool pen_attached_{true};
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CableDrawExecutor>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
