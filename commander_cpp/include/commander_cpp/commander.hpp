#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <unistd.h>
#include <cmath>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include "tf2/exceptions.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"

#include <kdl/frames.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <geometry_msgs/msg/pose.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <carla_msgs/msg/carla_ego_vehicle_control.hpp>
#include <carla_msgs/msg/carla_ego_vehicle_status.hpp>

class VehicleCommander : public rclcpp::Node
{
public:
    /// Constructor
    VehicleCommander();

    ~VehicleCommander();


private:

    // Callback Groups
    rclcpp::CallbackGroup::SharedPtr service_cb_group_;
    rclcpp::CallbackGroup::SharedPtr topic_cb_group_;

    // Sensor Callbacks
    bool kts1_camera_received_data = false;
    bool kts2_camera_received_data = false;
    bool left_bins_camera_received_data = false;
    bool right_bins_camera_received_data = false;

    rclcpp::Subscription<carla_msgs::msg::CarlaEgoVehicleStatus>::SharedPtr ego_vehicle_status_sub_;
    rclcpp::Publisher<carla_msgs::msg::CarlaEgoVehicleControl>::SharedPtr my_publisher_;
    void VehicleStatusCallback(const carla_msgs::msg::CarlaEgoVehicleStatus::ConstSharedPtr msg);
    // void kts2_camera_cb(const ariac_msgs::msg::AdvancedLogicalCameraImage::ConstSharedPtr msg);
    // void left_bins_camera_cb(const ariac_msgs::msg::AdvancedLogicalCameraImage::ConstSharedPtr msg);
    // void right_bins_camera_cb(const ariac_msgs::msg::AdvancedLogicalCameraImage::ConstSharedPtr msg);
};
