#include <commander_cpp/commander.hpp>

// ros <----> carla (python) <----> Unreal Engine (C++)
//=================================================
VehicleCommander::VehicleCommander()
    : Node("vehicle_commander")
{
    ego_vehicle_status_sub_ = this->create_subscription<carla_msgs::msg::CarlaEgoVehicleStatus>(
        "/carla/ego_vehicle/vehicle_status",
        rclcpp::QoS(1),
        std::bind(&VehicleCommander::VehicleStatusCallback, this, std::placeholders::_1));

    my_publisher_ = this->create_publisher<carla_msgs::msg::CarlaEgoVehicleControl>(
        "/carla/ego_vehicle/vehicle_control_cmd",
        rclcpp::QoS(1));
}

//=================================================
VehicleCommander::~VehicleCommander()
{
    RCLCPP_INFO(this->get_logger(), "Vehicle Commander Destructor");
}

//=================================================
void VehicleCommander::VehicleStatusCallback(const carla_msgs::msg::CarlaEgoVehicleStatus::ConstSharedPtr msg)
{
    // usleep(100000);
    RCLCPP_INFO(this->get_logger(), "Vehicle Status Callback");
    RCLCPP_INFO_STREAM(this->get_logger(), "-- velocity: " << msg->velocity);
    RCLCPP_INFO_STREAM(this->get_logger(), "-- control.throttle: " << msg->control.throttle);
    RCLCPP_INFO_STREAM(this->get_logger(), "-- control.steer: " << msg->control.steer);
    RCLCPP_INFO_STREAM(this->get_logger(), "-- control.brake: " << msg->control.brake);
}

//=================================================
int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto commander_node = std::make_shared<VehicleCommander>();
    rclcpp::spin(commander_node);

    rclcpp::shutdown();
}