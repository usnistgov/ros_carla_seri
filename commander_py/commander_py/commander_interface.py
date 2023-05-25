import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from carla_msgs.msg import CarlaEgoVehicleStatus, CarlaEgoVehicleControl
import pygame
import carla


class VehicleCommanderInterface(Node):
    '''
    Class for a vehicle commander node.

    Args:
        Node (rclpy.node.Node): Parent class for ROS nodes

    Raises:
        KeyboardInterrupt: Exception raised when the user uses Ctrl+C to kill a process
    '''

    def __init__(self):
        super().__init__('vehicle_commander')

        print("\N{cat} VehicleCommanderInterface Node has been initialised.")

        timer_group = MutuallyExclusiveCallbackGroup()
        subscription_group = MutuallyExclusiveCallbackGroup()

        # Flag to indicate if the kit has been completed
        self._kit_completed = False
        self._competition_started = False
        self._competition_state = None

        # # subscriber
        # self.create_subscription(CarlaEgoVehicleStatus, '/carla/ego_vehicle/vehicle_status',
        #                          self._vehicle_status_cb, 10, callback_group=subscription_group)
        
        # self.cmd_publisher = self.create_publisher(CarlaEgoVehicleControl, '/carla/ego_vehicle/vehicle_control_cmd', 10)

        # # timer
        # self._vehicle_action_timer = self.create_timer(1, self._vehicle_action_timer_callback,
                                                    #  callback_group=timer_group)
        

    def _vehicle_status_cb(self, msg: CarlaEgoVehicleStatus):
        '''
        /carla/ego_vehicle/vehicle_status topic callback function

        Args:
            msg (CarlaEgoVehicleStatus): CarlaEgoVehicleStatus message
        '''
        self.get_logger().info(f'Velocity: {msg.velocity}')
        self.get_logger().info(f'Throttle: {msg.control.throttle}')
        self.get_logger().info(f'Steer: {msg.control.steer}')
        self.get_logger().info(f'Break: {msg.control.brake}')

    def _vehicle_action_timer_callback(self):
        '''
        Callback for the timer 
        '''
        # self.get_logger().info(f'\N{dog} {pygame.time.Clock()}')
        # self.get_logger().info(f'\N{dog} {carla.libcarla.}')

        
        msg = CarlaEgoVehicleControl()
        msg.gear = 1
        msg.throttle = 0.3
        msg.reverse = False
        self.get_logger().info(f'\N{dog} Publishing throttle {msg.throttle}')
        self.cmd_publisher.publish(msg)
        