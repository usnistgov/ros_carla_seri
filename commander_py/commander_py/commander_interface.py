import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from carla_msgs.msg import CarlaEgoVehicleStatus, CarlaEgoVehicleControl
import pygame
import numpy as np

import carla


class VehicleCommanderInterface(Node):
    '''
    Class for a vehicle commander node.

    Args:
        Node (rclpy.node.Node): Parent class for ROS nodes

    Raises:
        KeyboardInterrupt: Exception raised when the user uses Ctrl+C to kill a process
    '''
    
    # State machine states
    FOLLOW_LANE = 0
    DECELERATE_TO_STOP = 1
    STAY_STOPPED = 2
    # Stop speed threshold
    STOP_THRESHOLD = 0.02
    # Number of cycles before moving from stop sign.
    STOP_COUNTS = 10

    def __init__(self):
        super().__init__('vehicle_commander')

        print("\N{cat} VehicleCommanderInterface Node has been initialised.")

        timer_group = MutuallyExclusiveCallbackGroup()
        subscription_group = MutuallyExclusiveCallbackGroup()

        # Flag to indicate if the kit has been completed
        self._kit_completed = False
        self._competition_started = False
        self._competition_state = None

        self._tesla_controller = CarlaEgoVehicleControl()

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

        self.tesla_controller.gear = 1
        self._tesla_controller.throttle = 0.3
        self._tesla_controller.reverse = False
        self.get_logger().info(f'\N{dog} Publishing throttle {self._tesla_controller.throttle}')
        self.cmd_publisher.publish(self._tesla_controller)
