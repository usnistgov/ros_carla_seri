import rclpy
import yaml
import os
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from carla_msgs.msg import CarlaEgoVehicleStatus, CarlaEgoVehicleControl
from std_msgs.msg import Bool
from launch_ros.substitutions import FindPackageShare
from commander_py import controller2d
import pygame
import numpy as np
import csv

import carla


class VehicleCommanderInterface(Node):
    '''
    Class for a vehicle commander node.

    Args:
        Node (rclpy.node.Node): Parent class for ROS nodes

    Raises:
        KeyboardInterrupt: Exception raised when the user uses Ctrl+C to kill a process
    '''

    STOP_SIGN_FENCELENGTH = 5.0  # meters

    def __init__(self):
        super().__init__('vehicle_commander')

        print("\N{cat} VehicleCommanderInterface Node has been initialised.")

        # Callback groups
        timer_group = MutuallyExclusiveCallbackGroup()
        # subscription_group = MutuallyExclusiveCallbackGroup()

        # seri.yaml
        pkg_share = FindPackageShare(package='carla_common').find('carla_common')
        config_file_name = "seri.yaml"
        config_file_path = os.path.join(pkg_share, 'config', config_file_name)
        self._yaml_data = self.read_yaml(config_file_path)

        # waypoints.txt
        waypoints_file_name = "waypoints.txt"
        self._waypoints_file_path = os.path.join(pkg_share, 'config', waypoints_file_name)

        #############################################
        # stop sign fences
        #############################################
        self._stopsign_fences = []     # [x0, y0, x1, y1]
        self._get_stopsign_fences()

        #############################################
        # parked vehicle boxes
        #############################################
        self._parked_vehicle_box_pts = []      # [x,y]
        self._get_parked_vehicle_boxes()

        #############################################
        # waypoints
        #############################################
        self._waypoints_np = None
        self._waypoints = None
        self._get_waypoints()
        
        #############################################
        # Controller 2D Class Declaration
        #############################################
        # This is where we take the controller2d.py class
        # and apply it to the simulator
        self._controller = controller2d.Controller2D(waypoints)

        # # subscriber
        # self.create_subscription(CarlaEgoVehicleStatus, '/carla/ego_vehicle/vehicle_status',
        #                          self._vehicle_status_cb, 10, callback_group=subscription_group)

        # publisher
        self._tesla_cmd_publisher = self.create_publisher(
            CarlaEgoVehicleControl, '/carla/ego_vehicle/vehicle_control_cmd', 10)
        # message to publish
        self._tesla_controller = CarlaEgoVehicleControl()

        self._volkswagen_autopilot_publisher = self.create_publisher(
            Bool, '/carla/hero1/enable_autopilot', 10)
        self._nissan_autopilot_publisher = self.create_publisher(
            Bool, '/carla/hero0/enable_autopilot', 10)

        # # timer
        # self._vehicle_action_timer = self.create_timer(1, self._vehicle_action_timer_callback,
        #  callback_group=timer_group)

        self._waypoint_follower_timer = self.create_timer(0.2, self._waypoint_follower_callback,
                                                          callback_group=timer_group)

    def read_yaml(self, path):
        '''
        Method to read a yaml file

        Args:
            path (str): absolute path to the yaml file

        '''
        with open(path, "r") as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError:
                self.get_logger().error("Unable to read configuration file")
                return {}

    def _get_waypoints(self):
        '''
        Get the waypoints from the waypoints file
        '''
        with open(self._waypoints_file_path) as waypoints_file_handle:
            self._waypoints = list(csv.reader(waypoints_file_handle,
                                        delimiter=',',
                                        quoting=csv.QUOTE_NONNUMERIC))
            self._waypoints_np = np.array(self._waypoints)

    def _get_parked_vehicle_boxes(self):
        parked_vehicles = None
        try:
            parked_vehicles = self._yaml_data["parked_vehicle"]
        except KeyError:
            self.get_logger().info(
                "YAML configuration file does not contain a 'parked_vehicle' section")

        if parked_vehicles is not None:
            for parked_vehicle in parked_vehicles:
                yaw = parked_vehicle["spawn_point"]["yaw"]
                yaw = np.deg2rad(yaw)
                x = parked_vehicle["spawn_point"]["x"]
                y = parked_vehicle["spawn_point"]["y"]
                # z = parked_vehicle["spawn_point"]["z"]
                xrad = parked_vehicle["box_radius"]["x"]
                yrad = parked_vehicle["box_radius"]["y"]
                # zrad = parked_vehicle["box_radius"]["z"]
                cpos = np.array([
                    [-xrad, -xrad, -xrad, 0,    xrad, xrad, xrad,  0],
                    [-yrad, 0,     yrad,  yrad, yrad, 0,    -yrad, -yrad]])
                rotyaw = np.array([
                    [np.cos(yaw), np.sin(yaw)],
                    [-np.sin(yaw), np.cos(yaw)]])
                cpos_shift = np.array([
                    [x, x, x, x, x, x, x, x],
                    [y, y, y, y, y, y, y, y]])
                cpos = np.add(np.matmul(rotyaw, cpos), cpos_shift)
                for j in range(cpos.shape[1]):
                    self._parked_vehicle_box_pts.append([cpos[0, j], cpos[1, j]])

    def _get_stopsign_fences(self):
        '''
        Get the stop sign fences from the yaml file
        '''
        stop_signs = None
        try:
            stop_signs = self._yaml_data["stop_sign"]
        except KeyError:
            self.get_logger().info(
                "YAML configuration file does not contain a 'stop_sign' section")

        if stop_signs is not None:
            for stop_sign in stop_signs:
                yaw = stop_sign["location"]["yaw"]
                # convert to radians
                yaw = np.deg2rad(yaw)
                x = stop_sign["location"]["x"]
                y = stop_sign["location"]["y"]
                # z = stop_sign["location"]["z"]
                yaw = yaw + np.pi / 2.0  # add 90 degrees for fence
                spos = np.array([
                    [0, 0],
                    [0, self.STOP_SIGN_FENCELENGTH]])
                rotyaw = np.array([
                    [np.cos(yaw), np.sin(yaw)],
                    [-np.sin(yaw), np.cos(yaw)]])
                spos_shift = np.array([
                    [x, x],
                    [y, y]])
                spos = np.add(np.matmul(rotyaw, spos), spos_shift)
                self._stopsign_fences.append([spos[0, 0], spos[1, 0], spos[0, 1], spos[1, 1]])

    def _waypoint_follower_callback(self):
        pass

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

        self._tesla_controller.gear = 1
        self._tesla_controller.throttle = 0.3
        self._tesla_controller.reverse = False
        self.get_logger().info(f'\N{dog} Publishing throttle {self._tesla_controller.throttle}')
        self._tesla_cmd_publisher.publish(self._tesla_controller)
