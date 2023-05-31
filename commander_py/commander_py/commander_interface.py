
'''
Node to command two vehicles: A leader and a follower.

- The leader and the follower go through different waypoints. 
    - Waypoints for the leader are located in ros_bridge/carla_common/config/waypoints_leader.txt.
    - Waypoints for the follower are located in ros_bridge/carla_common/config/waypoints_follower.txt.
- Once the follower reaches the leader, it will decrease its speed to keep a safe distance from the leader.
'''

# R0902: too-many-instance-attributes
# R0915: too-many-statements
# pylint: disable=locally-disabled, multiple-statements
# pylint: disable=locally-disabled, fixme
# pylint: disable=locally-disabled, line-too-long
# pylint: disable=locally-disabled, R0902
# pylint: disable=locally-disabled, R0915
# pylint: disable=locally-disabled, R0914
import os
import sys
import csv
import numpy as np
import tf_transformations
# import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from carla_msgs.msg import CarlaEgoVehicleStatus, CarlaEgoVehicleControl
from std_msgs.msg import Bool
from launch_ros.substitutions import FindPackageShare
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry
from commander_py import (
    controller2d,
    local_planner,
    behavioral_planner
)
# import common functions
from commander_py.common import read_yaml
# import pygame
# import carla


# ===============================================================
__author__ = "Zeid Kootbally"
__credits__ = ["Zeid Kootbally"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Zeid Kootbally"
__email__ = "zeid.kootbally@nist.gov"
__status__ = "Development"
# ===============================================================


class VehicleCommanderInterface(Node):
    '''
    Class for a vehicle commander node.

    Args:
        Node (rclpy.node.Node): Parent class for ROS nodes

    Raises:
        KeyboardInterrupt: Exception raised when the user uses Ctrl+C to kill a process

    Attributes:
        _timer_group                Callback group for the timer.
        _subscription_group         Callback group for all subscribers.
        _follower_current_velocity  Current velocity of the follower.
        _follower_current_x         Current x position of the follower.
        _follower_current_y         Current y position of the follower.
        _follower_current_rot_x     Current x rotation of the follower.
        _follower_current_rot_y     Current y rotation of the follower.
        _follower_current_rot_z     Current z rotation of the follower.
        _follower_current_rot_w     Current w rotation of the follower.
        _follower_current_roll      Current roll of the follower.
        _follower_current_pitch     Current pitch of the follower.
        _follower_current_yaw       Current yaw of the follower.
        _waypoints_acquired         Boolean to indicate if waypoints have been acquired from files.
    '''

    STOP_SIGN_FENCELENGTH = 5.0  # meters
    ITER_FOR_SIM_TIMESTEP = 10     # no. iterations to compute approx sim timestep

    # Planning Constants
    NUM_PATHS = 7
    BP_LOOKAHEAD_BASE = 8.0              # m
    BP_LOOKAHEAD_TIME = 2.0              # s
    PATH_OFFSET = 1.5              # m
    CIRCLE_OFFSETS = [-1.0, 1.0, 3.0]  # m
    CIRCLE_RADII = [1.5, 1.5, 1.5]  # m
    TIME_GAP = 1.0              # s
    DIST_THRESHOLD_TO_LAST_WAYPOINT = 2.0
    PATH_SELECT_WEIGHT = 10
    A_MAX = 1.5              # m/s^2
    SLOW_SPEED = 2.0              # m/s
    STOP_LINE_BUFFER = 3.5              # m
    LEAD_VEHICLE_LOOKAHEAD = 20.0             # m
    LP_FREQUENCY_DIVISOR = 2                # Frequency divisor to make the
    WAIT_TIME_BEFORE_START = 1.00   # game seconds (time before controller start)
    TOTAL_RUN_TIME = 1000.00  # game seconds (total runtime before sim end)
    TOTAL_FRAME_BUFFER = 300    # number of frames to buffer after total runtime
    INTERP_LOOKAHEAD_DISTANCE = 20   # lookahead in meters

    # selected path
    INTERP_DISTANCE_RES = 0.01  # distance between interpolated points

    def __init__(self):
        super().__init__('vehicle_commander')

        #############################################
        # Time related variables
        #############################################
        self._current_time = 0.0

        #############################################
        # Callback groups
        #############################################
        _timer_group = MutuallyExclusiveCallbackGroup()
        _subscription_group = MutuallyExclusiveCallbackGroup()

        #############################################
        # Follower
        #############################################

        # Subscribers
        self.create_subscription(CarlaEgoVehicleStatus,
                                 '/carla/ego_vehicle/vehicle_status',
                                 self._follower_status_cb,
                                 10,
                                 callback_group=_subscription_group)

        self.create_subscription(Odometry, '/carla/ego_vehicle/odometry',
                                 self._follower_odometry_cb, 10, callback_group=_subscription_group)

        self._follower_current_velocity = 0
        self._follower_current_x = 0
        self._follower_current_y = 0
        self._follower_current_rot_x = 0
        self._follower_current_rot_y = 0
        self._follower_current_rot_z = 0
        self._follower_current_rot_w = 0
        self._follower_current_roll = 0
        self._follower_current_pitch = 0
        self._follower_current_yaw = 0

        # Publishers
        self._follower_cmd_publisher = self.create_publisher(
            CarlaEgoVehicleControl, '/carla/ego_vehicle/vehicle_control_cmd', 10)
        self._follower_vehicle_control = CarlaEgoVehicleControl()

        #############################################
        # Leader
        #############################################
        self.create_subscription(CarlaEgoVehicleStatus,
                                 '/carla/hero/vehicle_status',
                                 self._leader_status_cb,
                                 10,
                                 callback_group=_subscription_group)

        self.create_subscription(Odometry, '/carla/hero/odometry',
                                 self._leader_odometry_cb, 10, callback_group=_subscription_group)

        self._leader_current_velocity = 0
        self._leader_current_x = 0
        self._leader_current_y = 0
        self._leader_current_rot_x = 0
        self._leader_current_rot_y = 0
        self._leader_current_rot_z = 0
        self._leader_current_rot_w = 0
        self._leader_current_roll = 0
        self._leader_current_pitch = 0
        self._leader_current_yaw = 0

        # Publisher
        self._leader_cmd_publisher = self.create_publisher(
            CarlaEgoVehicleControl, '/carla/hero/vehicle_control_cmd', 10)
        self._leader_vehicle_control = CarlaEgoVehicleControl()

        self._leader_autopilot_publisher = self.create_publisher(
            Bool, '/carla/hero/enable_autopilot', 10)
        self._leader_autopilot_control = Bool()

        #############################################
        # Simulation time
        #############################################
        self.create_subscription(Clock, '/clock',
                                 self._clock_cb, 10, callback_group=_subscription_group)
        self._current_time = 0

        #############################################
        # timer
        #############################################
        # self._vehicle_action_timer = self.create_timer(1, self._vehicle_action_timer_callback,
        #  callback_group=_timer_group)
        self._waypoint_follower_timer = self.create_timer(0.5, self._waypoint_follower_cb,
                                                          callback_group=_timer_group)

        # seri.yaml
        carla_common_pkg = FindPackageShare(package='carla_common').find('carla_common')
        config_file_name = "seri.yaml"
        config_file_path = os.path.join(carla_common_pkg, 'config', config_file_name)
        self._yaml_data = read_yaml(config_file_path)

        #############################################
        # stop sign fences
        #############################################
        self._stopsign_fences = []     # [x0, y0, x1, y1]
        # self._get_stopsign_fences()

        #############################################
        # parked vehicle boxes
        #############################################
        self._parked_vehicle_box_pts = []      # [x,y]
        # self._get_parked_vehicle_boxes()

        #############################################
        # Waypoints
        #############################################
        self._waypoints_acquired = False
        self._leader_csv_file = None
        self._follower_csv_file = None
        # Leader
        # --------------------------------------------
        self._waypoints_leader_file_name = "waypoints_leader.txt"
        self._waypoints_leader_file_path = os.path.join(carla_common_pkg, 'config', self._waypoints_leader_file_name)
        self._waypoints_leader_np = None
        # Index of waypoint that is currently closest to
        # the car (assumed to be the first index)
        self._leader_closest_index = 0
        # Closest distance of closest waypoint to car
        self._leader_closest_distance = 0

        # Follower
        # --------------------------------------------
        self._waypoints_follower_file_name = "waypoints_follower.txt"
        self._waypoints_follower_file_path = os.path.join(
            carla_common_pkg, 'config', self._waypoints_follower_file_name)
        self._waypoints_follower_np = None

        self._get_waypoints()
        self._waypoints_acquired = True
        # print(self._waypoints_leader_np.shape)
        # print(self._waypoints_follower_np.shape)

        # Leader follows waypoints
        self._leader_follow_waypoints()

        # This will be used in callback functions to start following waypoint

        print("\N{cat} VehicleCommanderInterface Node has been initialised.")

    def _get_waypoints(self):
        '''
        Get the waypoints for the leader and the follower.
        '''
        try:
            with open(self._waypoints_leader_file_path, encoding="utf8") as stream:
                self._leader_csv_file = list(csv.reader(stream, delimiter=',', quoting=csv.QUOTE_NONNUMERIC))
                self._waypoints_leader_np = np.array(self._leader_csv_file)
        except FileNotFoundError:
            self.get_logger().error(f"The file {self._waypoints_leader_file_path} does not exist.")
            sys.exit()

        try:
            with open(self._waypoints_follower_file_path, encoding="utf8") as stream:
                self._follower_csv_file = list(csv.reader(stream, delimiter=',', quoting=csv.QUOTE_NONNUMERIC))
                self._waypoints_follower_np = np.array(self._follower_csv_file)
        except FileNotFoundError:
            self.get_logger().error(f"The file {self._waypoints_follower_file_path} does not exist.")
            sys.exit()

    def _leader_follow_waypoints(self):
        '''
        Make both the leader and the follower follow the waypoints.
        '''
        # Update the controller waypoint path with the best local path.
        # Linear interpolation computation on the waypoints
        # is also used to ensure a fine resolution between points.

        # Because the waypoints are discrete and our controller performs better
        # with a continuous path, here we will send a subset of the waypoints
        # within some lookahead distance from the closest point to the vehicle.
        # Interpolating between each waypoint will provide a finer resolution
        # path and make it more "continuous". A simple linear interpolation
        # is used as a preliminary method to address this issue, though it is
        # better addressed with better interpolation methods (spline
        # interpolation, for example).
        # More appropriate interpolation methods will not be used here for the
        # sake of demonstration on what effects discrete paths can have on
        # the controller. It is made much more obvious with linear
        # interpolation, because in a way part of the path will be continuous
        # while the discontinuous parts (which happens at the waypoints) will
        # show just what sort of effects these points have on the controller.

        wp_distance = []   # distance array

        for i in range(1, self._waypoints_leader_np.shape[0]):
            distance = np.sqrt(
                (self._waypoints_leader_np[i, 0] - self._waypoints_leader_np[i - 1, 0]) ** 2 +
                (self._waypoints_leader_np[i, 1] - self._waypoints_leader_np[i - 1, 1]) ** 2)
            wp_distance.append(distance)
        # last distance is 0 because it is the distance
        # from the last waypoint to the last waypoint
        wp_distance.append(0)

        # Linearly interpolate between waypoints and store in a list
        wp_interp = []    # interpolated values
        # hash table which indexes waypoints_np
        # to the index of the waypoint in wp_interp
        wp_interp_hash = []
        # counter for current interpolated point index
        interp_counter = 0
        # (rows = waypoints, columns = [x, y, v])
        for i in range(self._waypoints_leader_np.shape[0] - 1):
            # Add original waypoint to interpolated waypoints list (and append
            # it to the hash table)
            wp_interp.append(list(self._waypoints_leader_np[i]))
            wp_interp_hash.append(interp_counter)
            interp_counter += 1

            # Interpolate to the next waypoint. First compute the number of
            # points to interpolate based on the desired resolution and
            # incrementally add interpolated points until the next waypoint
            # is about to be reached.
            num_pts_to_interp = int(np.floor(wp_distance[i] / float(self.INTERP_DISTANCE_RES)) - 1)
            wp_vector = self._waypoints_leader_np[i+1] - self._waypoints_leader_np[i]
            wp_uvector = wp_vector / np.linalg.norm(wp_vector[0:2])

            for j in range(num_pts_to_interp):
                next_wp_vector = self.INTERP_DISTANCE_RES * float(j+1) * wp_uvector
                wp_interp.append(list(self._waypoints_leader_np[i] + next_wp_vector))

        # add last waypoint at the end
        wp_interp.append(list(self._waypoints_leader_np[-1]))
        wp_interp_hash.append(interp_counter)
        interp_counter += 1

        #############################################
        # Controller 2D Class Declaration
        #############################################
        # This is where we take the controller2d.py class
        # and apply it to the simulator
        controller = controller2d.Controller2D(self._leader_csv_file)

        # Send a control command to proceed to next iteration.
        # This mainly applies for simulations that are in synchronous mode.
        self._send_leader_cmd(throttle=0.0, steer=0, brake=1.0)
        

    def run_leader(self):
        '''
        Function to run the leader vehicle in the timer loop
        '''
        ###
        # Controller update (this uses the controller2d.py implementation)
        ###

        # To reduce the amount of waypoints sent to the controller,
        # provide a subset of waypoints that are within some
        # lookahead distance from the closest point to the car. Provide
        # a set of waypoints behind the car as well.

        # Find closest waypoint index to car. First increment the index
        # from the previous index until the new distance calculations
        # are increasing. Apply the same rule decrementing the index.
        # The final index should be the closest point (it is assumed that
        # the car will always break out of instability points where there
        # are two indices with the same minimum distance, as in the
        # center of a circle)
        
        current_x = self._leader_current_x
        current_y = self._leader_current_y
        closest_distance = np.linalg.norm(np.array([
            self._waypoints_leader_np[self._leader_closest_index, 0] - current_x,
            self._waypoints_leader_np[self._leader_closest_index, 1] - current_y]))
        new_distance = closest_distance
        new_index = self._leader_closest_index
        while new_distance <= closest_distance:
            closest_distance = new_distance
            self._leader_closest_index = new_index
            new_index += 1
            if new_index >= self._waypoints_leader_np.shape[0]:  # End of path
                break
            new_distance = np.linalg.norm(np.array([
                self._waypoints_leader_np[new_index, 0] - current_x,
                self._waypoints_leader_np[new_index, 1] - current_y]))
        new_distance = closest_distance
        new_index = self._leader_closest_index
        while new_distance <= closest_distance:
            closest_distance = new_distance
            self._leader_closest_index = new_index
            new_index -= 1
            if new_index < 0:  # Beginning of path
                break
            new_distance = np.linalg.norm(np.array([
                self._waypoints_leader_np[new_index, 0] - current_x,
                self._waypoints_leader_np[new_index, 1] - current_y]))
        
        # Once the closest index is found, return the path that has 1
        # waypoint behind and X waypoints ahead, where X is the index
        # that has a lookahead distance specified by
        # INTERP_LOOKAHEAD_DISTANCE
        waypoint_subset_first_index = self._leader_closest_index - 1
        if waypoint_subset_first_index < 0:
            waypoint_subset_first_index = 0

        waypoint_subset_last_index = self._leader_closest_index
        total_distance_ahead = 0
        while total_distance_ahead < self.INTERP_LOOKAHEAD_DISTANCE:
            total_distance_ahead += wp_distance[waypoint_subset_last_index]
            waypoint_subset_last_index += 1
            if waypoint_subset_last_index >= waypoints_np.shape[0]:
                waypoint_subset_last_index = waypoints_np.shape[0] - 1
                break

    # def run(self):
    #     '''
    #     Scenario Execution Loop
    #     '''

    #     # Iterate the frames until the end of the waypoints is reached or
    #     # the TOTAL_EPISODE_FRAMES is reached. The controller simulation then
    #     # ouptuts the results to the controller output directory.
    #     reached_the_end = False
    #     skip_first_frame = True

    #     # # Initialize the current timestamp.
    #     current_timestamp = self._start_timestamp

    #     # # Initialize collision history
    #     prev_collision_vehicles = 0
    #     prev_collision_pedestrians = 0
    #     prev_collision_other = 0

    #     # for frame in range(TOTAL_EPISODE_FRAMES):
    #     # Gather current data from the CARLA server

    #     # Update pose and timestamp
    #     prev_timestamp = current_timestamp
    #     current_x = self._follower_current_x
    #     current_y = self._follower_current_y
    #     current_yaw = self._follower_current_yaw

    #     current_speed = self._follower_current_velocity
    #     current_timestamp = float(self._current_time)

    #     # Wait for some initial time before starting the demo
    #     self._send_follower_cmd(0.0, 0.0, 1.0)

    #     # Store history
    #     # self._x_history.append(current_x)
    #     # self._y_history.append(current_y)
    #     # self._yaw_history.append(current_yaw)
    #     # self._speed_history.append(current_speed)
    #     # self._time_history.append(current_timestamp)

    #     # Store collision history
    #     # collided_flag, prev_collision_vehicles, prev_collision_pedestrians, prev_collision_other = get_player_collided_flag(
    #     #     measurement_data, prev_collision_vehicles, prev_collision_pedestrians, prev_collision_other)
    #     # collided_flag_history.append(collided_flag)

    #     ###
    #     # Local Planner Update:
    #     #   This will use the behavioral_planner.py and local_planner.py
    #     ###

    #     # Obtain Lead Vehicle information.
    #     lead_car_pos = []
    #     lead_car_length = []
    #     lead_car_speed = []

    #     lead_car_pos.append([agent.vehicle.transform.location.x, agent.vehicle.transform.location.y])
    #     lead_car_length.append(agent.vehicle.bounding_box.extent.x)
    #     lead_car_speed.append(self._leader_current_velocity)

    #     # Execute the behaviour and local planning in the current instance
    #     # Note that updating the local path during every controller update
    #     # produces issues with the tracking performance (imagine everytime
    #     # the controller tried to follow the path, a new path appears). For
    #     # this reason, the local planner (LP) will update every X frame,
    #     # stored in the variable LP_FREQUENCY_DIVISOR, as it is analogous
    #     # to be operating at a frequency that is a division to the
    #     # simulation frequency.
    #     # if frame % self.LP_FREQUENCY_DIVISOR == 0:
    #     # --------------------------------------------------------------
    #     #  # Compute open loop speed estimate.
    #     open_loop_speed = self._local_planner._velocity_planner.get_open_loop_speed(current_timestamp - prev_timestamp)

    #     #  # Calculate the goal state set in the local frame for the local planner.
    #     #  # Current speed should be open loop for the velocity profile generation.
    #     ego_state = [current_x, current_y, current_yaw, open_loop_speed]

    #     #  # Set lookahead based on current speed.
    #     self._behavioral_planner.set_lookahead(self.BP_LOOKAHEAD_BASE + self.BP_LOOKAHEAD_TIME * open_loop_speed)

    #     #  # Perform a state transition in the behavioural planner.
    #     self._behavioral_planner.transition_state(self._waypoints, ego_state, current_speed)

    #     #  # Check to see if we need to follow the lead vehicle.
    #     # bp.check_for_lead_vehicle(ego_state, lead_car_pos[1])

    #     #  # Compute the goal state set from the behavioural planner's computed goal state.
    #     goal_state_set = self._local_planner.get_goal_state_set(
    #         self._behavioral_planner._goal_index,
    #         self._behavioral_planner._goal_state,
    #         self._waypoints, ego_state)

    #     #  # Calculate planned paths in the local frame.
    #     paths, path_validity = self._local_planner.plan_paths(goal_state_set)

    #     #  # Transform those paths back to the global frame.
    #     paths = local_planner.transform_paths(paths, ego_state)

    #     #  # Perform collision checking.
    #     collision_check_array = self._local_planner._collision_checker.collision_check(
    #         paths, [self._parked_vehicle_box_pts])

    #     #  # Compute the best local path.
    #     best_index = self._local_planner._collision_checker.select_best_path_index(
    #         paths, collision_check_array, self._behavioral_planner._goal_state)
    #     # If no path was feasible, continue to follow the previous best path.
    #     # if best_index == None:
    #     #     best_path = self._local_planner._prev_best_path
    #     # else:
    #     best_path = paths[best_index]
    #     self._local_planner._prev_best_path = best_path

    #     #  # Compute the velocity profile for the path, and compute the waypoints.
    #     #  # Use the lead vehicle to inform the velocity profile's dynamic obstacle handling.
    #     #  # In this scenario, the only dynamic obstacle is the lead vehicle at index 1.
    #     desired_speed = self._behavioral_planner._goal_state[2]
    #     lead_car_state = [lead_car_pos[1][0], lead_car_pos[1][1], lead_car_speed[1]]
    #     decelerate_to_stop = self._behavioral_planner._state == behavioral_planner.DECELERATE_TO_STOP
    #     local_waypoints = self._local_planner._velocity_planner.compute_velocity_profile(
    #         best_path, desired_speed, ego_state, current_speed, decelerate_to_stop, lead_car_state, self._behavioral_planner._follow_lead_vehicle)
    #     # --------------------------------------------------------------

    #     if local_waypoints != None:
    #         # Update the controller waypoint path with the best local path.
    #         # This controller is similar to that developed in Course 1 of this
    #         # specialization.  Linear interpolation computation on the waypoints
    #         # is also used to ensure a fine resolution between points.
    #         wp_distance = []   # distance array
    #         local_waypoints_np = np.array(local_waypoints)
    #         for i in range(1, local_waypoints_np.shape[0]):
    #             wp_distance.append(
    #                 np.sqrt((local_waypoints_np[i, 0] - local_waypoints_np[i-1, 0])**2 +
    #                         (local_waypoints_np[i, 1] - local_waypoints_np[i-1, 1])**2))
    #         wp_distance.append(0)  # last distance is 0 because it is the distance
    #         # from the last waypoint to the last waypoint

    #         # Linearly interpolate between waypoints and store in a list
    #         wp_interp = []    # interpolated values
    #         # (rows = waypoints, columns = [x, y, v])
    #         for i in range(local_waypoints_np.shape[0] - 1):
    #             # Add original waypoint to interpolated waypoints list (and append
    #             # it to the hash table)
    #             wp_interp.append(list(local_waypoints_np[i]))

    #             # Interpolate to the next waypoint. First compute the number of
    #             # points to interpolate based on the desired resolution and
    #             # incrementally add interpolated points until the next waypoint
    #             # is about to be reached.
    #             num_pts_to_interp = int(np.floor(wp_distance[i] / float(self.INTERP_DISTANCE_RES)) - 1)
    #             wp_vector = local_waypoints_np[i+1] - local_waypoints_np[i]
    #             wp_uvector = wp_vector / np.linalg.norm(wp_vector[0:2])

    #             for j in range(num_pts_to_interp):
    #                 next_wp_vector = self.INTERP_DISTANCE_RES * float(j+1) * wp_uvector
    #                 wp_interp.append(list(local_waypoints_np[i] + next_wp_vector))
    #         # add last waypoint at the end
    #         wp_interp.append(list(local_waypoints_np[-1]))

    #         # Update the other controller values and controls
    #         self._controller.update_waypoints(wp_interp)

    #         ###
    #         # Controller Update
    #         ###
    #         if local_waypoints is not None and local_waypoints != []:
    #             # self._controller.update_values(current_x, current_y, current_yaw,
    #             #                          current_speed,
    #             #                          current_timestamp, frame)
    #             self._controller.update_controls()
    #             cmd_throttle, cmd_steer, cmd_brake = self._controller.get_commands()
    #         else:
    #             cmd_throttle = 0.0
    #             cmd_steer = 0.0
    #             cmd_brake = 0.0

    #         # Skip the first frame or if there exists no local paths
    #         # if skip_first_frame and frame == 0:
    #         #     pass
    #         # elif local_waypoints == None:
    #         #     pass
    #         # else:
    #         #     pass

    #         # Output controller command to CARLA server
    #         self._send_follower_cmd(cmd_throttle, cmd_steer, cmd_brake)

    #         # Find if reached the end of waypoint. If the car is within
    #         # DIST_THRESHOLD_TO_LAST_WAYPOINT to the last waypoint,
    #         # the simulation will end.
    #         dist_to_last_waypoint = np.linalg.norm(np.array([
    #             self._waypoints[-1][0] - current_x,
    #             self._waypoints[-1][1] - current_y]))
    #         if dist_to_last_waypoint < self.DIST_THRESHOLD_TO_LAST_WAYPOINT:
    #             reached_the_end = True
    #         if reached_the_end:
    #             # stop the vehicle
    #             self._send_follower_cmd(0.0, 0.0, 1.0)

    # def _get_parked_vehicle_boxes(self):
    #     parked_vehicles = None
    #     try:
    #         parked_vehicles = self._yaml_data["parked_vehicle"]
    #     except KeyError:
    #         self.get_logger().info(
    #             "YAML configuration file does not contain a 'parked_vehicle' section")

    #     if parked_vehicles is not None:
    #         for parked_vehicle in parked_vehicles:
    #             yaw = parked_vehicle["spawn_point"]["yaw"]
    #             yaw = np.deg2rad(yaw)
    #             x = parked_vehicle["spawn_point"]["x"]
    #             y = parked_vehicle["spawn_point"]["y"]
    #             # z = parked_vehicle["spawn_point"]["z"]
    #             xrad = parked_vehicle["box_radius"]["x"]
    #             yrad = parked_vehicle["box_radius"]["y"]
    #             # zrad = parked_vehicle["box_radius"]["z"]
    #             cpos = np.array([
    #                 [-xrad, -xrad, -xrad, 0,    xrad, xrad, xrad,  0],
    #                 [-yrad, 0,     yrad,  yrad, yrad, 0,    -yrad, -yrad]])
    #             rotyaw = np.array([
    #                 [np.cos(yaw), np.sin(yaw)],
    #                 [-np.sin(yaw), np.cos(yaw)]])
    #             cpos_shift = np.array([
    #                 [x, x, x, x, x, x, x, x],
    #                 [y, y, y, y, y, y, y, y]])
    #             cpos = np.add(np.matmul(rotyaw, cpos), cpos_shift)
    #             for j in range(cpos.shape[1]):
    #                 self._parked_vehicle_box_pts.append([cpos[0, j], cpos[1, j]])

    # def _get_stopsign_fences(self):
    #     '''
    #     Get the stop sign fences from the yaml file
    #     '''
    #     stop_signs = None
    #     try:
    #         stop_signs = self._yaml_data["stop_sign"]
    #     except KeyError:
    #         self.get_logger().info(
    #             "YAML configuration file does not contain a 'stop_sign' section")

    #     if stop_signs is not None:
    #         for stop_sign in stop_signs:
    #             yaw = stop_sign["location"]["yaw"]
    #             # convert to radians
    #             yaw = np.deg2rad(yaw)
    #             x = stop_sign["location"]["x"]
    #             y = stop_sign["location"]["y"]
    #             # z = stop_sign["location"]["z"]
    #             yaw = yaw + np.pi / 2.0  # add 90 degrees for fence
    #             spos = np.array([
    #                 [0, 0],
    #                 [0, self.STOP_SIGN_FENCELENGTH]])
    #             rotyaw = np.array([
    #                 [np.cos(yaw), np.sin(yaw)],
    #                 [-np.sin(yaw), np.cos(yaw)]])
    #             spos_shift = np.array([
    #                 [x, x],
    #                 [y, y]])
    #             spos = np.add(np.matmul(rotyaw, spos), spos_shift)
    #             self._stopsign_fences.append([spos[0, 0], spos[1, 0], spos[0, 1], spos[1, 1]])

    def _waypoint_follower_cb(self):
        # self.run()
        pass

    def _clock_cb(self, msg: Clock):
        '''
        /clock topic callback function

        Args:
            msg (Clock): Clock message
        '''
        self._current_time = msg.clock.sec
        self.get_logger().info(f'Current time: {self._current_time}', throttle_duration_sec=2)

    def _follower_status_cb(self, msg: CarlaEgoVehicleStatus):
        '''
        /carla/ego_vehicle/vehicle_status topic callback function

        Args:
            msg (CarlaEgoVehicleStatus): CarlaEgoVehicleStatus message
        '''
        self._follower_current_velocity = msg.velocity
        self.get_logger().info(f'Follower velocity: {self._follower_current_velocity}',
                               throttle_duration_sec=2)
        # self.get_logger().info(f'Velocity: {msg.velocity}')
        # self.get_logger().info(f'Throttle: {msg.control.throttle}')
        # self.get_logger().info(f'Steer: {msg.control.steer}')
        # self.get_logger().info(f'Break: {msg.control.brake}')

    def _leader_status_cb(self, msg: CarlaEgoVehicleStatus):
        '''
        /carla/ego_vehicle/vehicle_status topic callback function

        Args:
            msg (CarlaEgoVehicleStatus): CarlaEgoVehicleStatus message
        '''
        self._leader_current_velocity = msg.velocity
        self.get_logger().info(f'Leader velocity: {self._leader_current_velocity}',
                               throttle_duration_sec=2)

    def _follower_odometry_cb(self, msg: Odometry):
        '''
        /carla/ego_vehicle/odometry topic callback function

        Args:
            msg (Odometry): Odometry message
        '''
        self._follower_current_x = msg.pose.pose.position.x
        self._follower_current_y = msg.pose.pose.position.y
        self._follower_current_rot_x = msg.pose.pose.orientation.x
        self._follower_current_rot_y = msg.pose.pose.orientation.y
        self._follower_current_rot_z = msg.pose.pose.orientation.z
        self._follower_current_rot_w = msg.pose.pose.orientation.w
        self._follower_current_yaw = tf_transformations.euler_from_quaternion(
            [self._follower_current_rot_x,
             self._follower_current_rot_y,
             self._follower_current_rot_z,
             self._follower_current_rot_w])[2]

    def _leader_odometry_cb(self, msg: Odometry):
        '''
        /carla/hero/odometry topic callback function

        Args:
            msg (Odometry): Odometry message
        '''
        self._leader_current_x = msg.pose.pose.position.x
        self._leader_current_y = msg.pose.pose.position.y
        self._leader_current_rot_x = msg.pose.pose.orientation.x
        self._leader_current_rot_y = msg.pose.pose.orientation.y
        self._leader_current_rot_z = msg.pose.pose.orientation.z
        self._leader_current_rot_w = msg.pose.pose.orientation.w
        self._leader_current_yaw = tf_transformations.euler_from_quaternion(
            [self._leader_current_rot_x,
             self._leader_current_rot_y,
             self._leader_current_rot_z,
             self._leader_current_rot_w])[2]

    def _send_follower_cmd(self, throttle: float, steer: float, brake: float):
        '''
        Send vehicle command to the follower

        Args:
            throttle (float): Throttle value
            steer (float): Steer value
            brake (float): Brake value
        '''

        self._follower_vehicle_control.throttle = throttle
        self._follower_vehicle_control.steer = steer
        self._follower_vehicle_control.brake = brake
        self._follower_cmd_publisher.publish(self._follower_vehicle_control)

    def _send_leader_cmd(self, throttle: float, steer: float, brake: float):
        '''
        Send vehicle command to the leader

        Args:
            throttle (float): Throttle value
            steer (float): Steer value
            brake (float): Brake value
        '''

        self._leader_vehicle_control.throttle = throttle
        self._leader_vehicle_control.steer = steer
        self._leader_vehicle_control.brake = brake
        self._leader_cmd_publisher.publish(self._leader_vehicle_control)

    # def _vehicle_action_timer_callback(self):
    #     '''
    #     Callback for the timer
    #     '''

    #     pass
    #     # self.get_logger().info(f'\N{dog} {pygame.time.Clock()}')
    #     # self.get_logger().info(f'\N{dog} {carla.libcarla.}')

    #     # self._follower_vehicle_control.gear = 1
    #     # self._follower_vehicle_control.throttle = 0.3
    #     # self._follower_vehicle_control.reverse = False
    #     # self.get_logger().info(f'\N{dog} Publishing throttle {self._follower_vehicle_control.throttle}')
    #     # self._follower_cmd_publisher.publish(self._follower_vehicle_control)
