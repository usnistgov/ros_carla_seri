
'''
Node to command the ego vehicle
'''

# R0902: too-many-instance-attributes
# R0915: too-many-statements
# pylint: disable=locally-disabled, multiple-statements
# pylint: disable=locally-disabled, fixme
# pylint: disable=locally-disabled, line-too-long
# pylint: disable=locally-disabled, R0902
# pylint: disable=locally-disabled, R0915
import os
import sys
import csv
import math
import numpy as np
import yaml
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

    # selected path
    INTERP_DISTANCE_RES = 0.01  # distance between interpolated points

    def __init__(self):
        super().__init__('vehicle_commander')

        print("\N{cat} VehicleCommanderInterface Node has been initialised.")
        #############################################
        # Callback groups
        #############################################
        timer_group = MutuallyExclusiveCallbackGroup()
        subscription_group = MutuallyExclusiveCallbackGroup()

        #############################################
        # Subscribers
        #############################################
        self.create_subscription(CarlaEgoVehicleStatus, '/carla/ego_vehicle/vehicle_status',
                                 self._leader_vehicle_status_cb, 10, callback_group=subscription_group)

        self._current_velocity = 0

        self.create_subscription(CarlaEgoVehicleStatus, '/carla/hero0/vehicle_status',
                                 self._hero_status_cb, 10, callback_group=subscription_group)

        self._current_hero_velocity = 0

        self._current_x = 0
        self._current_y = 0
        self._current_rot_x = 0
        self._current_rot_y = 0
        self._current_rot_z = 0
        self._current_rot_w = 0
        self._current_roll = 0
        self._current_pitch = 0
        self._current_yaw = 0

        self.create_subscription(Odometry, '/carla/ego_vehicle/odometry',
                                 self._vehicle_odometry_cb, 10, callback_group=subscription_group)

        self._current_x = 0
        self._current_y = 0
        self._current_rot_x = 0
        self._current_rot_y = 0
        self._current_rot_z = 0
        self._current_rot_w = 0
        self._current_roll = 0
        self._current_pitch = 0
        self._current_yaw = 0

        self.create_subscription(Clock, '/clock',
                                 self._clock_cb, 10, callback_group=subscription_group)

        self._current_time = 0

        #############################################
        # publisher
        #############################################
        self._tesla_cmd_publisher = self.create_publisher(
            CarlaEgoVehicleControl, '/carla/ego_vehicle/vehicle_control_cmd', 10)
        # message to publish
        self._tesla_controller = CarlaEgoVehicleControl()

        self._volkswagen_autopilot_publisher = self.create_publisher(
            Bool, '/carla/hero1/enable_autopilot', 10)
        self._nissan_autopilot_publisher = self.create_publisher(
            Bool, '/carla/hero0/enable_autopilot', 10)

        #############################################
        # timer
        #############################################
        # self._vehicle_action_timer = self.create_timer(1, self._vehicle_action_timer_callback,
        #  callback_group=timer_group)
        self._waypoint_follower_timer = self.create_timer(0.5, self._waypoint_follower_cb,
                                                          callback_group=timer_group)

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
        self._controller = controller2d.Controller2D(self._waypoints)

        # #############################################
        # # Determine simulation average timestep (and total frames)
        # #############################################
        # # Ensure at least one frame is used to compute average timestep
        # num_iterations = self.ITER_FOR_SIM_TIMESTEP
        # if (self.ITER_FOR_SIM_TIMESTEP < 1):
        #     num_iterations = 1

        # Gather current data from the CARLA server. This is used to get the
        # simulator starting game time. Note that we also need to
        # send a command back to the CARLA server because synchronous mode
        # is enabled.

        # sim_start_stamp = self._current_time
        # # Send a control command to proceed to next iteration.
        # # This mainly applies for simulations that are in synchronous mode.
        self._send_vehicle_cmd(throttle=0.0, steer=0.0, brake=1.0)

        # # Computes the average timestep based on several initial iterations
        # sim_duration = 0
        # for i in range(num_iterations):
        #     # Gather current data
        #     # Send a control command to proceed to next iteration
        #     self._send_vehicle_cmd(throttle=0.0, steer=0.0, brake=1.0)
        #     # Last stamp
        #     if i == num_iterations - 1:
        #         sim_duration = self._current_time - sim_start_stamp

        # # Outputs average simulation timestep and computes how many frames
        # # will elapse before the simulation should end based on various
        # # parameters that we set in the beginning.
        # # SIMULATION_TIME_STEP = sim_duration / float(num_iterations)
        # SIMULATION_TIME_STEP = 0.05

        # print("SERVER SIMULATION STEP APPROXIMATION: " + str(SIMULATION_TIME_STEP))
        # TOTAL_EPISODE_FRAMES = int(
        #     (self.TOTAL_RUN_TIME + self.WAIT_TIME_BEFORE_START) / SIMULATION_TIME_STEP) + self.TOTAL_FRAME_BUFFER

        #############################################
        # Frame-by-Frame Iteration and Initialization
        #############################################
        # Store pose history starting from the start position
        self._start_timestamp = self._current_time
        self._send_vehicle_cmd(throttle=0.0, steer=0.0, brake=1.0)
        self._x_history = [self._current_x]
        self._y_history = [self._current_y]
        self._yaw_history = [self._current_yaw]
        self._time_history = [0]
        self._speed_history = [0]
        self._collided_flag_history = [False]  # assume player starts off non-collided

        #############################################
        # Local Planner Variables
        #############################################
        self._wp_goal_index = 0
        self._local_waypoints = None
        self._path_validity = np.zeros((self.NUM_PATHS, 1), dtype=bool)
        self._local_planner = local_planner.LocalPlanner(self.NUM_PATHS,
                                                         self.PATH_OFFSET,
                                                         self.CIRCLE_OFFSETS,
                                                         self.CIRCLE_RADII,
                                                         self.PATH_SELECT_WEIGHT,
                                                         self.TIME_GAP,
                                                         self.A_MAX,
                                                         self.SLOW_SPEED,
                                                         self.STOP_LINE_BUFFER)
        self._behavioral_planner = behavioral_planner.BehaviouralPlanner(self.BP_LOOKAHEAD_BASE,
                                                                         self._stopsign_fences,
                                                                         self.LEAD_VEHICLE_LOOKAHEAD)

        # #############################################
        # # Scenario Execution Loop
        # #############################################

        # # Iterate the frames until the end of the waypoints is reached or
        # # the TOTAL_EPISODE_FRAMES is reached. The controller simulation then
        # # ouptuts the results to the controller output directory.
        # reached_the_end = False
        # skip_first_frame = True

        # # Initialize the current timestamp.
        # current_timestamp = start_timestamp

        # # Initialize collision history
        # prev_collision_vehicles = 0
        # prev_collision_pedestrians = 0
        # prev_collision_other = 0

        # for frame in range(TOTAL_EPISODE_FRAMES):
        #     # Gather current data from the CARLA server

        #     # Update pose and timestamp
        #     prev_timestamp = current_timestamp
        #     current_x = self._current_x
        #     current_y = self._current_y
        #     current_yaw = self._current_yaw

        #     current_speed = self._current_velocity
        #     current_timestamp = float(self._current_time)

        #     # Wait for some initial time before starting the demo
        #     if current_timestamp <= self.WAIT_TIME_BEFORE_START:
        #         self._send_vehicle_cmd(0.0, 0.0, 1.0)
        #         continue
        #     else:
        #         current_timestamp = current_timestamp - self.WAIT_TIME_BEFORE_START

        #     # Store history
        #     x_history.append(current_x)
        #     y_history.append(current_y)
        #     yaw_history.append(current_yaw)
        #     speed_history.append(current_speed)
        #     time_history.append(current_timestamp)

        #     # Store collision history
        #     # collided_flag, prev_collision_vehicles, prev_collision_pedestrians, prev_collision_other = get_player_collided_flag(
        #     #     measurement_data, prev_collision_vehicles, prev_collision_pedestrians, prev_collision_other)
        #     # collided_flag_history.append(collided_flag)

        #     ###
        #     # Local Planner Update:
        #     #   This will use the behavioural_planner.py and local_planner.py
        #     #   implementations that the learner will be tasked with in
        #     #   the Course 4 final project
        #     ###

        #     # Obtain Lead Vehicle information.
        #     lead_car_pos = []
        #     lead_car_length = []
        #     lead_car_speed = []

        #     # lead_car_pos.append([agent.vehicle.transform.location.x, agent.vehicle.transform.location.y])
        #     # lead_car_length.append(agent.vehicle.bounding_box.extent.x)
        #     # lead_car_speed.append(self._current_hero_velocity)

        #     # Execute the behaviour and local planning in the current instance
        #     # Note that updating the local path during every controller update
        #     # produces issues with the tracking performance (imagine everytime
        #     # the controller tried to follow the path, a new path appears). For
        #     # this reason, the local planner (LP) will update every X frame,
        #     # stored in the variable LP_FREQUENCY_DIVISOR, as it is analogous
        #     # to be operating at a frequency that is a division to the
        #     # simulation frequency.
        #     if frame % self.LP_FREQUENCY_DIVISOR == 0:
        #         # TODO Once you have completed the prerequisite functions of each of these
        #         # lines, you can uncomment the code below the dashed line to run the planner.
        #         # Note that functions lower in this block often require outputs from the functions
        #         # earlier in this block, so it may be easier to implement those first to
        #         # get a more intuitive flow of the planner.
        #         # In addition, some of these functions have already been implemented for you,
        #         # but it is useful for you to understand what each function is doing.
        #         # Before you uncomment a function, please take the time to take a look at
        #         # it and understand what is going on. It will also help inform you on the
        #         # flow of the planner, which in turn will help you implement the functions
        #         # flagged for you in the TODO's.

        #         # TODO: Uncomment each code block between the dashed lines to run the planner.
        #         # --------------------------------------------------------------
        #         #  # Compute open loop speed estimate.
        #         open_loop_speed = lp._velocity_planner.get_open_loop_speed(current_timestamp - prev_timestamp)

        #         #  # Calculate the goal state set in the local frame for the local planner.
        #         #  # Current speed should be open loop for the velocity profile generation.
        #         ego_state = [current_x, current_y, current_yaw, open_loop_speed]

        #         #  # Set lookahead based on current speed.
        #         bp.set_lookahead(self.BP_LOOKAHEAD_BASE + self.BP_LOOKAHEAD_TIME * open_loop_speed)

        #         #  # Perform a state transition in the behavioural planner.
        #         bp.transition_state(self._waypoints, ego_state, current_speed)

        #         #  # Check to see if we need to follow the lead vehicle.
        #         bp.check_for_lead_vehicle(ego_state, lead_car_pos[1])

        #         #  # Compute the goal state set from the behavioural planner's computed goal state.
        #         goal_state_set = lp.get_goal_state_set(bp._goal_index, bp._goal_state, self._waypoints, ego_state)

        #         #  # Calculate planned paths in the local frame.
        #         paths, path_validity = lp.plan_paths(goal_state_set)

        #         #  # Transform those paths back to the global frame.
        #         paths = local_planner.transform_paths(paths, ego_state)

        #         #  # Perform collision checking.
        #         collision_check_array = lp._collision_checker.collision_check(paths, [self._parked_vehicle_box_pts])

        #         #  # Compute the best local path.
        #         best_index = lp._collision_checker.select_best_path_index(paths, collision_check_array, bp._goal_state)
        #         # If no path was feasible, continue to follow the previous best path.
        #         if best_index == None:
        #             best_path = lp._prev_best_path
        #         else:
        #             best_path = paths[best_index]
        #             lp._prev_best_path = best_path

        #         #  # Compute the velocity profile for the path, and compute the waypoints.
        #         #  # Use the lead vehicle to inform the velocity profile's dynamic obstacle handling.
        #         #  # In this scenario, the only dynamic obstacle is the lead vehicle at index 1.
        #         desired_speed = bp._goal_state[2]
        #         lead_car_state = [lead_car_pos[1][0], lead_car_pos[1][1], lead_car_speed[1]]
        #         decelerate_to_stop = bp._state == behavioral_planner.DECELERATE_TO_STOP
        #         local_waypoints = lp._velocity_planner.compute_velocity_profile(
        #             best_path, desired_speed, ego_state, current_speed, decelerate_to_stop, lead_car_state, bp._follow_lead_vehicle)
        #         # --------------------------------------------------------------

        #         if local_waypoints != None:
        #             # Update the controller waypoint path with the best local path.
        #             # This controller is similar to that developed in Course 1 of this
        #             # specialization.  Linear interpolation computation on the waypoints
        #             # is also used to ensure a fine resolution between points.
        #             wp_distance = []   # distance array
        #             local_waypoints_np = np.array(local_waypoints)
        #             for i in range(1, local_waypoints_np.shape[0]):
        #                 wp_distance.append(
        #                     np.sqrt((local_waypoints_np[i, 0] - local_waypoints_np[i-1, 0])**2 +
        #                             (local_waypoints_np[i, 1] - local_waypoints_np[i-1, 1])**2))
        #             wp_distance.append(0)  # last distance is 0 because it is the distance
        #             # from the last waypoint to the last waypoint

        #             # Linearly interpolate between waypoints and store in a list
        #             wp_interp = []    # interpolated values
        #             # (rows = waypoints, columns = [x, y, v])
        #             for i in range(local_waypoints_np.shape[0] - 1):
        #                 # Add original waypoint to interpolated waypoints list (and append
        #                 # it to the hash table)
        #                 wp_interp.append(list(local_waypoints_np[i]))

        #                 # Interpolate to the next waypoint. First compute the number of
        #                 # points to interpolate based on the desired resolution and
        #                 # incrementally add interpolated points until the next waypoint
        #                 # is about to be reached.
        #                 num_pts_to_interp = int(np.floor(wp_distance[i] / float(self.INTERP_DISTANCE_RES)) - 1)
        #                 wp_vector = local_waypoints_np[i+1] - local_waypoints_np[i]
        #                 wp_uvector = wp_vector / np.linalg.norm(wp_vector[0:2])

        #                 for j in range(num_pts_to_interp):
        #                     next_wp_vector = self.INTERP_DISTANCE_RES * float(j+1) * wp_uvector
        #                     wp_interp.append(list(local_waypoints_np[i] + next_wp_vector))
        #             # add last waypoint at the end
        #             wp_interp.append(list(local_waypoints_np[-1]))

        #             # Update the other controller values and controls
        #             self._controller.update_waypoints(wp_interp)

        #     ###
        #     # Controller Update
        #     ###
        #     if local_waypoints is not None and local_waypoints != []:
        #         self._controller.update_values(current_x, current_y, current_yaw,
        #                                  current_speed,
        #                                  current_timestamp, frame)
        #         self._controller.update_controls()
        #         cmd_throttle, cmd_steer, cmd_brake = self._controller.get_commands()
        #     else:
        #         cmd_throttle = 0.0
        #         cmd_steer = 0.0
        #         cmd_brake = 0.0

        #     # Skip the first frame or if there exists no local paths
        #     if skip_first_frame and frame == 0:
        #         pass
        #     elif local_waypoints == None:
        #         pass
        #     else:
        #         pass

        #     # Output controller command to CARLA server
        #     self._send_vehicle_cmd(cmd_throttle, cmd_steer, cmd_brake)

        #     # Find if reached the end of waypoint. If the car is within
        #     # DIST_THRESHOLD_TO_LAST_WAYPOINT to the last waypoint,
        #     # the simulation will end.
        #     dist_to_last_waypoint = np.linalg.norm(np.array([
        #         self._waypoints[-1][0] - current_x,
        #         self._waypoints[-1][1] - current_y]))
        #     if dist_to_last_waypoint < self.DIST_THRESHOLD_TO_LAST_WAYPOINT:
        #         reached_the_end = True
        #     if reached_the_end:
        #         break

        # # End of demo - Stop vehicle and Store outputs to the controller output
        # # directory.
        # if reached_the_end:
        #     print("Reached the end of path. Writing to controller_output...")
        # else:
        #     print("Exceeded assessment time. Writing to controller_output...")
        # # Stop the car
        # self._send_vehicle_cmd(0.0, 0.0, 1.0)

    def run(self):
        '''
        Scenario Execution Loop
        '''

        # Iterate the frames until the end of the waypoints is reached or
        # the TOTAL_EPISODE_FRAMES is reached. The controller simulation then
        # ouptuts the results to the controller output directory.
        reached_the_end = False
        skip_first_frame = True

        # # Initialize the current timestamp.
        current_timestamp = self._start_timestamp

        # # Initialize collision history
        prev_collision_vehicles = 0
        prev_collision_pedestrians = 0
        prev_collision_other = 0

        # for frame in range(TOTAL_EPISODE_FRAMES):
        # Gather current data from the CARLA server

        # Update pose and timestamp
        prev_timestamp = current_timestamp
        current_x = self._current_x
        current_y = self._current_y
        current_yaw = self._current_yaw

        current_speed = self._current_velocity
        current_timestamp = float(self._current_time)

        # Wait for some initial time before starting the demo
        self._send_vehicle_cmd(0.0, 0.0, 1.0)
        # if current_timestamp <= self.WAIT_TIME_BEFORE_START:
        #     self._send_vehicle_cmd(0.0, 0.0, 1.0)
        #     continue
        # else:
        #     current_timestamp = current_timestamp - self.WAIT_TIME_BEFORE_START

        # Store history
        self._x_history.append(current_x)
        self._y_history.append(current_y)
        self._yaw_history.append(current_yaw)
        self._speed_history.append(current_speed)
        self._time_history.append(current_timestamp)

        # Store collision history
        # collided_flag, prev_collision_vehicles, prev_collision_pedestrians, prev_collision_other = get_player_collided_flag(
        #     measurement_data, prev_collision_vehicles, prev_collision_pedestrians, prev_collision_other)
        # collided_flag_history.append(collided_flag)

        ###
        # Local Planner Update:
        #   This will use the behavioral_planner.py and local_planner.py
        ###

        # Obtain Lead Vehicle information.
        lead_car_pos = []
        lead_car_length = []
        lead_car_speed = []

        # lead_car_pos.append([agent.vehicle.transform.location.x, agent.vehicle.transform.location.y])
        # lead_car_length.append(agent.vehicle.bounding_box.extent.x)
        # lead_car_speed.append(self._current_hero_velocity)

        # Execute the behaviour and local planning in the current instance
        # Note that updating the local path during every controller update
        # produces issues with the tracking performance (imagine everytime
        # the controller tried to follow the path, a new path appears). For
        # this reason, the local planner (LP) will update every X frame,
        # stored in the variable LP_FREQUENCY_DIVISOR, as it is analogous
        # to be operating at a frequency that is a division to the
        # simulation frequency.
        # if frame % self.LP_FREQUENCY_DIVISOR == 0:
        # --------------------------------------------------------------
        #  # Compute open loop speed estimate.
        open_loop_speed = self._local_planner._velocity_planner.get_open_loop_speed(current_timestamp - prev_timestamp)

        #  # Calculate the goal state set in the local frame for the local planner.
        #  # Current speed should be open loop for the velocity profile generation.
        ego_state = [current_x, current_y, current_yaw, open_loop_speed]

        #  # Set lookahead based on current speed.
        self._behavioral_planner.set_lookahead(self.BP_LOOKAHEAD_BASE + self.BP_LOOKAHEAD_TIME * open_loop_speed)

        #  # Perform a state transition in the behavioural planner.
        self._behavioral_planner.transition_state(self._waypoints, ego_state, current_speed)

        #  # Check to see if we need to follow the lead vehicle.
        # bp.check_for_lead_vehicle(ego_state, lead_car_pos[1])

        #  # Compute the goal state set from the behavioural planner's computed goal state.
        goal_state_set = self._local_planner.get_goal_state_set(
            self._behavioral_planner._goal_index,
            self._behavioral_planner._goal_state,
            self._waypoints, ego_state)

        #  # Calculate planned paths in the local frame.
        paths, path_validity = self._local_planner.plan_paths(goal_state_set)

        #  # Transform those paths back to the global frame.
        paths = local_planner.transform_paths(paths, ego_state)

        #  # Perform collision checking.
        collision_check_array = self._local_planner._collision_checker.collision_check(
            paths, [self._parked_vehicle_box_pts])

        #  # Compute the best local path.
        best_index = self._local_planner._collision_checker.select_best_path_index(
            paths, collision_check_array, self._behavioral_planner._goal_state)
        # If no path was feasible, continue to follow the previous best path.
        # if best_index == None:
        #     best_path = self._local_planner._prev_best_path
        # else:
        best_path = paths[best_index]
        self._local_planner._prev_best_path = best_path

        #  # Compute the velocity profile for the path, and compute the waypoints.
        #  # Use the lead vehicle to inform the velocity profile's dynamic obstacle handling.
        #  # In this scenario, the only dynamic obstacle is the lead vehicle at index 1.
        desired_speed = self._behavioral_planner._goal_state[2]
        lead_car_state = [lead_car_pos[1][0], lead_car_pos[1][1], lead_car_speed[1]]
        decelerate_to_stop = self._behavioral_planner._state == behavioral_planner.DECELERATE_TO_STOP
        local_waypoints = self._local_planner._velocity_planner.compute_velocity_profile(
            best_path, desired_speed, ego_state, current_speed, decelerate_to_stop, lead_car_state, self._behavioral_planner._follow_lead_vehicle)
        # --------------------------------------------------------------

        if local_waypoints != None:
            # Update the controller waypoint path with the best local path.
            # This controller is similar to that developed in Course 1 of this
            # specialization.  Linear interpolation computation on the waypoints
            # is also used to ensure a fine resolution between points.
            wp_distance = []   # distance array
            local_waypoints_np = np.array(local_waypoints)
            for i in range(1, local_waypoints_np.shape[0]):
                wp_distance.append(
                    np.sqrt((local_waypoints_np[i, 0] - local_waypoints_np[i-1, 0])**2 +
                            (local_waypoints_np[i, 1] - local_waypoints_np[i-1, 1])**2))
            wp_distance.append(0)  # last distance is 0 because it is the distance
            # from the last waypoint to the last waypoint

            # Linearly interpolate between waypoints and store in a list
            wp_interp = []    # interpolated values
            # (rows = waypoints, columns = [x, y, v])
            for i in range(local_waypoints_np.shape[0] - 1):
                # Add original waypoint to interpolated waypoints list (and append
                # it to the hash table)
                wp_interp.append(list(local_waypoints_np[i]))

                # Interpolate to the next waypoint. First compute the number of
                # points to interpolate based on the desired resolution and
                # incrementally add interpolated points until the next waypoint
                # is about to be reached.
                num_pts_to_interp = int(np.floor(wp_distance[i] / float(self.INTERP_DISTANCE_RES)) - 1)
                wp_vector = local_waypoints_np[i+1] - local_waypoints_np[i]
                wp_uvector = wp_vector / np.linalg.norm(wp_vector[0:2])

                for j in range(num_pts_to_interp):
                    next_wp_vector = self.INTERP_DISTANCE_RES * float(j+1) * wp_uvector
                    wp_interp.append(list(local_waypoints_np[i] + next_wp_vector))
            # add last waypoint at the end
            wp_interp.append(list(local_waypoints_np[-1]))

            # Update the other controller values and controls
            self._controller.update_waypoints(wp_interp)

            ###
            # Controller Update
            ###
            if local_waypoints is not None and local_waypoints != []:
                # self._controller.update_values(current_x, current_y, current_yaw,
                #                          current_speed,
                #                          current_timestamp, frame)
                self._controller.update_controls()
                cmd_throttle, cmd_steer, cmd_brake = self._controller.get_commands()
            else:
                cmd_throttle = 0.0
                cmd_steer = 0.0
                cmd_brake = 0.0

            # Skip the first frame or if there exists no local paths
            # if skip_first_frame and frame == 0:
            #     pass
            # elif local_waypoints == None:
            #     pass
            # else:
            #     pass

            # Output controller command to CARLA server
            self._send_vehicle_cmd(cmd_throttle, cmd_steer, cmd_brake)

            # Find if reached the end of waypoint. If the car is within
            # DIST_THRESHOLD_TO_LAST_WAYPOINT to the last waypoint,
            # the simulation will end.
            dist_to_last_waypoint = np.linalg.norm(np.array([
                self._waypoints[-1][0] - current_x,
                self._waypoints[-1][1] - current_y]))
            if dist_to_last_waypoint < self.DIST_THRESHOLD_TO_LAST_WAYPOINT:
                reached_the_end = True
            if reached_the_end:
                # stop the vehicle
                self._send_vehicle_cmd(0.0, 0.0, 1.0)

    def euler_from_quaternion(self):
        """
        Converts quaternion (w in last place) to euler roll, pitch, yaw
        quaternion = [x, y, z, w]
        """
        x = self._current_rot_x
        y = self._current_rot_y
        z = self._current_rot_z
        w = self._current_rot_w

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        self._current_roll = roll
        self._current_pitch = pitch

        self._current_yaw = math.radians(yaw)

    def read_yaml(self, path):
        '''
        Method to read a yaml file

        Args:
            path (str): absolute path to the yaml file

        '''
        with open(path, "r", encoding="utf8") as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError:
                self.get_logger().error("Unable to read configuration file")
                return {}

    def _get_waypoints(self):
        '''
        Get the waypoints from the waypoints.txt file
        '''
        try:
            with open(self._waypoints_file_path, encoding="utf8") as stream:
                self._waypoints = list(csv.reader(stream,
                                                  delimiter=',',
                                                  quoting=csv.QUOTE_NONNUMERIC))
                self._waypoints_np = np.array(self._waypoints)
        except FileNotFoundError:
            self.get_logger().error(f"The file {self._waypoints_file_path} does not exist.")
            sys.exit()

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

    def _waypoint_follower_cb(self):
        self.run()

    def _clock_cb(self, msg: Clock):
        '''
        /clock topic callback function

        Args:
            msg (Clock): Clock message
        '''
        self._current_time = msg.clock.sec

    def _leader_vehicle_status_cb(self, msg: CarlaEgoVehicleStatus):
        '''
        /carla/ego_vehicle/vehicle_status topic callback function

        Args:
            msg (CarlaEgoVehicleStatus): CarlaEgoVehicleStatus message
        '''
        self._current_velocity = msg.velocity
        # self.get_logger().info(f'Velocity: {msg.velocity}')
        # self.get_logger().info(f'Throttle: {msg.control.throttle}')
        # self.get_logger().info(f'Steer: {msg.control.steer}')
        # self.get_logger().info(f'Break: {msg.control.brake}')

    def _hero_status_cb(self, msg: CarlaEgoVehicleStatus):
        '''
        /carla/ego_vehicle/vehicle_status topic callback function

        Args:
            msg (CarlaEgoVehicleStatus): CarlaEgoVehicleStatus message
        '''
        self._current_hero_velocity = msg.velocity

    def _vehicle_odometry_cb(self, msg: Odometry):
        '''
        /carla/ego_vehicle/odometry topic callback function

        Args:
            msg (Odometry): Odometry message
        '''
        self._current_x = msg.pose.pose.position.x
        self._current_y = msg.pose.pose.position.y
        self._current_rot_x = msg.pose.pose.orientation.x
        self._current_rot_y = msg.pose.pose.orientation.y
        self._current_rot_z = msg.pose.pose.orientation.z
        self._current_rot_w = msg.pose.pose.orientation.w
        self.euler_from_quaternion()

    def _send_vehicle_cmd(self, throttle: float, steer: float, brake: float):
        '''
        Send vehicle command to the vehicle

        Args:
            throttle (float): Throttle value
            steer (float): Steer value
            brake (float): Brake value
        '''

        self._tesla_controller.throttle = throttle
        self._tesla_controller.steer = steer
        self._tesla_controller.brake = brake
        self._tesla_cmd_publisher.publish(self._tesla_controller)

    def _vehicle_action_timer_callback(self):
        '''
        Callback for the timer 
        '''
        pass
        # self.get_logger().info(f'\N{dog} {pygame.time.Clock()}')
        # self.get_logger().info(f'\N{dog} {carla.libcarla.}')

        # self._tesla_controller.gear = 1
        # self._tesla_controller.throttle = 0.3
        # self._tesla_controller.reverse = False
        # self.get_logger().info(f'\N{dog} Publishing throttle {self._tesla_controller.throttle}')
        # self._tesla_cmd_publisher.publish(self._tesla_controller)
