
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
import emoji
import csv
import rclpy
import time
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
from nav_msgs.msg import Path
from commander_py.common import read_yaml
from commander_py import controller2d
# (
#     controller2d,
#     local_planner,
#     behavioral_planner
# )


# ===============================================================
__author__ = "Zeid Kootbally"
__credits__ = ["Zeid Kootbally"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Zeid Kootbally"
__email__ = "zeid.kootbally@nist.gov"
__status__ = "Development"
# ===============================================================


class Timer(object):
    """ Timer Class

    The steps are used to calculate FPS, while the lap or seconds since lap is
    used to compute elapsed time.
    """

    def __init__(self, period):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()
        self._period_for_lap = period

    def tick(self):
        self.step += 1

    def has_exceeded_lap_period(self):
        if self.elapsed_seconds_since_lap() >= self._period_for_lap:
            return True
        else:
            return False

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) /\
            self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time


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
        _follower_current_speed  Current velocity of the follower.
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

    # commander output directory
    FOLLOWER_OUTPUT_FILE = 'follower_waypoints.txt'

    # controller output directory
    GROUND_TRUTH_OUTPUT_FOLDER = os.path.dirname(os.path.realpath(__file__)) +\
        '/ground_truth/'

    def __init__(self):
        super().__init__('vehicle_commander')

        # with open('waypoints.txt', 'w') as f:
        #     f.write('Create a new text file!')

        self._waypoint_log = ""
        self._waypoint_log_list = []

        #########################################################
        # Whether or not we need to collect ground truth data
        #########################################################
        self.declare_parameter('is_ground_truth', rclpy.Parameter.Type.BOOL)
        self._is_ground_truth = self.get_parameter("is_ground_truth")

        #########################################################
        # Controller Method: Pure Pursuit, Stanley, MPC
        #########################################################

        # self.declare_parameter('control_method', rclpy.Parameter.Type.STRING)
        self.declare_parameter('control_method', rclpy.Parameter.Type.STRING)
        self._control_method = self.get_parameter("control_method").value
        self.get_logger().info(f":ghost: Control Method: {self._control_method}")

        # Callback groups
        #############################################
        self._leader_timer_group = MutuallyExclusiveCallbackGroup()
        self._follower_timer_group = MutuallyExclusiveCallbackGroup()
        self._subscription_group = MutuallyExclusiveCallbackGroup()
        self._follower_subscription_group = MutuallyExclusiveCallbackGroup()
        #############################################
        # Publishers
        #############################################

        # Publisher to control the follower vehicle
        self._follower_cmd_publisher = self.create_publisher(
            CarlaEgoVehicleControl,
            '/carla/ego_vehicle/vehicle_control_cmd',
            10)
        # Message to control the follower vehicle
        self._follower_vehicle_control = CarlaEgoVehicleControl()

        # Publisher to control the leader vehicle
        self._leader_cmd_publisher = self.create_publisher(
            CarlaEgoVehicleControl,
            '/carla/hero/vehicle_control_cmd',
            10)
        # Message to control the leader vehicle
        self._leader_autopilot_publisher = self.create_publisher(
            Bool,
            '/carla/hero/enable_autopilot',
            10)

        #############################################
        # Subscribers
        #############################################

        # Subscriber to the follower vehicle status
        self.create_subscription(CarlaEgoVehicleStatus,
                                 '/carla/ego_vehicle/vehicle_status',
                                 self._follower_status_cb,
                                 10,
                                 callback_group=self._subscription_group)

        # Subscriber to the follower vehicle odometry
        self.create_subscription(Odometry,
                                 '/carla/ego_vehicle/odometry',
                                 self._follower_odometry_cb,
                                 10,
                                 callback_group=self._subscription_group)

        # Subscriber to waypoints topic
        # self.create_subscription(Path,
        #                          '/carla/ego_vehicle/waypoints',
        #                          self._follower_waypoints_cb,
        #                          10,
        #                          callback_group=self._subscription_group)

        # Subscriber to the leader vehicle status
        self.create_subscription(CarlaEgoVehicleStatus,
                                 '/carla/hero/vehicle_status',
                                 self._leader_status_cb,
                                 10,
                                 callback_group=self._subscription_group)

        # Subscriber to the leader vehicle odometry
        self.create_subscription(Odometry,
                                 '/carla/hero/odometry',
                                 self._leader_odometry_cb,
                                 10,
                                 callback_group=self._subscription_group)

        # Subscriber to the clock (to get the current time)
        self.create_subscription(Clock, '/clock',
                                 self._clock_cb,
                                 10,
                                 callback_group=self._subscription_group)

        #############################################
        # Time related variables
        #############################################
        self._current_time = 0.0

        #############################################
        # Follower related variables
        #############################################
        self._follower_current_speed = 0
        self._follower_current_x = 0
        self._follower_current_y = 0
        self._follower_current_rot_x = 0
        self._follower_current_rot_y = 0
        self._follower_current_rot_z = 0
        self._follower_current_rot_w = 0
        self._follower_current_roll = 0
        self._follower_current_pitch = 0
        self._follower_current_yaw = 0

        self._follower_ready = False

        # Used for logging waypoints every 1 m
        self._waypoints_within_distance = False
        self._follower_log_x = 0
        self._follower_log_y = 0
        self._output = ""

        # Controller
        # ---------------------------
        self._follower_controller = None
        self._follower_vehicle_control = CarlaEgoVehicleControl()
        self._follower_autopilot_control = Bool()

        #############################################
        # Leader related variables
        #############################################
        self._leader_current_speed = 0
        self._leader_current_x = 0
        self._leader_current_y = 0
        self._leader_current_rot_x = 0
        self._leader_current_rot_y = 0
        self._leader_current_rot_z = 0
        self._leader_current_rot_w = 0
        self._leader_current_roll = 0
        self._leader_current_pitch = 0
        self._leader_current_yaw = 0

        # Controller
        # ---------------------------
        self._leader_controller = None
        self._leader_vehicle_control = CarlaEgoVehicleControl()
        self._leader_autopilot_control = Bool()

        #############################################
        # Timer to run the control loop for the leader
        #############################################
        # self._run_leader_timer = self.create_timer(1.0, self._run_leader,
        #                                            callback_group=self._leader_timer_group)

        self._run_follower_timer = self.create_timer(1.0, self._run_follower,
                                                     callback_group=self._follower_timer_group)

        #########################################################
        # Variables related to the configuration file: seri.yaml
        #########################################################
        carla_common_pkg = FindPackageShare(package='carla_common').find('carla_common')
        config_file_name = "seri.yaml"
        config_file_path = os.path.join(carla_common_pkg, 'config', config_file_name)
        self._yaml_data = read_yaml(config_file_path)

        #############################################
        # Stop sign fences
        #############################################
        self._stopsign_fences = []     # [x0, y0, x1, y1]
        # self._get_stopsign_fences()

        #############################################
        # Parked vehicle boxes
        #############################################
        self._parked_vehicle_box_pts = []      # [x,y]
        # self._get_parked_vehicle_boxes()

        #############################################
        # Variables related to Waypoints
        #############################################
        self._leader_csv_file = None
        self._follower_csv_file = None
        self._follower_waypoints = []
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
        self._leader_wp_distance = []   # distance array
        self._leader_reached_the_end = False
        # Linearly interpolate between waypoints and store in a list
        self._leader_wp_interp = []    # interpolated values
        # hash table which indexes waypoints_np
        # to the index of the waypoint in wp_interp
        self._leader_wp_interp_hash = []

        # Follower
        # --------------------------------------------
        self._waypoints_follower_file_name = "follower_waypoints.txt"
        self._waypoints_follower_file_path = os.path.join(
            carla_common_pkg, 'config', self._waypoints_follower_file_name)
        self._waypoints_follower_np = None

        self._follower_wp_distance = []   # distance array
        # Linearly interpolate between waypoints and store in a list
        self._follower_wp_interp = []       # interpolated values
        # (rows = waypoints, columns = [x, y, v])
        self._follower_wp_interp_hash = []  # hash table which indexes waypoints_np
        # to the index of the waypoint in wp_interp
        self._follower_waypoints_acquired = False
        self._follower_reached_the_end = False
        self._follower_closest_index = 0  # Index of waypoint that is currently closest to
        # the car (assumed to be the first index)
        self._follower_closest_distance = 0  # Closest distance of closest waypoint to car

        self._set_up_follower_waypoints()

        print("\N{cat} VehicleCommanderInterface Node has been initialised.")

    def _set_up_follower_waypoints(self):

        # Opens the waypoint file and stores it to "waypoints"
        with open(self._waypoints_follower_file_path) as waypoints_file_handle:
            self._follower_waypoints = list(csv.reader(waypoints_file_handle,
                                                       delimiter=',',
                                                       quoting=csv.QUOTE_NONNUMERIC))
            self._waypoints_follower_np = np.array(self._follower_waypoints)

        # Linear interpolation computations
        # Compute a list of distances between waypoints

        for i in range(1, self._waypoints_follower_np.shape[0]):
            distance = np.sqrt((self._waypoints_follower_np[i, 0] - self._waypoints_follower_np[i-1, 0])**2 +
                               (self._waypoints_follower_np[i, 1] - self._waypoints_follower_np[i-1, 1])**2)

            # self.get_logger().info(
            #     f"Waypoint 1: [{self._waypoints_follower_np[i-1, 0], self._waypoints_follower_np[i-1, 1]}]")
            # self.get_logger().info(
            #     f"Waypoint 2: [{self._waypoints_follower_np[i, 0], self._waypoints_follower_np[i, 1]}]")
            # self.get_logger().info(f"Distance between waypoints: {distance}")
            self._follower_wp_distance.append(distance)

        # sys.exit()
        # last distance is 0 because it is the distance
        # from the last waypoint to the last waypoint
        self._follower_wp_distance.append(0)

        interp_counter = 0   # counter for current interpolated point index

        for i in range(self._waypoints_follower_np.shape[0] - 1):
            # Add original waypoint to interpolated waypoints list (and append
            # it to the hash table)
            self._follower_wp_interp.append(list(self._waypoints_follower_np[i]))
            self._follower_wp_interp_hash.append(interp_counter)
            interp_counter += 1

            # Interpolate to the next waypoint. First compute the number of
            # points to interpolate based on the desired resolution and
            # incrementally add interpolated points until the next waypoint
            # is about to be reached.
            num_pts_to_interp = int(np.floor(self._follower_wp_distance[i] /
                                             float(self.INTERP_DISTANCE_RES)) - 1)
            wp_vector = self._waypoints_follower_np[i+1] - self._waypoints_follower_np[i]
            wp_uvector = wp_vector / np.linalg.norm(wp_vector)
            for j in range(num_pts_to_interp):
                next_wp_vector = self.INTERP_DISTANCE_RES * float(j+1) * wp_uvector
                self._follower_wp_interp.append(list(self._waypoints_follower_np[i] + next_wp_vector))
                interp_counter += 1
        # add last waypoint at the end
        self._follower_wp_interp.append(list(self._waypoints_follower_np[-1]))
        self._follower_wp_interp_hash.append(interp_counter)
        interp_counter += 1

        #############################################
        # Controller 2D Class Declaration
        #############################################
        # This is where we take the controller2d.py class
        # and apply it to the simulator
        self._follower_controller = controller2d.Controller2D(self._follower_waypoints, self._control_method)
        self._follower_ready = True

    # def _get_waypoints(self):
    #     '''
    #     Get the waypoints for the leader and the follower.
    #     '''

    #     try:
    #         with open(self._waypoints_leader_file_path, encoding="utf8") as stream:
    #             self._leader_csv_file = list(csv.reader(stream, delimiter=',', quoting=csv.QUOTE_NONNUMERIC))
    #             self._waypoints_leader_np = np.array(self._leader_csv_file)
    #     except FileNotFoundError:
    #         self.get_logger().error(f"The file {self._waypoints_leader_file_path} does not exist.")
    #         sys.exit()

    #     # try:
    #     #     with open(self._waypoints_follower_file_path, encoding="utf8") as stream:
    #     #         self._follower_csv_file = list(csv.reader(stream, delimiter=',', quoting=csv.QUOTE_NONNUMERIC))
    #     #         print(self._follower_csv_file)
    #     #         self._waypoints_follower_np = np.array(self._follower_csv_file)
    #     # except FileNotFoundError:
    #     #     self.get_logger().error(f"The file {self._waypoints_follower_file_path} does not exist.")
    #     #     sys.exit()

    # def _leader_follow_waypoints(self):
    #     '''
    #     Make both the leader and the follower follow the waypoints.
    #     '''
    #     # Update the controller waypoint path with the best local path.
    #     # Linear interpolation computation on the waypoints
    #     # is also used to ensure a fine resolution between points.

    #     # Because the waypoints are discrete and our controller performs better
    #     # with a continuous path, here we will send a subset of the waypoints
    #     # within some lookahead distance from the closest point to the vehicle.
    #     # Interpolating between each waypoint will provide a finer resolution
    #     # path and make it more "continuous". A simple linear interpolation
    #     # is used as a preliminary method to address this issue, though it is
    #     # better addressed with better interpolation methods (spline
    #     # interpolation, for example).
    #     # More appropriate interpolation methods will not be used here for the
    #     # sake of demonstration on what effects discrete paths can have on
    #     # the controller. It is made much more obvious with linear
    #     # interpolation, because in a way part of the path will be continuous
    #     # while the discontinuous parts (which happens at the waypoints) will
    #     # show just what sort of effects these points have on the controller.

    #     for i in range(1, self._waypoints_leader_np.shape[0]):
    #         distance = np.sqrt(
    #             (self._waypoints_leader_np[i, 0] - self._waypoints_leader_np[i - 1, 0]) ** 2 +
    #             (self._waypoints_leader_np[i, 1] - self._waypoints_leader_np[i - 1, 1]) ** 2)
    #         self._leader_wp_distance.append(distance)
    #     # last distance is 0 because it is the distance
    #     # from the last waypoint to the last waypoint
    #     self._leader_wp_distance.append(0)

    #     # counter for current interpolated point index
    #     interp_counter = 0
    #     # (rows = waypoints, columns = [x, y, v])
    #     for i in range(self._waypoints_leader_np.shape[0] - 1):
    #         # Add original waypoint to interpolated waypoints list (and append
    #         # it to the hash table)
    #         self._leader_wp_interp.append(list(self._waypoints_leader_np[i]))
    #         self._leader_wp_interp_hash.append(interp_counter)
    #         interp_counter += 1

    #         # Interpolate to the next waypoint. First compute the number of
    #         # points to interpolate based on the desired resolution and
    #         # incrementally add interpolated points until the next waypoint
    #         # is about to be reached.
    #         num_pts_to_interp = int(np.floor(self._leader_wp_distance[i] / float(self.INTERP_DISTANCE_RES)) - 1)
    #         wp_vector = self._waypoints_leader_np[i+1] - self._waypoints_leader_np[i]
    #         wp_uvector = wp_vector / np.linalg.norm(wp_vector[0:2])

    #         for j in range(num_pts_to_interp):
    #             next_wp_vector = self.INTERP_DISTANCE_RES * float(j+1) * wp_uvector
    #             self._leader_wp_interp.append(list(self._waypoints_leader_np[i] + next_wp_vector))

    #     # add last waypoint at the end
    #     self._leader_wp_interp.append(list(self._waypoints_leader_np[-1]))
    #     self._leader_wp_interp_hash.append(interp_counter)
    #     interp_counter += 1

    #     #############################################
    #     # Controller 2D Class Declaration
    #     #############################################
    #     # This is where we take the controller2d.py class
    #     # and apply it to the simulator
    #     self._leader_controller = controller2d.Controller2D(self._leader_csv_file)

    #     # Send a control command to proceed to next iteration.
    #     # This mainly applies for simulations that are in synchronous mode.
    #     self._send_leader_cmd(throttle=0.0, steer=0.0, brake=1.0)

    def _run_follower(self):
        '''
        Function to run the follower vehicle in the timer loop.
        '''

        # Check if the follower is ready.
        # This is set in the function _set_up_follower_waypoints()
        if not self._follower_ready:
            return

        if self._follower_reached_the_end:
            return
        
        if int(self._follower_current_x) == 0 and int(self._follower_current_y) == 0:
            return

        self.get_logger().info("Running follower")
        # ---------------------------------------------
        # Gather current data from the CARLA server
        # ---------------------------------------------

        # Update pose, timestamp
        current_x = self._follower_current_x
        current_y = self._follower_current_y
        current_yaw = self._follower_current_yaw
        current_speed = self._follower_current_speed
        # current_timestamp = self._current_time
        
        length = 0.0
        # Shift x, y coordinates
        if self._control_method == 'PurePursuit':
            length = -1.5
        elif self._control_method in ['Stanley', 'MPC']:
            length = 1.5
            
        # self.get_logger().info(f"Control Method: {self._control_method}")

        self.get_logger().info(f"X, Y, Yaw, Length: {current_x}, {current_y}, {current_yaw}, {length}")
        current_x, current_y = self._follower_controller.get_shifted_coordinate(current_x, current_y, current_yaw, length)

        self.get_logger().info(f"X, Y: {current_x}, {current_y}")
        
        
        # ---------------------------------------------
        # Controller update (this uses the controller2d.py implementation)
        # ---------------------------------------------

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
        closest_distance = np.linalg.norm(np.array([
            self._waypoints_follower_np[self._follower_closest_index, 0] - current_x,
            self._waypoints_follower_np[self._follower_closest_index, 1] - current_y]))

        new_distance = closest_distance
        new_index = self._follower_closest_index
        while new_distance <= closest_distance:
            closest_distance = new_distance
            self._follower_closest_index = new_index
            new_index += 1
            if new_index >= self._waypoints_follower_np.shape[0]:  # End of path
                break
            new_distance = np.linalg.norm(np.array([
                self._waypoints_follower_np[new_index, 0] - current_x,
                self._waypoints_follower_np[new_index, 1] - current_y]))
        new_distance = closest_distance
        new_index = self._follower_closest_index
        while new_distance <= closest_distance:
            closest_distance = new_distance
            self._follower_closest_index = new_index
            new_index -= 1
            if new_index < 0:  # Beginning of path
                break
            new_distance = np.linalg.norm(np.array([
                self._waypoints_follower_np[new_index, 0] - current_x,
                self._waypoints_follower_np[new_index, 1] - current_y]))

        # Once the closest index is found, return the path that has 1
        # waypoint behind and X waypoints ahead, where X is the index
        # that has a lookahead distance specified by
        # INTERP_LOOKAHEAD_DISTANCE
        waypoint_subset_first_index = self._follower_closest_index - 1
        if waypoint_subset_first_index < 0:
            waypoint_subset_first_index = 0

        waypoint_subset_last_index = self._follower_closest_index
        total_distance_ahead = 0
        while total_distance_ahead < self.INTERP_LOOKAHEAD_DISTANCE:
            total_distance_ahead += self._follower_wp_distance[waypoint_subset_last_index]
            waypoint_subset_last_index += 1
            if waypoint_subset_last_index >= self._waypoints_follower_np.shape[0]:
                waypoint_subset_last_index = self._waypoints_follower_np.shape[0] - 1
                break

        # Use the first and last waypoint subset indices into the hash
        # table to obtain the first and last indicies for the interpolated
        # list. Update the interpolated waypoints to the controller
        # for the next controller update.
        new_waypoints = \
            self._follower_wp_interp[self._follower_wp_interp_hash[waypoint_subset_first_index]:
                                     self._follower_wp_interp_hash[waypoint_subset_last_index] + 1]
        self._follower_controller.update_waypoints(new_waypoints)

        # Update the other controller values and controls
        self._follower_controller.update_values(current_x, current_y, current_yaw,
                                                current_speed, new_distance)

        self._follower_controller.update_controls()
        cmd_throttle, cmd_steer, cmd_brake = self._follower_controller.get_commands()

        self.get_logger().info(f"Throttle, Steer, Brake: {cmd_throttle}, {cmd_steer}, {cmd_brake}")
        # Output controller command to CARLA server
        self._send_follower_cmd(throttle=float(cmd_throttle), steer=float(cmd_steer), brake=float(cmd_brake))

        # Find if reached the end of waypoint. If the car is within
        # DIST_THRESHOLD_TO_LAST_WAYPOINT to the last waypoint,
        # the simulation will end.
        dist_to_last_waypoint = np.linalg.norm(np.array([
            self._follower_waypoints[-1][0] - current_x,
            self._follower_waypoints[-1][1] - current_y]))
        if dist_to_last_waypoint < self.DIST_THRESHOLD_TO_LAST_WAYPOINT:
            self._follower_reached_the_end = True
            self._send_follower_cmd(throttle=0.0, steer=0.0, brake=1.0)

    # def _run_leader(self):
    #     '''
    #     Function to run the leader vehicle in the timer loop
    #     '''
    #     ###
    #     # Controller update (this uses the controller2d.py implementation)
    #     ###

    #     # To reduce the amount of waypoints sent to the controller,
    #     # provide a subset of waypoints that are within some
    #     # lookahead distance from the closest point to the car. Provide
    #     # a set of waypoints behind the car as well.

    #     # Find closest waypoint index to car. First increment the index
    #     # from the previous index until the new distance calculations
    #     # are increasing. Apply the same rule decrementing the index.
    #     # The final index should be the closest point (it is assumed that
    #     # the car will always break out of instability points where there
    #     # are two indices with the same minimum distance, as in the
    #     # center of a circle)

    #     current_x = self._leader_current_x
    #     current_y = self._leader_current_y
    #     current_yaw = self._leader_current_yaw
    #     current_speed = self._leader_current_speed
    #     current_timestamp = self._current_time

    #     closest_distance = np.linalg.norm(np.array([
    #         self._waypoints_leader_np[self._leader_closest_index, 0] - current_x,
    #         self._waypoints_leader_np[self._leader_closest_index, 1] - current_y]))
    #     new_distance = closest_distance
    #     new_index = self._leader_closest_index
    #     while new_distance <= closest_distance:
    #         closest_distance = new_distance
    #         self._leader_closest_index = new_index
    #         new_index += 1
    #         if new_index >= self._waypoints_leader_np.shape[0]:  # End of path
    #             break
    #         new_distance = np.linalg.norm(np.array([
    #             self._waypoints_leader_np[new_index, 0] - current_x,
    #             self._waypoints_leader_np[new_index, 1] - current_y]))
    #     new_distance = closest_distance
    #     new_index = self._leader_closest_index
    #     while new_distance <= closest_distance:
    #         closest_distance = new_distance
    #         self._leader_closest_index = new_index
    #         new_index -= 1
    #         if new_index < 0:  # Beginning of path
    #             break
    #         new_distance = np.linalg.norm(np.array([
    #             self._waypoints_leader_np[new_index, 0] - current_x,
    #             self._waypoints_leader_np[new_index, 1] - current_y]))

    #     # Once the closest index is found, return the path that has 1
    #     # waypoint behind and X waypoints ahead, where X is the index
    #     # that has a lookahead distance specified by
    #     # INTERP_LOOKAHEAD_DISTANCE
    #     waypoint_subset_first_index = self._leader_closest_index - 1
    #     if waypoint_subset_first_index < 0:
    #         waypoint_subset_first_index = 0

    #     waypoint_subset_last_index = self._leader_closest_index
    #     total_distance_ahead = 0
    #     while total_distance_ahead < self.INTERP_LOOKAHEAD_DISTANCE:
    #         total_distance_ahead += self._leader_wp_distance[waypoint_subset_last_index]
    #         waypoint_subset_last_index += 1
    #         if waypoint_subset_last_index >= self._waypoints_leader_np.shape[0]:
    #             waypoint_subset_last_index = self._waypoints_leader_np.shape[0] - 1
    #             break

    #     # Use the first and last waypoint subset indices into the hash
    #     # table to obtain the first and last indicies for the interpolated
    #     # list. Update the interpolated waypoints to the controller
    #     # for the next controller update.
    #     new_waypoints = \
    #         self._leader_wp_interp[self._leader_wp_interp_hash[waypoint_subset_first_index]:
    #                                self._leader_wp_interp_hash[waypoint_subset_last_index] + 1]
    #     self._leader_controller.update_waypoints(new_waypoints)

    #     # Update the other controller values and controls
    #     self._leader_controller.update_values(current_x, current_y, current_yaw,
    #                                           current_speed,
    #                                           current_timestamp)
    #     self._leader_controller.update_controls()
    #     cmd_throttle, cmd_steer, cmd_brake = self._leader_controller.get_commands()

    #     # Output controller command to CARLA server
    #     self._send_leader_cmd(throttle=float(cmd_throttle), steer=float(cmd_steer), brake=float(cmd_brake))

    #     # Find if reached the end of waypoint. If the car is within
    #     # DIST_THRESHOLD_TO_LAST_WAYPOINT to the last waypoint,
    #     # the simulation will end.
    #     dist_to_last_waypoint = np.linalg.norm(np.array([
    #         self._leader_csv_file[-1][0] - current_x,
    #         self._leader_csv_file[-1][1] - current_y]))
    #     if dist_to_last_waypoint < self.DIST_THRESHOLD_TO_LAST_WAYPOINT:
    #         self._leader_reached_the_end = True
    #     if self._leader_reached_the_end:
    #         self.get_logger().info("Reached the end of path.")
    #         self._send_leader_cmd(throttle=0.0, steer=0.0, brake=1.0)

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

    def _clock_cb(self, msg: Clock):
        '''
        /clock topic callback function

        Args:
            msg (Clock): Clock message
        '''
        # self.get_logger().info('_clock_cb', throttle_duration_sec=3)
        self._current_time = msg.clock.sec
        # self.get_logger().info(f'Current time: {self._current_time}', throttle_duration_sec=2)

    def _follower_status_cb(self, msg: CarlaEgoVehicleStatus):
        '''
        /carla/ego_vehicle/vehicle_status topic callback function

        Args:
            msg (CarlaEgoVehicleStatus): CarlaEgoVehicleStatus message
        '''

        self._follower_current_speed = msg.velocity
        # self.get_logger().info(self._follower_current_speed, throttle_duration_sec=3)
        # self.get_logger().info(f'Throttle: {msg.control.throttle}')
        # self.get_logger().info(f'Steer: {msg.control.steer}')
        # self.get_logger().info(f'Break: {msg.control.brake}')

        # We are collecting ground truth data
        if self._is_ground_truth:
            if self._waypoints_within_distance:
                # self.get_logger().info(f'Follower velocity: {self._follower_current_speed}')
                self._output += f'{self._follower_current_speed}'
                # self.get_logger().info(f'Output: {self._output}')
                self._waypoint_log_list.append(self._output)
                self._output = ""

                if self._follower_current_x > 88.0 and self._follower_current_y < -176.0:
                    self._log_ground_truth()

    def _log_ground_truth(self):
        if not os.path.exists(self.GROUND_TRUTH_OUTPUT_FOLDER):
            os.makedirs(self.GROUND_TRUTH_OUTPUT_FOLDER)

        file_name = os.path.join(self.GROUND_TRUTH_OUTPUT_FOLDER, self.FOLLOWER_OUTPUT_FILE)
        with open(file_name, 'w') as f:
            for item in self._waypoint_log_list:
                f.write("%s\n" % item)
        self.get_logger().info(f'Ground truth logged to {file_name}')

        # close file
        f.close()

    def _leader_status_cb(self, msg: CarlaEgoVehicleStatus):
        '''
        /carla/ego_vehicle/vehicle_status topic callback function

        Args:
            msg (CarlaEgoVehicleStatus): CarlaEgoVehicleStatus message
        '''
        self._leader_current_speed = msg.velocity
        # self.get_logger().info(f'Leader velocity: {self._leader_current_speed}',
        #                        throttle_duration_sec=2)

    def _follower_waypoints_cb(self, msg: Path):
        '''
        /carla/ego_vehicle/waypoints topic callback function

        Args:
            msg (Path): Path message
        '''
        if not self._follower_waypoints_acquired:
            for pose in msg.poses:
                # self.get_logger().info(f'Waypoint Pose: [{pose.pose.position.x}, {pose.pose.position.y}, {pose.pose.position.z}] [{pose.pose.orientation.x}, {pose.pose.orientation.y}, {pose.pose.orientation.z}, {pose.pose.orientation.w}]')
                # yaw = tf_transformations.euler_from_quaternion( [pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w])[2]
                # x, y, velocity
                self._follower_waypoints.append([pose.pose.position.x, pose.pose.position.y, 15.0])
            self._waypoints_follower_np = np.array(self._follower_waypoints)
            self._follower_waypoints_acquired = True
            # self.get_logger().info('Follower waypoints acquired')
            self._set_up_follower_waypoints()

    def _compute_distance(self, x1, y1, x2, y2):
        '''
        Compute distance between two points

        Args:
            x1 (float): x coordinate of first point
            y1 (float): y coordinate of first point
            x2 (float): x coordinate of second point
            y2 (float): y coordinate of second point

        Returns:
            float: distance between the two points
        '''
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def _follower_odometry_cb(self, msg: Odometry):
        '''
        /carla/ego_vehicle/odometry topic callback function

        Args:
            msg (Odometry): Odometry message
        '''

        if self._is_ground_truth:
            self._waypoints_within_distance = False
            if self._compute_distance(
                    self._follower_log_x, self._follower_log_y, msg.pose.pose.position.x, msg.pose.pose.position.y) >= 1.0:
                self._follower_log_x = msg.pose.pose.position.x
                self._follower_log_y = msg.pose.pose.position.y
                self._waypoints_within_distance = True
                self._output += f'{self._follower_log_x }, {self._follower_log_y}, '

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
        # self.get_logger().info(f'\N{dog} {self._follower_current_yaw}')

    def _leader_odometry_cb(self, msg: Odometry):
        '''
        /carla/hero/odometry topic callback function

        Args:
            msg (Odometry): Odometry message
        '''

        # self.get_logger().info('_leader_odometry_cb')
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
        # self.get_logger().info(f'Leader throttle: {throttle}')
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
