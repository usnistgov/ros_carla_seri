#!/usr/bin/env python
#
# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
"""
Rosbridge class:

Class that handle communication between CARLA and ROS
"""

import os
import sys
import csv
import numpy as np
from distutils.version import LooseVersion
from threading import Thread, Lock, Event
import yaml
import pkg_resources
from launch_ros.substitutions import FindPackageShare
try:
    import queue
except ImportError:
    import Queue as queue
import carla
import ros_compatibility as roscomp
from ros_compatibility.node import CompatibleNode

# from carla_ros_bridge.actor import Actor
from carla_ros_bridge.actor_factory import ActorFactory
from carla_ros_bridge.carla_status_publisher import CarlaStatusPublisher
from carla_ros_bridge.debug_helper import DebugHelper
from carla_ros_bridge.ego_vehicle import EgoVehicle
from carla_ros_bridge.world_info import WorldInfo

from carla_msgs.msg import CarlaControl, CarlaWeatherParameters
from carla_msgs.srv import SpawnObject, DestroyObject, GetBlueprints
from rosgraph_msgs.msg import Clock
from launch_ros.substitutions import FindPackageShare


class CarlaRosBridge(CompatibleNode):

    """
    Carla Ros bridge
    """

    with open(os.path.join(os.path.dirname(__file__), "CARLA_VERSION")) as f:
        CARLA_VERSION = f.read()[:-1]

    # in synchronous mode, if synchronous_mode_wait_for_vehicle_control_command is True,
    # wait for this time until a next tick is triggered.
    VEHICLE_CONTROL_TIMEOUT = 1.

    def __init__(self):
        """
        Constructor

        :param carla_world: carla world object
        :type carla_world: carla.World
        :param params: dict of parameters, see settings.yaml
        :type params: dict
        """
        super(CarlaRosBridge, self).__init__("ros_bridge_node")
        self._output = ""

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

    def set_weather(self):
        """
        Set CARLA weather according to the weather defined in the seri.yaml file

        :param weather: weather object
        :type weather: carla.WeatherParameters
        """
        pkg_share = FindPackageShare(package='carla_common').find('carla_common')
        print(pkg_share)
        config_file_name = "seri.yaml"
        config_file_path = os.path.join(pkg_share, 'config', config_file_name)

        data = self.read_yaml(config_file_path)

        try:
            weather = data["weather"]
        except KeyError:
            self.get_logger().info(
                "YAML configuration file does not contain a 'weather' section")
            sys.exit(1)

        msg = CarlaWeatherParameters()
        msg.cloudiness = weather.get("cloudiness", 0.0)
        msg.precipitation = weather.get("precipitation", 0.0)
        msg.precipitation_deposits = weather.get("precipitation_deposits", 0.0)
        msg.wind_intensity = weather.get("wind_intensity", 0.0)
        msg.fog_density = weather.get("fog_density", 0.0)
        msg.fog_distance = weather.get("fog_distance", 0.0)
        msg.wetness = weather.get("wetness", 0.0)
        msg.sun_azimuth_angle = weather.get("sun_azimuth_angle", 0.0)
        msg.sun_altitude_angle = weather.get("sun_altitude_angle", 0.0)
        self.carla_weather_publisher.publish(msg)

    # pylint: disable=attribute-defined-outside-init

    def initialize_bridge(self, carla_world, params):
        """
        Initialize the bridge
        """
        self.parameters = params
        self.carla_world = carla_world

        self.ros_timestamp = roscomp.ros_timestamp()
        self.callback_group = roscomp.callback_groups.ReentrantCallbackGroup()

        self.synchronous_mode_update_thread = None
        self.shutdown = Event()

        self.carla_settings = carla_world.get_settings()
        if not self.parameters["passive"]:
            # workaround: settings can only applied within non-sync mode
            if self.carla_settings.synchronous_mode:
                self.carla_settings.synchronous_mode = False
                carla_world.apply_settings(self.carla_settings)

            # self.loginfo("synchronous_mode: {}".format(
            #     self.parameters["synchronous_mode"]))
            # output
            self._output += f"synchronous_mode: {str(self.parameters['synchronous_mode'])}\n"

            self.carla_settings.synchronous_mode = self.parameters["synchronous_mode"]
            # self.loginfo("fixed_delta_seconds: {}".format(
            #     self.parameters["fixed_delta_seconds"]))
            # output
            self._output += f"fixed_delta_seconds: {str(self.parameters['fixed_delta_seconds'])}\n"

            self.carla_settings.fixed_delta_seconds = self.parameters["fixed_delta_seconds"]
            carla_world.apply_settings(self.carla_settings)

        self.loginfo("Parameters:")
        # output
        self._output += "Parameters\n"

        for key in self.parameters:
            # self.loginfo("  {}: {}".format(key, self.parameters[key]))
            # output
            self._output += f"  {key}: {str(self.parameters[key])}\n"

        # active sync mode in the ros bridge only if CARLA world is configured in sync mode and
        # passive mode is not enabled.
        self.sync_mode = self.carla_settings.synchronous_mode and not self.parameters["passive"]
        if self.carla_settings.synchronous_mode and self.parameters["passive"]:
            # self.loginfo(
            # "Passive mode is enabled and CARLA world is configured in synchronous mode. This configuration requires another client ticking the CARLA world.")
            self._output += "Passive mode is enabled and CARLA world is configured in synchronous mode. This configuration requires another client ticking the CARLA world.\n"
        self.carla_control_queue = queue.Queue()

        # actor factory
        self.actor_factory = ActorFactory(self, carla_world, self.sync_mode)

        # add world info
        self.world_info = WorldInfo(carla_world=self.carla_world, node=self)
        # add debug helper
        self.debug_helper = DebugHelper(carla_world.debug, self)

        # Communication topics
        self.clock_publisher = self.new_publisher(Clock, 'clock', 10)

        self.status_publisher = CarlaStatusPublisher(
            self.carla_settings.synchronous_mode,
            self.carla_settings.fixed_delta_seconds,
            self)

        # for waiting for ego vehicle control commands in synchronous mode,
        # their ids are maintained in a list.
        # Before tick(), the list is filled and the loop waits until the list is empty.
        self._all_vehicle_control_commands_received = Event()
        self._expected_ego_vehicle_control_command_ids = []
        self._expected_ego_vehicle_control_command_ids_lock = Lock()

        if self.sync_mode:
            self.carla_run_state = CarlaControl.PLAY

            self.carla_control_subscriber = \
                self.new_subscription(CarlaControl, "/carla/control",
                                      lambda control: self.carla_control_queue.put(control.command),
                                      qos_profile=10, callback_group=self.callback_group)

            self.synchronous_mode_update_thread = Thread(
                target=self._synchronous_mode_update)
            self.synchronous_mode_update_thread.start()
        else:
            self.timestamp_last_run = 0.0

            self.actor_factory.start()

            # register callback to update actors
            self.on_tick_id = self.carla_world.on_tick(self._carla_time_tick)

        # services configuration.
        self._registered_actors = []
        self.spawn_object_service = self.new_service(SpawnObject, "/carla/spawn_object",
                                                     self.spawn_object)
        self.destroy_object_service = self.new_service(DestroyObject, "/carla/destroy_object",
                                                       self.destroy_object)

        self.get_blueprints_service = self.new_service(GetBlueprints, "/carla/get_blueprints",
                                                       self.get_blueprints, callback_group=self.callback_group)

        self.carla_weather_subscriber = \
            self.new_subscription(CarlaWeatherParameters, "/carla/weather_control",
                                  self.on_weather_changed, qos_profile=10, callback_group=self.callback_group)

        self.carla_weather_publisher = self.new_publisher(CarlaWeatherParameters, "/carla/weather_control", 10)

        # retrieve weather information from seri.yaml and publish it
        self.set_weather()


        # ---------------------------------------------
        # Draw waypoints for the leader in the world
        # ---------------------------------------------

        # waypoints_leader_np = None
        # carla_common_pkg = FindPackageShare(package='carla_common').find('carla_common')
        # waypoints_leader_file_name = "waypoints_leader.txt"
        # waypoints_leader_file_path = os.path.join(carla_common_pkg, 'config', waypoints_leader_file_name)
        # try:
        #     with open(waypoints_leader_file_path, encoding="utf8") as stream:
        #         leader_csv_file = list(csv.reader(stream, delimiter=',', quoting=csv.QUOTE_NONNUMERIC))
        #         waypoints_leader_np = np.array(leader_csv_file)
        # except FileNotFoundError:
        #     self.get_logger().error(f"The file {waypoints_leader_file_path} does not exist.")

        # if waypoints_leader_np is not None:
        #     for i in range(1, waypoints_leader_np.shape[0]):
        #         waypoint = carla.Waypoint(carla.Location(x=waypoints_leader_np[i, 0], y=waypoints_leader_np[i, 1], z=0.000000))
        #         carla_world.debug.draw_string(waypoint.transform.location, 'o', draw_shadow=False, color=carla.Color(
        #             r=255, g=255, b=255), life_time=0.2, persistent_lines=True)

        # print output
        self.loginfo(self._output)

    def spawn_object(self, req, response=None):
        '''
        Spawn objects in the world

        Args:
            req (carla_msgs/srv/SpawnObject): Client request
            response (carla_msgs/srv/SpawnObject, optional): Server response Defaults to None.

        Returns:
            _type_: Server response
        '''
        response = roscomp.get_service_response(SpawnObject)
        if not self.shutdown.is_set():
            try:
                id_ = self.actor_factory.spawn_actor(req)
                self._registered_actors.append(id_)
                response.id = id_
            except Exception as e:
                self.logwarn("Error spawning object '{}': {}".format(req.type, e))
                response.id = -1
                response.error_string = str(e)
        else:
            response.id = -1
            response.error_string = 'Bridge is shutting down, object will not be spawned.'
        return response

    def destroy_object(self, req, response=None):
        response = roscomp.get_service_response(DestroyObject)
        destroyed_actors = self.actor_factory.destroy_actor(req.id)
        response.success = bool(destroyed_actors)
        for actor in destroyed_actors:
            if actor in self._registered_actors:
                self._registered_actors.remove(actor)
        return response

    def get_blueprints(self, req):
        response = roscomp.get_service_response(GetBlueprints)
        if req.filter:
            bp_filter = req.filter
        else:
            bp_filter = "*"

        response.blueprints = [
            bp.id for bp in self.carla_world.get_blueprint_library().filter(bp_filter)]
        response.blueprints.extend(self.actor_factory.get_pseudo_sensor_types())
        response.blueprints.sort()
        return response

    def on_weather_changed(self, msg: CarlaWeatherParameters):
        """
        Callback on new weather parameters
        :return:
        """
        if not self.carla_world:
            return
        # self.loginfo("Applying weather parameters...")
        self._output += "Applying weather parameters...\n"

        weather = carla.WeatherParameters()
        weather.cloudiness = msg.cloudiness
        weather.precipitation = msg.precipitation
        weather.precipitation_deposits = msg.precipitation_deposits
        weather.wind_intensity = msg.wind_intensity
        weather.fog_density = msg.fog_density
        weather.fog_distance = msg.fog_distance
        weather.wetness = msg.wetness
        weather.sun_azimuth_angle = msg.sun_azimuth_angle
        weather.sun_altitude_angle = msg.sun_altitude_angle
        self.carla_world.set_weather(weather)

    def process_run_state(self):
        """
        process state changes
        """
        command = None

        # get last command
        while not self.carla_control_queue.empty():
            command = self.carla_control_queue.get()

        while command is not None and roscomp.ok():
            self.carla_run_state = command

            if self.carla_run_state == CarlaControl.PAUSE:
                # wait for next command
                self.loginfo("State set to PAUSED")
                self.status_publisher.set_synchronous_mode_running(False)
                command = self.carla_control_queue.get()
            elif self.carla_run_state == CarlaControl.PLAY:
                self.loginfo("State set to PLAY")
                self.status_publisher.set_synchronous_mode_running(True)
                return
            elif self.carla_run_state == CarlaControl.STEP_ONCE:
                self.loginfo("Execute single step.")
                self.status_publisher.set_synchronous_mode_running(True)
                self.carla_control_queue.put(CarlaControl.PAUSE)
                return

    def _synchronous_mode_update(self):
        """
        execution loop for synchronous mode
        """
        while not self.shutdown.is_set() and roscomp.ok():
            self.process_run_state()

            if self.parameters['synchronous_mode_wait_for_vehicle_control_command']:
                # fill list of available ego vehicles
                self._expected_ego_vehicle_control_command_ids = []
                with self._expected_ego_vehicle_control_command_ids_lock:
                    for actor_id, actor in self.actor_factory.actors.items():
                        if isinstance(actor, EgoVehicle):
                            self._expected_ego_vehicle_control_command_ids.append(
                                actor_id)

            self.actor_factory.update_available_objects()
            frame = self.carla_world.tick()

            world_snapshot = self.carla_world.get_snapshot()

            self.status_publisher.set_frame(frame)
            self.update_clock(world_snapshot.timestamp)
            self.logdebug("Tick for frame {} returned. Waiting for sensor data...".format(
                frame))
            self._update(frame, world_snapshot.timestamp.elapsed_seconds)
            self.logdebug("Waiting for sensor data finished.")

            if self.parameters['synchronous_mode_wait_for_vehicle_control_command']:
                # wait for all ego vehicles to send a vehicle control command
                if self._expected_ego_vehicle_control_command_ids:
                    if not self._all_vehicle_control_commands_received.wait(CarlaRosBridge.VEHICLE_CONTROL_TIMEOUT):
                        self.logwarn("Timeout ({}s) while waiting for vehicle control commands. "
                                     "Missing command from actor ids {}".format(CarlaRosBridge.VEHICLE_CONTROL_TIMEOUT,
                                                                                self._expected_ego_vehicle_control_command_ids))
                    self._all_vehicle_control_commands_received.clear()

    def _carla_time_tick(self, carla_snapshot):
        """
        Private callback registered at carla.World.on_tick()
        to trigger cyclic updates.

        After successful locking the update mutex
        (only perform trylock to respect bridge processing time)
        the clock and the children are updated.
        Finally the ROS messages collected to be published are sent out.

        :param carla_timestamp: the current carla time
        :type carla_timestamp: carla.Timestamp
        :return:
        """
        if not self.shutdown.is_set():
            if self.timestamp_last_run < carla_snapshot.timestamp.elapsed_seconds:
                self.timestamp_last_run = carla_snapshot.timestamp.elapsed_seconds
                self.update_clock(carla_snapshot.timestamp)
                self.status_publisher.set_frame(carla_snapshot.frame)
                self._update(carla_snapshot.frame,
                             carla_snapshot.timestamp.elapsed_seconds)

    def _update(self, frame_id, timestamp):
        """
        update all actors
        :return:
        """
        self.world_info.update(frame_id, timestamp)
        self.actor_factory.update_actor_states(frame_id, timestamp)

    def _ego_vehicle_control_applied_callback(self, ego_vehicle_id):
        if not self.sync_mode or \
                not self.parameters['synchronous_mode_wait_for_vehicle_control_command']:
            return
        with self._expected_ego_vehicle_control_command_ids_lock:
            if ego_vehicle_id in self._expected_ego_vehicle_control_command_ids:
                self._expected_ego_vehicle_control_command_ids.remove(
                    ego_vehicle_id)
            else:
                self.logwarn(
                    "Unexpected vehicle control command received from {}".format(ego_vehicle_id))
            if not self._expected_ego_vehicle_control_command_ids:
                self._all_vehicle_control_commands_received.set()

    def update_clock(self, carla_timestamp):
        """
        perform the update of the clock

        :param carla_timestamp: the current carla time
        :type carla_timestamp: carla.Timestamp
        :return:
        """
        if roscomp.ok():
            self.ros_timestamp = roscomp.ros_timestamp(carla_timestamp.elapsed_seconds, from_sec=True)
            self.clock_publisher.publish(Clock(clock=self.ros_timestamp))
            # self.loginfo(f"Clock: {Clock(clock=self.ros_timestamp)}")

    def destroy(self):
        """
        Function to destroy this object.

        :return:
        """
        self.loginfo("Shutting down...")
        self.shutdown.set()
        if not self.sync_mode:
            if self.on_tick_id:
                self.carla_world.remove_on_tick(self.on_tick_id)
            self.actor_factory.thread.join()
        else:
            self.synchronous_mode_update_thread.join()
        self.loginfo("Object update finished.")
        self.debug_helper.destroy()
        self.status_publisher.destroy()
        self.destroy_service(self.spawn_object_service)
        self.destroy_service(self.destroy_object_service)
        self.destroy_subscription(self.carla_weather_subscriber)
        self.carla_control_queue.put(CarlaControl.STEP_ONCE)

        for uid in self._registered_actors:
            self.actor_factory.destroy_actor(uid)
        self.actor_factory.update_available_objects()
        self.actor_factory.clear()
        super(CarlaRosBridge, self).destroy()


def main(args=None):
    """
    main function for carla simulator ROS bridge
    maintaining the communication client and the CarlaBridge object
    """
    roscomp.init("bridge", args=args)

    carla_bridge = None
    carla_world = None
    carla_client = None
    executor = None
    parameters = {}

    executor = roscomp.executors.MultiThreadedExecutor()
    carla_bridge = CarlaRosBridge()
    executor.add_node(carla_bridge)

    roscomp.on_shutdown(carla_bridge.destroy)

    parameters['host'] = carla_bridge.get_param('host', 'localhost')
    parameters['port'] = carla_bridge.get_param('port', 2000)
    parameters['timeout'] = carla_bridge.get_param('timeout', 2)
    parameters['passive'] = carla_bridge.get_param('passive', False)
    parameters['synchronous_mode'] = carla_bridge.get_param('synchronous_mode', True)
    parameters['synchronous_mode_wait_for_vehicle_control_command'] = carla_bridge.get_param(
        'synchronous_mode_wait_for_vehicle_control_command', False)
    parameters['fixed_delta_seconds'] = carla_bridge.get_param('fixed_delta_seconds',
                                                               0.05)
    parameters['register_all_sensors'] = carla_bridge.get_param('register_all_sensors', True)
    parameters['town'] = carla_bridge.get_param('town', 'Town01')
    role_name = carla_bridge.get_param('ego_vehicle_role_name',
                                       ["hero", "ego_vehicle", "hero1", "hero2", "hero3"])
    parameters["ego_vehicle"] = {"role_name": role_name}

    carla_bridge.loginfo(f"Trying to connect to {parameters['host']}:{parameters['port']}")

    try:
        carla_client = carla.Client(
            host=parameters['host'],
            port=parameters['port'])
        carla_client.set_timeout(parameters['timeout'])

        # check carla version
        dist = pkg_resources.get_distribution("carla")
        if LooseVersion(dist.version) != LooseVersion(CarlaRosBridge.CARLA_VERSION):
            carla_bridge.logfatal("CARLA python module version {} required. Found: {}".format(
                CarlaRosBridge.CARLA_VERSION, dist.version))
            sys.exit(1)

        if LooseVersion(carla_client.get_server_version()) != \
           LooseVersion(carla_client.get_client_version()):
            carla_bridge.logwarn(
                "Version mismatch detected: You are trying to connect to a simulator that might be incompatible with this API. Client API version: {}. Simulator API version: {}"
                .format(carla_client.get_client_version(),
                        carla_client.get_server_version()))

        carla_world = carla_client.get_world()

        if "town" in parameters and not parameters['passive']:
            if parameters["town"].endswith(".xodr"):
                carla_bridge.loginfo(f"""Loading opendrive world from file '{str(parameters["town"])}'""")

                pkg_share = FindPackageShare(package='seri_maps').find('seri_maps')
                # carla_bridge.loginfo(f"MAP package: '{str(pkg_share)}'")

                # Set the path to the world file
                map_file_name = str(parameters["town"])
                map_file = os.path.join(pkg_share, 'maps', map_file_name)
                # carla_bridge.loginfo(map_file)

                with open(file=map_file) as od_file:
                    data = od_file.read()
                carla_world = carla_client.generate_opendrive_world(str(data))
            else:
                if carla_world.get_map().name != parameters["town"]:
                    # self._output += f"Loading town {str(parameters['town'])} (previous: {carla_world.get_map().name}).\n"
                    carla_bridge.loginfo("Loading town '{}' (previous: '{}').".format(
                        parameters["town"], carla_world.get_map().name))
                    carla_world = carla_client.load_world(parameters["town"])
                    # spawn_points = carla_world.get_map().get_spawn_points()
                    # for point in spawn_points:
                    #     carla_bridge.loginfo(f"Position: {point.location.x}, {point.location.y}, {point.location.z}")
                    #     carla_bridge.loginfo(f"Rotation: {point.rotation.roll}, {point.rotation.pitch}, {point.rotation.yaw}")

            carla_world.tick()

        carla_bridge.initialize_bridge(carla_client.get_world(), parameters)

        carla_bridge.spin()

    except (IOError, RuntimeError) as e:
        carla_bridge.logerr("Error: {}".format(e))
    except KeyboardInterrupt:
        pass
    finally:
        roscomp.shutdown()
        del carla_world
        del carla_client


if __name__ == "__main__":
    main()
