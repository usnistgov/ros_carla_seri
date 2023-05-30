
import random
from agents.navigation.global_route_planner import GlobalRoutePlanner
import agents.navigation.controller as controller
import carla
import time
import math
import numpy as np


def spawn_vehicle():
    """
    This function spawn vehicles in the given spawn points. If no spawn 
    point is provided it spawns vehicle in this 
    position x=27.607,y=3.68402,z=0.02
    """
    # spawnPoint = carla.Transform(
    #     carla.Location(x=112.51, y=-197.75, z=0.02),
    #     carla.Rotation(pitch=0.0, yaw=0.0, roll=180.0))

    spawnPoint = carla.Transform(carla.Location(x=112.51, y=-197.75, z=0.275307),
                                 carla.Rotation(pitch=0.0, yaw=0.0, roll=0.000000))
    world = client.get_world()
    # blueprint_library = world.get_blueprint_library()
    # bp = blueprint_library.filter('vehicle.*')[3]
    # bp = blueprint_library.filter('vehicle.ford.ambulance')[0]

    # vehicle = world.spawn_actor(bp, spawnPoint)

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter("model3")[0]
    vehicle_transform = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(vehicle_bp, vehicle_transform)
    vehicle.set_simulate_physics(True)
    # vehicle.set_autopilot(arg.no_autopilot)
    vehicle.set_autopilot(False)

    # Add spectator camera
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_transform = carla.Transform(carla.Location(x=-10, z=10), carla.Rotation(-45, 0, 0))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    return vehicle, camera


def drive_through_plan(planned_route, vehicle, speed, PID, camera):
    """
    This function drives throught the planned_route with the speed passed in the argument
    
    """
    i = 0
    target = planned_route[0]
    while True:
        spectator = world.get_spectator()
        # transform = vehicle.get_transform()
        spectator.set_transform(camera.get_transform())

        # vehicle_loc= vehicle.get_location()
        # distance_v =find_dist_veh(vehicle_loc,target)
        vehicle_location = vehicle.get_location()  # this method get a location type object

        # location type is the attribute of transform type.
        # location type has three attribute: x, y, z
        # therefore, you only modify your formulation:
        distance = np.sqrt((target.transform.location.x - vehicle_location.x) ** 2 +
                           (target.transform.location.y - vehicle_location.y) ** 2)
        control = PID.run_step(speed, target)
        vehicle.apply_control(control)

        if i == (len(planned_route)-1):
            print("last waypoint reached")
            break

        if (distance < 3.5):
            control = PID.run_step(speed, target)
            vehicle.apply_control(control)
            i = i+1
            target = planned_route[i]

    control = PID.run_step(0, planned_route[len(planned_route)-1])
    vehicle.apply_control(control)


def find_dist(start, end):
    dist = math.sqrt((start.transform.location.x - end.transform.location.x) ** 2 +
                     (start.transform.location.y - end.transform.location.y) ** 2)

    return dist


def find_dist_veh(vehicle_loc, target):
    dist = math.sqrt((target.transform.location.x - vehicle_loc.x) **
                     2 + (target.transform.location.y - vehicle_loc.y)**2)

    return dist


def setup_PID(vehicle):
    """
    This function creates a PID controller for the vehicle passed to it 
    """

    args_lateral_dict = {
        'K_P': 1.95,
        'K_D': 0.2,
        'K_I': 0.07,
        'dt': 1.0 / 10.0
    }

    args_long_dict = {
        'K_P': 1,
        'K_D': 0.0,
        'K_I': 0.75,
        'dt': 1.0 / 10.0
    }

    PID = controller.VehiclePIDController(vehicle, args_lateral=args_lateral_dict, args_longitudinal=args_long_dict)

    return PID


client = carla.Client("localhost", 2000)
client.set_timeout(10)
world = client.load_world('Town01')


amap = world.get_map()
sampling_resolution = 3.0
grp = GlobalRoutePlanner(amap, sampling_resolution)
# grp.setup()
spawn_points = world.get_map().get_spawn_points()
# a = carla.Location(spawn_points[0].location)
a = carla.Location(112.51, -197.75, 0.3)
# b = carla.Location(spawn_points[100].location)
b = carla.Location(150.90, -197.75, 0.3)
w1 = grp.trace_route(a, b)

# world.debug.draw_point(a,color=carla.Color(r=255, g=0, b=0),size=1.0 ,life_time=120.0)
# world.debug.draw_point(b,color=carla.Color(r=255, g=0, b=0),size=1.0 ,life_time=120.0)

wps = []

for i in range(len(w1)):
    wps.append(w1[i][0])
    world.debug.draw_point(w1[i][0].transform.location, color=carla.Color(r=255, g=0, b=0), size=0.4, life_time=120.0)


vehicle, camera = spawn_vehicle()
print("vehicle spawned")


PID = setup_PID(vehicle)
print("PID setup")

speed = 20
drive_through_plan(wps, vehicle, speed, PID, camera)
