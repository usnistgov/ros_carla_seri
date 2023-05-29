#!/usr/bin/env python3
'''
Main for Carla custom node.
'''

import rclpy
from rclpy.executors import MultiThreadedExecutor
from commander_py.commander_interface import VehicleCommanderInterface

__all__ = ()

def main(args=None):
    '''
    Main function for the ego vehicle.
    '''
    rclpy.init(args=args)
    node = VehicleCommanderInterface()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except rclpy.executors.ExternalShutdownException:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
