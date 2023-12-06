import sys, os
import rospy
import tf
import numpy as np
from math import pi
from time import perf_counter

from core.interfaces import ArmController
from lib.rrt import rrt

from lib.loadmap import loadmap
from copy import deepcopy
from lib.IK_position_null import IK


starts = [np.array([0, -1, 0, -2, 0, 1.57, 0]),
          np.array([0, 0.4, 0, -2.5, 0, 2.7, 0.707]),
          np.array([-0.7407, -0.264, -0.9245, -2.0476, -0.2632, 1.862, -0.8108]),
          np.array([0.01967448,  0.04546567, -0.02241544, -2.53605974,  0.03051571,  2.5404769, 0.72864662]),
         ]

goals = [np.array([-1.2, 1.57, 1.57, -2.07, -1.57, 1.57, 0.7]),
         np.array([1.9, 1.57, -1.57, -1.57, 1.57, 1.57, 0.707]),
         np.array([0.5696, 0.364, 0.108, -1.778, -0.00035, 2.118, 1.427]),
         np.array([-0.0178, -0.1318, 0.0198, -1.2949, 0.0298, 2.3562, 0.7534]),
         ]

mapNames = ["map1",
            "map2",
            "map3",
            "map4",
            "emptyMap"]
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("usage:\n\tpython rrt_demo.py 1\n\tpython rrt_demo.py 2\n\tpython rrt_demo.py 3 ...")
        exit()

    rospy.init_node('RRT')

    arm = ArmController()
    index = int(sys.argv[1])-1
    print("Running test "+sys.argv[1])
    print("Moving to Start Position")
    arm.safe_move_to_position(starts[index])

    dir_path = os.path.dirname(os.path.realpath(__file__))

    map_struct = loadmap(dir_path + "/../../maps/"+mapNames[index] + ".txt")
    # map_struct = loadmap(dir_path + "/../../maps/emptyMap.txt")
    print("Map = "+ mapNames[index])

    print("Starting to plan")
    start = perf_counter()
    path = rrt(deepcopy(map_struct), deepcopy(starts[index]), deepcopy(goals[index]))
    stop = perf_counter()
    dt = stop - start
    print("RRT took {time:2.2f} sec. Path is.".format(time=dt))
    print(np.round(path,4))
    input("Press Enter to Send Path to Arm")

    for joint_set in path:
        arm.safe_move_to_position(joint_set)
    print("Trajectory Complete!")
