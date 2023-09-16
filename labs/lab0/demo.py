"""
Date: 08/26/2021

Purpose: This script creates an ArmController and uses it to command the arm's
joint positions and gripper.

Try changing the target position to see what the arm does!

"""

import sys
import rospy
import numpy as np
from math import pi

from core.interfaces import ArmController

rospy.init_node('demo')

arm = ArmController()
arm.set_arm_speed(0.2)

# arm.close_gripper()

# q = arm.neutral_position()
# arm.safe_move_to_position(q)

arm.open_gripper()

""" Configurations """
# Straight up:
# q = np.array([0, 0, 0, -0.07, 0, 1.57, 0.785])

# One thousand years of death:
# q = np.array([0, -1.76280, 0, -3, 0, 3.7, -0.785])

# Look over shoulder:
# q = np.array([0.7, 0.7, 0.7, -0.07, 0, 1.57, 0.7])

# Bowing:
q = np.array([-1.57, 1.57, 0, -0.07, 0, 0.7, 0.785])


arm.safe_move_to_position(q)

arm.close_gripper()
